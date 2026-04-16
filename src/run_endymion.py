# - System flow is based on the current Endymion architecture and project write-up:
#   DataHandler -> FeatureExtractor -> CraterPredictor -> HazardAssessor -> Pathfinder -> Evaluator.

# - Implementation contract follows the existing module/storage design already used in the project:
#   * DataHandler.save_derived(...) writes dem_m.npy / slope_deg.npy / roughness_rms.npy
#   * DataHandler.save_navigation(...) writes hazard.npy / cost.npy / path_rc.npy / nav_meta.json
#   * Evaluator then reads those artifacts and writes metrics.json

# - Design goal for this file:
#   keep one clean, official, reproducible pipeline runner separate from the benchmark runner
#   and separate from the presentation notebook.

# - Main ideas reused from the project files and notes:
#   * canonical ROI via DataHandler.canonical_roi(...)
#   * GeoTIFF-first DEM workflow
#   * crater catalogue cache/build path via CraterPredictor(catalogue_raster_v1)
#   * hazard -> cost via build_cost_from_hazard(...)
#   * corridor-aware Weighted A* via Pathfinder.find_path_corridor(...)
#   * post-run evaluation via Evaluator(...)
#
# - It supports both local GeoTIFF use and cache/download resolution.
# - It protects previous runs by auto-versioning repeated run names.
# - It validates ROI-local start/goal coordinates before pathfinding starts.

# Visual flow:
#
#   DEM / GeoTIFF
#       -> DataHandler
#       -> FeatureExtractor
#       -> CraterPredictor (optional)
#       -> HazardAssessor
#       -> build_cost_from_hazard
#       -> Pathfinder
#       -> DataHandler.save_navigation
#       -> Evaluator.save
#      -> pipeline_summary.json (with config, metrics, and important paths)
#
#   CLI arguments allow overriding all important config options, with sensible defaults for a canonical run.
#
#   The output directory structure is designed to be clean and navigable, with one run per subdirectory under the derived directory for the tile/ROI. Each run contains all relevant outputs and metadata, and a summary JSON for easy parsing.


# import statements are grouped into three sections:
# 1. Standard library imports   
# 2. Third-party imports
# 3. Local application imports (with a fallback strategy for flat src aliases)

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# This path setup allows the file to run both as:
# - python -m src.run_endymion
# - python src/run_endymion.py
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
REPO_ROOT = SRC_DIR.parent

for p in (REPO_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


# -----------------------------------------------------------------------------
# Import strategy
# -----------------------------------------------------------------------------
# The fallback below exists for cases where the package layout is slightly messy
# during cleanup. Normal use should still prefer:
#   python -m src.run_endymion
def _install_flat_src_aliases() -> None:
    src_pkg = sys.modules.setdefault("src", types.ModuleType("src"))

    for pkg_name in ["data", "features", "models", "planning", "evaluation"]:
        full_name = f"src.{pkg_name}"
        pkg = sys.modules.get(full_name)
        if pkg is None:
            pkg = types.ModuleType(full_name)
            sys.modules[full_name] = pkg
        setattr(src_pkg, pkg_name, pkg)

    alias_map = {
        "src.data.data_handler": "data.data_handler",
        "src.data.crater_raster": "data.crater_raster",
        "src.features.feature_extractor": "features.feature_extractor",
        "src.models.crater_predictor": "models.crater_predictor",
        "src.models.hazard_assessor": "models.hazard_assessor",
        "src.planning.pathfinder": "planning.pathfinder",
        "src.evaluation.evaluator": "evaluation.evaluator",
    }

    for alias, flat_name in alias_map.items():
        if alias in sys.modules:
            continue

        module = importlib.import_module(flat_name)
        sys.modules[alias] = module

        parent_name, attr_name = alias.rsplit(".", 1)
        parent = sys.modules[parent_name]
        setattr(parent, attr_name, module)


try:
    from src.data.data_handler import DataHandler, GeoTiffTileSpec, ROI
    from src.features.feature_extractor import FeatureExtractor
    from src.models.crater_predictor import CraterPredictor
    from src.models.hazard_assessor import HazardAssessor
    from src.planning.pathfinder import Pathfinder, build_cost_from_hazard
    from src.evaluation.evaluator import Evaluator
except ImportError:
    _install_flat_src_aliases()
    from src.data.data_handler import DataHandler, GeoTiffTileSpec, ROI
    from src.features.feature_extractor import FeatureExtractor
    from src.models.crater_predictor import CraterPredictor
    from src.models.hazard_assessor import HazardAssessor
    from src.planning.pathfinder import Pathfinder, build_cost_from_hazard
    from src.evaluation.evaluator import Evaluator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RC = Tuple[int, int]


@dataclass
class RunConfig:
    """Configuration for one official Endymion pipeline run."""

    # Data source / storage
    base_url: str = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp"
    tile_id: str = "ldem_80s_20m"
    tif_filename: str = "LDEM_80S_20MPP_ADJ.TIF"
    tif_url: Optional[str] = None
    tif_path: Optional[str] = None  # If provided, a local GeoTIFF is used.
    persistent_dir: str = "./persistent"
    runtime_dir: Optional[str] = None
    allow_download: bool = True
    force_download: bool = False
    timeout_s: int = 120

    # ROI
    roi_size_px: int = 1024
    use_canonical_roi: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None

    # Feature extraction
    pixel_size_m: float = 20.0
    roughness_method: str = "rms"
    roughness_window: int = 5
    pad_mode: str = "edge"
    reuse_cached_derived: bool = True

    # Crater products
    use_craters: bool = True
    crater_model_id: str = "catalogue_raster_v1"
    cache_product_name: str = "crater_mask"
    robbins_csv_path: Optional[str] = None
    rebuild_crater_if_missing: bool = False

    # Hazard model
    w_slope: float = 0.7
    w_roughness: float = 0.3
    slope_deg_max: float = 34.55
    roughness_rms_max: float = 18.56
    crater_distance_safe_m: float = 200.0
    crater_density_max: float = 0.25
    w_crater_mask: float = 0.4
    w_crater_distance: float = 0.4
    w_crater_density: float = 0.2
    terrain_weight: float = 0.7
    crater_weight: float = 0.3
    use_impassable_mask: bool = True
    impassable_slope_deg: float = 40.0

    # Cost model
    alpha: float = 10.0
    hazard_block: float = 0.95
    block_cost: float = 1e6

    # Pathfinding
    start_rc: RC = (269, 499)
    goal_rc: RC = (122, 558)
    connectivity: int = 8
    heuristic_weight: float = 1.2
    corridor_radii: Tuple[int, ...] = (25, 50, 80, 120, 180, 260, 400)

    # Output
    run_name: str = "final_run_v1.5"
    write_metrics: bool = True


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def log_step(message: str) -> None:
    """Print a short progress message and flush it immediately."""
    print(f"[Endymion] {message}", flush=True)


def log_path(label: str, value: str | Path | None) -> None:
    """Print important paths in a consistent format."""
    if value is None:
        print(f"[Endymion] {label}: None", flush=True)
    else:
        print(f"[Endymion] {label}: {value}", flush=True)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Convert values into JSON-safe Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return str(value)


def _resolve_roi(dh: DataHandler, cfg: RunConfig) -> ROI:
    """Resolve the ROI for the current run."""
    if cfg.use_canonical_roi:
        return dh.canonical_roi(cfg.tile_id, size=cfg.roi_size_px)

    if cfg.roi is None:
        raise ValueError("Provide roi when use_canonical_roi=False.")

    return tuple(int(x) for x in cfg.roi)  # type: ignore[return-value]


def _prepare_tif_source(cfg: RunConfig) -> tuple[str, Optional[str], bool]:
    """
    Resolve the GeoTIFF source for DataHandler.

    Returns:
        tif_filename, tif_url, allow_download
    """
    tif_filename = cfg.tif_filename
    tif_url = cfg.tif_url
    allow_download = cfg.allow_download

    if cfg.tif_path is None:
        log_step("No local GeoTIFF path provided; using cache/download resolution.")
        return tif_filename, tif_url, allow_download

    src = Path(cfg.tif_path)
    if not src.exists():
        raise FileNotFoundError(f"Local GeoTIFF not found: {src}")

    persistent_dir = Path(cfg.persistent_dir)
    persistent_dir.mkdir(parents=True, exist_ok=True)

    cached_tif = persistent_dir / src.name

    if src.resolve() != cached_tif.resolve():
        if (not cached_tif.exists()) or (cached_tif.stat().st_size != src.stat().st_size):
            log_step("Copying local GeoTIFF into persistent cache...")
            shutil.copy2(src, cached_tif)

    log_path("Local GeoTIFF", src)
    log_path("Cached GeoTIFF", cached_tif)

    # When a local file is provided, downloading is no longer needed.
    return src.name, None, False


def _precheck_crater_inputs(dh: DataHandler, cfg: RunConfig, roi: ROI) -> None:
    """
    Fail early with a clear message if crater mode is enabled but neither:
    - cached crater products exist, nor
    - a valid Robbins CSV path was provided.
    """
    if not cfg.use_craters:
        return

    derived_dir = Path(dh.derived_dir(cfg.tile_id, roi))
    mask_path = derived_dir / f"{cfg.cache_product_name}.npy"
    distance_path = derived_dir / f"{cfg.cache_product_name}_distance.npy"
    density_path = derived_dir / f"{cfg.cache_product_name}_density.npy"
    meta_path = derived_dir / f"{cfg.cache_product_name}_meta.json"

    cache_exists = all(
        p.exists() for p in (mask_path, distance_path, density_path, meta_path)
    )

    if cache_exists:
        log_step("Crater cache found in derived directory.")
        return

    if cfg.robbins_csv_path is None:
        raise FileNotFoundError(
            "Crater cache was not found for this ROI, and no Robbins CSV path was provided. "
            "Use --robbins-csv <path> or run with --no-craters."
        )

    csv_path = Path(cfg.robbins_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Robbins CSV not found: {csv_path}")

    log_path("Robbins CSV", csv_path)


def _validate_start_goal(cfg: RunConfig, raster_shape: tuple[int, int], roi: ROI) -> None:
    """
    Validate that start and goal lie inside the current ROI grid.

    The message is intentionally explicit because users may confuse
    ROI-local coordinates with full-raster coordinates.
    """
    h, w = raster_shape
    sr, sc = cfg.start_rc
    gr, gc = cfg.goal_rc

    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError(
            f"Start point {cfg.start_rc} is out of bounds for ROI shape {raster_shape}. "
            f"Valid row range: 0 to {h - 1}; valid col range: 0 to {w - 1}. "
            f"The resolved ROI is {roi}. Coordinates must be ROI-local, not full-raster pixel coordinates."
        )

    if not (0 <= gr < h and 0 <= gc < w):
        raise ValueError(
            f"Goal point {cfg.goal_rc} is out of bounds for ROI shape {raster_shape}. "
            f"Valid row range: 0 to {h - 1}; valid col range: 0 to {w - 1}. "
            f"The resolved ROI is {roi}. Coordinates must be ROI-local, not full-raster pixel coordinates."
        )


def _resolve_run_name(dh: DataHandler, tile_id: str, roi: ROI, run_name: str) -> str:
    """
    Return a safe run name that does not overwrite an existing run directory.
    If the requested name already exists, _v2, _v3, ... is appended.
    """
    base_dir = dh.derived_dir(tile_id, roi) / "navigation"
    candidate = run_name
    run_dir = base_dir / candidate

    if not run_dir.exists():
        return candidate

    version = 2
    while True:
        candidate = f"{run_name}_v{version}"
        run_dir = base_dir / candidate
        if not run_dir.exists():
            return candidate
        version += 1


def _load_or_build_derived(dh: DataHandler, cfg: RunConfig, roi: ROI) -> Dict[str, Any]:
    """Load terrain-derived rasters from cache or build them from the DEM."""
    if cfg.reuse_cached_derived:
        cached = dh.load_derived(cfg.tile_id, roi)
        if cached is not None:
            required = {"dem_m", "slope_deg", "roughness_rms"}
            if required.issubset(cached.keys()):
                return cached

    patch = dh.get_patch(cfg.tile_id, roi, verbose=True)
    dem_m = patch.data.astype(np.float32, copy=False)

    feature_extractor = FeatureExtractor(
        pixel_size_m=cfg.pixel_size_m,
        slope_units="degrees",
        expected_shape=dem_m.shape,
        strict_shape=False,
    )

    features = feature_extractor.extract(
        dem_m,
        roughness_method=cfg.roughness_method,
        roughness_window=cfg.roughness_window,
        pad_mode=cfg.pad_mode,
    )

    out_dir = dh.save_derived(
        cfg.tile_id,
        roi,
        dem_m=dem_m,
        features=features,
        meta_extra={
            "source_format": patch.meta.extra.get("source_format"),
            "pixel_size_m": cfg.pixel_size_m,
            "roughness_method": cfg.roughness_method,
            "roughness_window": cfg.roughness_window,
            "pad_mode": cfg.pad_mode,
            "patch_meta": patch.meta.extra,
        },
    )

    return {
        "meta": {"derived_dir": str(out_dir)},
        "dem_m": dem_m,
        **features,
    }


def _build_or_load_crater_outputs(
    dh: DataHandler,
    cfg: RunConfig,
    roi: ROI,
    dem_m: np.ndarray,
    features: Dict[str, np.ndarray],
) -> Optional[Dict[str, np.ndarray]]:
    """Load crater products from cache or build them from the catalogue."""
    if not cfg.use_craters:
        return None

    crater_predictor = CraterPredictor(
        model_id=cfg.crater_model_id,
        cache_product_name=cfg.cache_product_name,
    )

    tile_spec = dh.tiles[cfg.tile_id]
    dem_img_path = getattr(tile_spec, "tif_filename", None)
    if dem_img_path is not None:
        dem_img_path = str(Path(dh.runtime_dir) / str(dem_img_path))

    crater_out = crater_predictor.predict_proba_map(
        features=features,
        dem_m=dem_m,
        tile_id=cfg.tile_id,
        roi_pixels=roi,
        dh=dh,
        dem_img_path=dem_img_path,
        robbins_csv_path=cfg.robbins_csv_path,
        rebuild_if_missing=cfg.rebuild_crater_if_missing,
    )
    return crater_out


def _build_hazard(
    cfg: RunConfig,
    dem_m: np.ndarray,
    slope_deg: np.ndarray,
    roughness_rms: np.ndarray,
    crater_out: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Build the unified hazard map."""
    assessor = HazardAssessor(
        slope_deg_max=cfg.slope_deg_max,
        roughness_rms_max=cfg.roughness_rms_max,
        w_slope=cfg.w_slope,
        w_roughness=cfg.w_roughness,
        crater_distance_safe_m=cfg.crater_distance_safe_m,
        crater_density_max=cfg.crater_density_max,
        w_crater_mask=cfg.w_crater_mask,
        w_crater_distance=cfg.w_crater_distance,
        w_crater_density=cfg.w_crater_density,
        terrain_weight=cfg.terrain_weight,
        crater_weight=cfg.crater_weight,
        use_impassable_mask=cfg.use_impassable_mask,
        impassable_slope_deg=cfg.impassable_slope_deg,
    )

    crater_kwargs: Dict[str, Any] = {}
    if crater_out is not None:
        crater_kwargs = {
            "crater_mask": crater_out.get("crater_mask"),
            "crater_distance_m": crater_out.get("crater_distance_m"),
            "crater_density": crater_out.get("crater_density"),
        }

    return assessor.assess(
        dem_m=dem_m,
        slope_deg=slope_deg,
        roughness_rms=roughness_rms,
        **crater_kwargs,
    )


def _run_pathfinding(cfg: RunConfig, cost: np.ndarray, hazard: np.ndarray) -> Dict[str, Any]:
    """Run corridor-aware pathfinding over the cost grid."""
    pathfinder = Pathfinder(
        connectivity=cfg.connectivity,
        block_cost=cfg.block_cost,
        heuristic_weight=cfg.heuristic_weight,
    )

    corr = pathfinder.find_path_corridor(
        cost=cost,
        start=tuple(int(x) for x in cfg.start_rc),
        goal=tuple(int(x) for x in cfg.goal_rc),
        corridor_radii=cfg.corridor_radii,
        hazard=hazard,
    )

    best = corr.get("best")
    success = bool(best is not None and best.get("success", False))

    if success:
        path_rc = np.asarray(best["path_rc"], dtype=np.int32)
        result_meta = best.get("meta", {})
        result = {
            "success": True,
            "failure_reason": None,
            "total_cost": best.get("total_cost"),
            "expansions": result_meta.get("expansions"),
        }
    else:
        path_rc = np.zeros((0, 2), dtype=np.int32)
        failure_reason = None
        if corr.get("experiment"):
            failure_reason = corr["experiment"][-1].get("reason")
        result = {
            "success": False,
            "failure_reason": failure_reason or "no_path",
            "total_cost": None,
            "expansions": None,
        }

    return {
        "success": success,
        "path_rc": path_rc,
        "corridor": corr,
        "result": result,
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_endymion(cfg: RunConfig) -> Dict[str, Any]:
    log_step("Initialising run...")
    log_path("Persistent dir", cfg.persistent_dir)
    log_path("Runtime dir", cfg.runtime_dir)

    tif_filename, tif_url, allow_download = _prepare_tif_source(cfg)

    log_step("Initialising DataHandler...")
    tile_spec = GeoTiffTileSpec(
        tile_id=cfg.tile_id,
        tif_filename=tif_filename,
        tif_url=tif_url,
    )

    dh = DataHandler(
        base_url=cfg.base_url,
        tiles=[tile_spec],
        persistent_dir=cfg.persistent_dir,
        runtime_dir=cfg.runtime_dir,
        allow_download=allow_download,
        force_download=cfg.force_download,
        timeout_s=cfg.timeout_s,
    )

    log_step("Resolving ROI...")
    roi = _resolve_roi(dh, cfg)
    log_step(f"Using ROI: {roi}")

    safe_run_name = _resolve_run_name(dh, cfg.tile_id, roi, cfg.run_name)
    if safe_run_name != cfg.run_name:
        log_step(f"Run name '{cfg.run_name}' already exists; using '{safe_run_name}' instead.")

    _precheck_crater_inputs(dh, cfg, roi)

    t0 = perf_counter()
    log_step("Loading or building derived terrain rasters...")
    derived = _load_or_build_derived(dh, cfg, roi)
    log_step(f"Derived terrain ready in {perf_counter() - t0:.2f}s")

    dem_m = np.asarray(derived["dem_m"], dtype=np.float32)
    slope_deg = np.asarray(derived["slope_deg"], dtype=np.float32)
    roughness_rms = np.asarray(derived["roughness_rms"], dtype=np.float32)

    # The ROI shape is known now, so the path coordinates can be checked early.
    _validate_start_goal(cfg, dem_m.shape, roi)

    terrain_features = {
        "slope_deg": slope_deg,
        "roughness_rms": roughness_rms,
    }

    if cfg.use_craters:
        t0 = perf_counter()
        log_step("Loading or building crater products...")
    else:
        log_step("Skipping crater products (--no-craters enabled).")

    crater_out = _build_or_load_crater_outputs(
        dh=dh,
        cfg=cfg,
        roi=roi,
        dem_m=dem_m,
        features=terrain_features,
    )

    if cfg.use_craters:
        log_step(f"Crater products ready in {perf_counter() - t0:.2f}s")

    t0 = perf_counter()
    log_step("Building hazard map...")
    hazard_out = _build_hazard(
        cfg=cfg,
        dem_m=dem_m,
        slope_deg=slope_deg,
        roughness_rms=roughness_rms,
        crater_out=crater_out,
    )
    log_step(f"Hazard map ready in {perf_counter() - t0:.2f}s")

    hazard = np.asarray(hazard_out["hazard"], dtype=np.float32)

    t0 = perf_counter()
    log_step("Building traversal cost grid...")
    cost = build_cost_from_hazard(
        hazard,
        alpha=cfg.alpha,
        hazard_block=cfg.hazard_block,
        block_cost=cfg.block_cost,
    )
    log_step(f"Cost grid ready in {perf_counter() - t0:.2f}s")

    t0 = perf_counter()
    log_step("Running pathfinder...")
    path_out = _run_pathfinding(cfg, cost=cost, hazard=hazard)
    path_rc = path_out["path_rc"]
    log_step(f"Pathfinder finished in {perf_counter() - t0:.2f}s")

    nav_meta = {
        "start_rc": list(tuple(int(x) for x in cfg.start_rc)),
        "goal_rc": list(tuple(int(x) for x in cfg.goal_rc)),
        "hazard_assessor": _json_safe(hazard_out.get("meta", {})),
        "cost_model": {
            "alpha": cfg.alpha,
            "hazard_block": cfg.hazard_block,
            "block_cost": cfg.block_cost,
        },
        "pathfinder": {
            "connectivity": cfg.connectivity,
            "heuristic_weight": cfg.heuristic_weight,
            "corridor_radii": list(cfg.corridor_radii),
            "used_radius": path_out["corridor"].get("best_radius_px"),
        },
        "corridor_experiment": _json_safe(path_out["corridor"].get("experiment", [])),
        "result": _json_safe(path_out["result"]),
        "pipeline": {
            "tile_id": cfg.tile_id,
            "roi": list(roi),
            "use_canonical_roi": cfg.use_canonical_roi,
            "pixel_size_m": cfg.pixel_size_m,
            "use_craters": cfg.use_craters,
            "crater_model_id": cfg.crater_model_id if cfg.use_craters else None,
            "run_name": safe_run_name,
        },
    }

    if crater_out is not None:
        nav_meta["crater_predictor"] = _json_safe(crater_out.get("meta", {}))

    log_step("Saving navigation outputs...")
    run_dir = dh.save_navigation(
        tile_id=cfg.tile_id,
        roi=roi,
        hazard=hazard,
        cost=cost,
        path_rc=path_rc,
        run_name=safe_run_name,
        nav_meta=nav_meta,
    )
    log_path("Run dir", run_dir)

    metrics_path: Optional[Path] = None
    metrics: Optional[Dict[str, Any]] = None
    if cfg.write_metrics:
        log_step("Running evaluator...")
        evaluator = Evaluator(run_dir, pixel_size_m=cfg.pixel_size_m)
        metrics = evaluator.evaluate()
        metrics_path = evaluator.save()
        log_path("Metrics path", metrics_path)
    else:
        log_step("Skipping evaluator (--no-metrics enabled).")

    pipeline_summary = {
        "config": _json_safe(asdict(cfg)),
        "tile_id": cfg.tile_id,
        "roi": list(roi),
        "derived_dir": str(dh.derived_dir(cfg.tile_id, roi)),
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "result": _json_safe(path_out["result"]),
        "hazard_stats": _json_safe(hazard_out.get("meta", {}).get("stats", {})),
        "metrics": _json_safe(metrics) if metrics is not None else None,
        "run_name": safe_run_name,
    }

    (Path(run_dir) / "pipeline_summary.json").write_text(
        json.dumps(_json_safe(pipeline_summary), indent=2, allow_nan=False)
    )

    log_step("Run complete.")

    return {
        "config": cfg,
        "roi": roi,
        "derived": derived,
        "crater": crater_out,
        "hazard": hazard_out,
        "cost": cost,
        "path": path_out,
        "run_dir": run_dir,
        "metrics": metrics,
        "pipeline_summary": pipeline_summary,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one official Endymion pipeline execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source / storage
    parser.add_argument("--persistent-dir", default=RunConfig.persistent_dir)
    parser.add_argument("--runtime-dir", default=RunConfig.runtime_dir)
    parser.add_argument("--tile-id", default=RunConfig.tile_id)
    parser.add_argument("--tif-filename", default=RunConfig.tif_filename)
    parser.add_argument("--tif-url", default=RunConfig.tif_url)
    parser.add_argument("--tif-path", default=RunConfig.tif_path)
    parser.add_argument("--base-url", default=RunConfig.base_url)
    parser.add_argument("--run-name", default=RunConfig.run_name)

    # ROI
    parser.add_argument("--roi-size", type=int, default=RunConfig.roi_size_px)
    parser.add_argument("--no-canonical-roi", action="store_true")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"))

    # Feature extraction
    parser.add_argument("--rebuild-derived", action="store_true")

    # Crater products
    parser.add_argument("--no-craters", action="store_true")
    parser.add_argument("--robbins-csv", dest="robbins_csv_path", default=None)
    parser.add_argument("--rebuild-crater-if-missing", action="store_true")

    # Pathfinding
    parser.add_argument("--start", nargs=2, type=int, default=RunConfig.start_rc, metavar=("ROW", "COL"))
    parser.add_argument("--goal", nargs=2, type=int, default=RunConfig.goal_rc, metavar=("ROW", "COL"))
    parser.add_argument(
        "--corridor-radii",
        nargs="+",
        type=int,
        default=list(RunConfig.corridor_radii),
        metavar="R",
    )
    parser.add_argument("--heuristic-weight", type=float, default=RunConfig.heuristic_weight)

    # Cost model
    parser.add_argument("--alpha", type=float, default=RunConfig.alpha)
    parser.add_argument("--hazard-block", type=float, default=RunConfig.hazard_block)
    parser.add_argument("--block-cost", type=float, default=RunConfig.block_cost)

    # Download / output options
    parser.add_argument("--disable-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Convert CLI arguments into a RunConfig object."""
    use_craters = not args.no_craters
    allow_download = not args.disable_download

    roi = tuple(args.roi) if args.roi is not None else None

    return RunConfig(
        base_url=args.base_url,
        tile_id=args.tile_id,
        tif_filename=args.tif_filename,
        tif_url=args.tif_url,
        tif_path=args.tif_path,
        persistent_dir=args.persistent_dir,
        runtime_dir=args.runtime_dir,
        allow_download=allow_download,
        force_download=args.force_download,
        roi_size_px=args.roi_size,
        use_canonical_roi=not args.no_canonical_roi,
        roi=roi,
        reuse_cached_derived=not args.rebuild_derived,
        use_craters=use_craters,
        robbins_csv_path=args.robbins_csv_path,
        rebuild_crater_if_missing=args.rebuild_crater_if_missing,
        start_rc=tuple(args.start),
        goal_rc=tuple(args.goal),
        corridor_radii=tuple(args.corridor_radii),
        heuristic_weight=args.heuristic_weight,
        alpha=args.alpha,
        hazard_block=args.hazard_block,
        block_cost=args.block_cost,
        run_name=args.run_name,
        write_metrics=not args.no_metrics,
    )


def _print_run_report(out: Dict[str, Any]) -> None:
    """Print a short terminal summary after the run completes."""
    summary = out["pipeline_summary"]
    result = summary.get("result", {})
    metrics = summary.get("metrics") or {}

    print("=" * 72)
    print("Endymion run complete")
    print("=" * 72)
    print(f"Run dir      : {summary['run_dir']}")
    print(f"Derived dir  : {summary['derived_dir']}")
    print(f"ROI          : {tuple(summary['roi'])}")
    print(f"Run name     : {summary.get('run_name')}")
    print(f"Success      : {result.get('success')}")
    print(f"Failure      : {result.get('failure_reason')}")
    print(f"Total cost   : {result.get('total_cost')}")
    print(f"Expansions   : {result.get('expansions')}")

    if metrics:
        geom = metrics.get("geometry", {})
        safety = metrics.get("safety", {})
        eff = metrics.get("efficiency", {})
        print("-" * 72)
        print(f"Path length m: {geom.get('path_length_m')}")
        print(f"Mean hazard  : {safety.get('path_hazard_mean')}")
        print(f"Max hazard   : {safety.get('path_hazard_max')}")
        print(f"Safety score : {safety.get('safety_score')}")
        print(f"Cost / m     : {eff.get('cost_per_m')}")
        print(f"Metrics path : {summary.get('metrics_path')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    try:
        out = run_endymion(cfg)
        _print_run_report(out)
        return 0
    except Exception as exc:
        print(f"[run_endymion] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
