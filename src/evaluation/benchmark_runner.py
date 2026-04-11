# -*- coding: utf-8 -*-
"""
BenchmarkRunner for Endymion
# - Redesign based on your current BenchmarkRunner, Evaluator, HazardAssessor,
#   Pathfinder, DataHandler, and crater integration direction in Endymion.
# - Keeps compatibility with the existing per-run Evaluator contract:
#   hazard.npy, cost.npy, path_rc.npy, nav_meta.json -> metrics.json
# - Main design change:
#   benchmark_runner now supports:
#       1) multiple benchmark cases
#       2) multiple hazard model variants
#       3) benchmark-level summary aggregation
# - csv is required as to write the sumarry file, which is written as a csv
# - math is used to detect non-finite plain python floats 
# - Idea source:
#   your current benchmark_runner.py only labels hazard_model_id in outputs,
#   but still always calls the same HazardAssessor internally.
#   This version turns hazard_model_id into an actual model registry.
"""


from __future__ import annotations

import csv
import json
from logging import root
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from src.data.data_handler import DataHandler, ROI
from src.models.hazard_assessor import HazardAssessor
from src.planning.pathfinder import Pathfinder, build_cost_from_hazard
from src.evaluation.evaluator import Evaluator

RC = Tuple[int, int]


@dataclass
class BenchmarkConfig:
    """Fixed settings kept constant across models for fair comparison."""
    tile_id: str
    roi: ROI
    benchmark_id: str = "benchmark_v2"
    pixel_size_m: float = 20.0

    # Cost model
    alpha: float = 10.0
    hazard_block: float = 0.95
    block_cost: float = 1e6

    # Pathfinder
    connectivity: int = 8
    heuristic_weight: float = 2.0
    corridor_radii: Tuple[int, ...] = (25, 50, 80, 120, 180, 260, 400)


@dataclass
class BenchmarkCase:
    """One benchmark start/goal pair plus optional metadata."""
    case_id: str
    start_rc: RC
    goal_rc: RC
    tier: str = "default"
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], tier: str = "default") -> "BenchmarkCase":
        return cls(
            case_id=d["id"],
            start_rc=tuple(d["start_rc"]),
            goal_rc=tuple(d["goal_rc"]),
            tier=d.get("tier", tier),
            notes=d.get("notes"),
        )


@dataclass
class HazardModelSpec:
    """
    Hazard model variant definition.

    kind:
      - terrain_only
      - terrain_plus_crater
      - terrain_plus_crater_ml
      - precomputed_hazard
    """
    model_id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Multi-case, multi-model benchmark runner for Endymion.

    Strategy
    --------
    1) Load terrain rasters once for the tile+ROI.
    2) Build each hazard model once.
    3) Run all benchmark cases against each model.
    4) Save one run directory per (case, model).
    5) Aggregate results into summary.json + summary.csv.
    """

    def __init__(self, dh: DataHandler, cfg: BenchmarkConfig):
        self.dh = dh
        self.cfg = cfg

        self.pathfinder = Pathfinder(
            connectivity=cfg.connectivity,
            block_cost=cfg.block_cost,
            heuristic_weight=cfg.heuristic_weight,
        )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _benchmark_root(self) -> Path:
        return self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi) / "benchmarks" / self.cfg.benchmark_id

    def _case_dir(self, case_id: str, model_id: str) -> Path:
        return self._benchmark_root() / case_id / model_id
    
    # JSON safety helper to ensure all values are JSON serializable, converting NaNs to None and numpy types to native Python types.
    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]

        if isinstance(value, np.ndarray):
            return {
                "_type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value) if np.isfinite(value) else None

        if isinstance(value, float):
            return value if math.isfinite(value) else None

        if isinstance(value, (np.bool_,)):
            return bool(value)

        return value

    # ------------------------------------------------------------------
    # Base loading
    # ------------------------------------------------------------------

    def _load_base_inputs(self) -> Dict[str, Any]:
        derived = self.dh.load_derived(self.cfg.tile_id, self.cfg.roi)
        if derived is None:
            raise RuntimeError(
                "Derived rasters not found. Run the pipeline first so the ROI cache "
                "contains dem_m, slope_deg, and roughness_rms."
            )

        required = ["dem_m", "slope_deg", "roughness_rms"]
        missing = [k for k in required if k not in derived]
        if missing:
            raise RuntimeError(f"Missing required derived rasters: {missing}")

        return derived

    def _load_crater_products_from_cache(
        self,
        expected_shape: Tuple[int, int],
        cache_product_name: str = "crater_mask",
    ) -> Dict[str, Any]:
        """
        Load crater products previously cached by CraterPredictor into the ROI derived dir.

        Expected files:
          crater_mask.npy
          crater_mask_distance.npy
          crater_mask_density.npy
          crater_mask_meta.json
        """
        derived_dir = Path(self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi))

        mask_path = derived_dir / f"{cache_product_name}.npy"
        distance_path = derived_dir / f"{cache_product_name}_distance.npy"
        density_path = derived_dir / f"{cache_product_name}_density.npy"
        meta_path = derived_dir / f"{cache_product_name}_meta.json"

        missing = [str(p.name) for p in [mask_path, distance_path, density_path, meta_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Crater cache not found in the ROI derived directory. Missing: "
                f"{missing}. Build crater products with CraterPredictor first."
            )

        crater_mask = np.load(mask_path)
        crater_distance_m = np.load(distance_path)
        crater_density = np.load(density_path)
        crater_meta = json.loads(meta_path.read_text())

        for name, arr in {
            "crater_mask": crater_mask,
            "crater_distance_m": crater_distance_m,
            "crater_density": crater_density,
        }.items():
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Cached {name} shape {arr.shape} does not match expected ROI shape {expected_shape}."
                )

        return {
            "crater_mask": crater_mask.astype(np.uint8, copy=False),
            "crater_distance_m": crater_distance_m.astype(np.float32, copy=False),
            "crater_density": crater_density.astype(np.float32, copy=False),
            "meta": crater_meta,
            "paths": {
                "crater_mask": mask_path.name,
                "crater_distance_m": distance_path.name,
                "crater_density": density_path.name,
                "meta": meta_path.name,
            },
        }
    
    # ------------------------------------------------------------------
    # Raster loading helpers
    # ------------------------------------------------------------------
    # This method is a general helper for loading any single raster, either from an explicit path or from the ROI derived dir by product name. It checks for existence and shape, and returns a dict with the array and metadata.
    def _load_single_raster(
        self,
        *,
        expected_shape: Tuple[int, int],
        raster_path: Optional[str | Path] = None,
        product_name: Optional[str] = None,
        dtype: np.dtype = np.float32,
    ) -> Dict[str, Any]:
        """
        Load one ROI-aligned raster either from an explicit path or from the ROI derived dir.

        Parameters
        ----------
        raster_path:
            Full path to a .npy file.
        product_name:
            Basename inside the ROI derived dir, without '.npy'.
            Example: 'crater_proba_ml_v1' -> crater_proba_ml_v1.npy
        """
        if raster_path is None and product_name is None:
            raise ValueError("Provide either raster_path or product_name.") # Ensure at least one of raster_path or product_name is provided.

        if raster_path is not None:
            path = Path(raster_path) # Use the provided explicit path.
        else:
            path = Path(self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi)) / f"{product_name}.npy" # Construct path from product name in the ROI derived directory.

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}") # Check that the raster file exists.

        arr = np.load(path).astype(dtype, copy=False) # Load the raster and convert to the desired dtype without copying if possible.
        if arr.shape != expected_shape:
            raise ValueError(
                f"Raster shape {arr.shape} does not match expected ROI shape {expected_shape}: {path}"
            ) 
        # Check that the raster shape matches the expected shape for the ROI.

        # If the raster is valid, return it along with metadata about the path and name.
        return {
            "array": arr,
            "path": str(path),
            "name": path.name,
        }

    # ------------------------------------------------------------------
    # Hazard builders
    # ------------------------------------------------------------------

    def _build_hazard_for_model(
        self,
        model: HazardModelSpec,
        base: Dict[str, Any],
    ) -> Dict[str, Any]:
        dem_m = base["dem_m"]
        slope_deg = base["slope_deg"]
        roughness_rms = base["roughness_rms"]

        if model.kind == "terrain_only":
            assessor = HazardAssessor(
                slope_deg_max=model.params.get("slope_deg_max", 34.55),
                roughness_rms_max=model.params.get("roughness_rms_max", 18.56),
                w_slope=model.params.get("w_slope", 0.7),
                w_roughness=model.params.get("w_roughness", 0.3),
                use_impassable_mask=model.params.get("use_impassable_mask", True),
                impassable_slope_deg=model.params.get("impassable_slope_deg", 40.0),
            )

            out = assessor.assess(
                slope_deg=slope_deg,
                roughness_rms=roughness_rms,
                dem_m=dem_m,
            )

            meta = dict(out["meta"])
            meta["model"] = model.model_id
            meta["benchmark_model_kind"] = model.kind

            return {
                "hazard": out["hazard"],
                "components": out.get("components", {}),
                "hazard_meta": meta,
            }

        if model.kind == "terrain_plus_crater":
            crater_density = model.params.get("crater_density")
            crater_distance_m = model.params.get("crater_distance_m")
            crater_mask = model.params.get("crater_mask")
            crater_cache_meta = None
            crater_cache_paths = None

            if crater_density is None or crater_distance_m is None:
                cache_product_name = model.params.get("cache_product_name", "crater_mask")
                crater_cache = self._load_crater_products_from_cache(
                    expected_shape=slope_deg.shape,
                    cache_product_name=cache_product_name,
                )
                crater_mask = crater_cache["crater_mask"]
                crater_density = crater_cache["crater_density"]
                crater_distance_m = crater_cache["crater_distance_m"]
                crater_cache_meta = crater_cache["meta"]
                crater_cache_paths = crater_cache["paths"]

            crater_density = np.asarray(crater_density, dtype=np.float32)
            crater_distance_m = np.asarray(crater_distance_m, dtype=np.float32)

            if crater_mask is not None:
                crater_mask = np.asarray(crater_mask, dtype=np.uint8)

            if crater_density.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: crater_density shape {crater_density.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )
            if crater_distance_m.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: crater_distance_m shape {crater_distance_m.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )
            if crater_mask is not None and crater_mask.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: crater_mask shape {crater_mask.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )

            terrain_assessor = HazardAssessor(
                slope_deg_max=model.params.get("slope_deg_max", 34.55),
                roughness_rms_max=model.params.get("roughness_rms_max", 18.56),
                w_slope=model.params.get("w_slope", 0.7),
                w_roughness=model.params.get("w_roughness", 0.3),
                use_impassable_mask=model.params.get("use_impassable_mask", True),
                impassable_slope_deg=model.params.get("impassable_slope_deg", 40.0),
            )

            terrain_out = terrain_assessor.assess(
                slope_deg=slope_deg,
                roughness_rms=roughness_rms,
                dem_m=dem_m,
            )
            terrain_hazard = terrain_out["hazard"].astype(np.float32, copy=False)

            density_n = np.clip(crater_density, 0.0, 1.0)

            max_distance_m = float(model.params.get("max_distance_m", 1000.0))
            if max_distance_m <= 0:
                raise ValueError("max_distance_m must be > 0 for terrain_plus_crater.")

            distance_n = np.clip(crater_distance_m / max_distance_m, 0.0, 1.0)
            proximity_n = 1.0 - distance_n

            w_density = float(model.params.get("w_crater_density", 0.5))
            w_proximity = float(model.params.get("w_crater_proximity", 0.5))
            crater_term = (w_density * density_n + w_proximity * proximity_n).astype(np.float32)

            w_terrain = float(model.params.get("w_terrain", 0.8))
            w_crater = float(model.params.get("w_crater", 0.2))

            hazard = np.clip(
                w_terrain * terrain_hazard + w_crater * crater_term,
                0.0,
                1.0,
            ).astype(np.float32)

            if model.params.get("use_impassable_mask", True):
                impassable_slope_deg = float(model.params.get("impassable_slope_deg", 40.0))
                hazard[slope_deg > impassable_slope_deg] = 1.0

            meta = {
                "model": model.model_id,
                "benchmark_model_kind": model.kind,
                "terrain_component": terrain_out["meta"],
                "crater_component": {
                    "source": "direct_params" if crater_cache_meta is None else "derived_cache",
                    "cache_product_name": model.params.get("cache_product_name", "crater_mask"),
                    "cache_products": crater_cache_paths,
                    "cache_catalogue_meta": crater_cache_meta,
                    "w_crater_density": w_density,
                    "w_crater_proximity": w_proximity,
                    "max_distance_m": max_distance_m,
                    "density_stats": {
                        "min": float(np.nanmin(density_n)),
                        "max": float(np.nanmax(density_n)),
                        "mean": float(np.nanmean(density_n)),
                    },
                    "proximity_stats": {
                        "min": float(np.nanmin(proximity_n)),
                        "max": float(np.nanmax(proximity_n)),
                        "mean": float(np.nanmean(proximity_n)),
                    },
                },
                "fusion": {
                    "w_terrain": w_terrain,
                    "w_crater": w_crater,
                },
                "stats": {
                    "hazard_min": float(np.nanmin(hazard)),
                    "hazard_max": float(np.nanmax(hazard)),
                    "hazard_mean": float(np.nanmean(hazard)),
                    "nan_ratio": float(np.isnan(hazard).mean()),
                },
            }

            components = {
                "terrain_hazard": terrain_hazard,
                "crater_density_norm": density_n,
                "crater_proximity_norm": proximity_n,
                "crater_term": crater_term,
            }
            if crater_mask is not None:
                components["crater_mask"] = crater_mask

            return {
                "hazard": hazard,
                "components": components,
                "hazard_meta": meta,
            }

        if model.kind == "precomputed_hazard":
            hazard = np.asarray(model.params["hazard"], dtype=np.float32)
            if hazard.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: precomputed hazard shape {hazard.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )

            meta = {
                "model": model.model_id,
                "benchmark_model_kind": model.kind,
                "stats": {
                    "hazard_min": float(np.nanmin(hazard)),
                    "hazard_max": float(np.nanmax(hazard)),
                    "hazard_mean": float(np.nanmean(hazard)),
                    "nan_ratio": float(np.isnan(hazard).mean()),
                },
            }

            return {
                "hazard": hazard,
                "components": {},
                "hazard_meta": meta,
            }

        raise ValueError(f"Unknown hazard model kind: {model.kind}")

    # ------------------------------------------------------------------
    # One case x one model
    # ------------------------------------------------------------------

    def run_case_model(
        self,
        case: BenchmarkCase,
        model: HazardModelSpec,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        hazard = model_output["hazard"]
        hazard_meta = model_output["hazard_meta"]

        cost = build_cost_from_hazard(
            hazard,
            alpha=self.cfg.alpha,
            hazard_block=self.cfg.hazard_block,
            block_cost=self.cfg.block_cost,
        )

        corr = self.pathfinder.find_path_corridor(
            cost=cost,
            start=case.start_rc,
            goal=case.goal_rc,
            corridor_radii=self.cfg.corridor_radii,
            hazard=hazard,
        )

        best = corr.get("best")
        best_radius = corr.get("best_radius_px")
        experiment = corr.get("experiment", [])

        success = bool(best and best.get("success", False))
        path_rc = (
            np.asarray(best["path_rc"], dtype=np.int32)
            if success
            else np.zeros((0, 2), dtype=np.int32)
        )

        out_dir = self._case_dir(case.case_id, model.model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "hazard.npy", hazard.astype(np.float32, copy=False))
        np.save(out_dir / "cost.npy", cost.astype(np.float32, copy=False))
        np.save(out_dir / "path_rc.npy", path_rc)

        # When writing nav_meta.json, ensure all values are JSON serializable.
        def _json_safe(value):
            import numpy as np

            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, tuple):
                return [_json_safe(v) for v in value]
            if isinstance(value, (np.floating, float)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.bool_, bool)):
                return bool(value)
            return value

        nav_meta = {
            "tile_id": self.cfg.tile_id,
            "roi": list(self.cfg.roi),
            "shape": list(hazard.shape),
            "run_name": f"{self.cfg.benchmark_id}/{case.case_id}/{model.model_id}",
            "products": ["hazard", "cost", "path_rc"],
            "nav_meta": {
                "start_rc": list(case.start_rc),
                "goal_rc": list(case.goal_rc),
                "hazard_assessor": hazard_meta,
                "cost_model": {
                    "alpha": self.cfg.alpha,
                    "hazard_block": self.cfg.hazard_block,
                    "block_cost": self.cfg.block_cost,
                },
                "pathfinder": {
                    "connectivity": self.cfg.connectivity,
                    "heuristic_weight": self.cfg.heuristic_weight,
                    "corridor_radii": list(self.cfg.corridor_radii),
                    "corridor_radius_px_used": best_radius,
                },
                "result": {
                    "success": success,
                    "total_cost": float(best["total_cost"]) if success else None,
                    "path_len": int(path_rc.shape[0]),
                    "expansions": int((best.get("meta") or {}).get("expansions", -1))
                    if best is not None
                    else -1,
                    "failure_reason": None if success else "no_path_found_in_any_corridor",
                },
                "benchmark": {
                    "benchmark_id": self.cfg.benchmark_id,
                    "case_id": case.case_id,
                    "tier": case.tier,
                    "hazard_model_id": model.model_id,
                    "hazard_model_kind": model.kind,
                    "case_notes": case.notes,
                },
                "corridor_experiment": experiment,
            },
        }

        (out_dir / "nav_meta.json").write_text(json.dumps(self._json_safe(nav_meta), indent=2, allow_nan=False)) # Ensure all values are JSON serializable, converting NaNs to None and numpy types to native Python types.

        evaluator = Evaluator(out_dir, pixel_size_m=self.cfg.pixel_size_m)
        metrics_path = evaluator.save("metrics.json")
        metrics = json.loads(metrics_path.read_text())

        return {
            "case_id": case.case_id,
            "tier": case.tier,
            "hazard_model_id": model.model_id,
            "hazard_model_kind": model.kind,
            "success": success,
            "out_dir": str(out_dir),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_result_row(result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = result["metrics"]
        return {
            "case_id": result["case_id"],
            "tier": result["tier"],
            "hazard_model_id": result["hazard_model_id"],
            "hazard_model_kind": result["hazard_model_kind"],
            "success": metrics["status"]["success"],
            "failure_reason": metrics["status"]["failure_reason"],

            "path_nodes": metrics["geometry"]["path_nodes"],
            "path_length_m": metrics["geometry"]["path_length_m"],
            "straight_line_m": metrics["geometry"]["straight_line_m"],
            "detour_ratio": metrics["geometry"]["detour_ratio"],
            "turn_count": metrics["geometry"]["turn_count"],
            "turns_per_m": metrics["geometry"]["turns_per_m"],

            "path_hazard_mean": metrics["safety"]["path_hazard_mean"],
            "path_hazard_max": metrics["safety"]["path_hazard_max"],
            "path_hazard_p95": metrics["safety"]["path_hazard_p95"],
            "haz_per_m": metrics["safety"]["haz_per_m"],
            "frac_hazard_ge_block": metrics["safety"]["frac_hazard_ge_block"],
            "hazard_mean_ratio": metrics["safety"]["hazard_mean_ratio"],
            "safety_score": metrics["safety"]["safety_score"],

            "cost_sum_along_path": metrics["efficiency"]["cost_sum_along_path"],
            "cost_per_m": metrics["efficiency"]["cost_per_m"],
            "planner_total_cost": metrics["efficiency"]["planner_total_cost"],
            "expansions": metrics["efficiency"]["expansions"],
        }

    @staticmethod
    def _mean_ignore_none(values: List[Any]) -> Optional[float]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    def _build_model_summary(self, flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in flat_rows:
            grouped.setdefault(row["hazard_model_id"], []).append(row)

        summary = []
        for model_id, rows in grouped.items():
            summary.append({
                "hazard_model_id": model_id,
                "hazard_model_kind": rows[0]["hazard_model_kind"],
                "num_cases": len(rows),
                "num_success": int(sum(bool(r["success"]) for r in rows)),
                "success_rate": float(np.mean([bool(r["success"]) for r in rows])) if rows else None,
                "mean_path_length_m": self._mean_ignore_none([r["path_length_m"] for r in rows]),
                "mean_path_hazard_mean": self._mean_ignore_none([r["path_hazard_mean"] for r in rows]),
                "mean_path_hazard_max": self._mean_ignore_none([r["path_hazard_max"] for r in rows]),
                "mean_safety_score": self._mean_ignore_none([r["safety_score"] for r in rows]),
                "mean_cost_per_m": self._mean_ignore_none([r["cost_per_m"] for r in rows]),
                "mean_expansions": self._mean_ignore_none([r["expansions"] for r in rows]),
            })
        return summary

    def _json_safe(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types into safe summaries for manifest writing."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        return str(obj)

    def _write_summary_files(self, results: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Dict[str, Any]:
        root = self._benchmark_root()
        root.mkdir(parents=True, exist_ok=True)

        flat_rows = [self._flatten_result_row(r) for r in results]
        model_summary = self._build_model_summary(flat_rows)

        safe_manifest = self._json_safe(manifest)
        summary = {
            "benchmark_manifest": safe_manifest,
            "model_summary": model_summary,
            "rows": flat_rows,
        }

        (root / "manifest.json").write_text(json.dumps(self._json_safe(manifest), indent=2, allow_nan=False)) # Ensure all values are JSON serializable, converting NaNs to None and numpy types to native Python types.
        (root / "summary.json").write_text(json.dumps(self._json_safe(summary), indent=2, allow_nan=False)) # Ensure all values are JSON serializable, converting NaNs to None and numpy types to native Python types.

        if flat_rows:
            fieldnames = list(flat_rows[0].keys())
            with open(root / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)

        return summary

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------

    def run(
        self,
        cases: List[BenchmarkCase],
        hazard_models: List[HazardModelSpec],
    ) -> Dict[str, Any]:
        if not cases:
            raise ValueError("No benchmark cases provided.")
        if not hazard_models:
            raise ValueError("No hazard models provided.")

        base = self._load_base_inputs()

        model_outputs: Dict[str, Dict[str, Any]] = {}
        for model in hazard_models:
            model_outputs[model.model_id] = self._build_hazard_for_model(model, base)

        results: List[Dict[str, Any]] = []
        for case in cases:
            for model in hazard_models:
                result = self.run_case_model(
                    case=case,
                    model=model,
                    model_output=model_outputs[model.model_id],
                )
                results.append(result)

        manifest = {
            "benchmark_id": self.cfg.benchmark_id,
            "tile_id": self.cfg.tile_id,
            "roi": list(self.cfg.roi),
            "num_cases": len(cases),
            "num_models": len(hazard_models),
            "cases": [
                {
                    "case_id": c.case_id,
                    "tier": c.tier,
                    "start_rc": list(c.start_rc),
                    "goal_rc": list(c.goal_rc),
                    "notes": c.notes,
                }
                for c in cases
            ],
            "hazard_models": [
                {
                    "model_id": m.model_id,
                    "kind": m.kind,
                    "params": self._json_safe(m.params),
                }
                for m in hazard_models
            ],

            "fixed_settings": {
                "alpha": self.cfg.alpha,
                "hazard_block": self.cfg.hazard_block,
                "block_cost": self.cfg.block_cost,
                "connectivity": self.cfg.connectivity,
                "heuristic_weight": self.cfg.heuristic_weight,
                "corridor_radii": list(self.cfg.corridor_radii),
                "pixel_size_m": self.cfg.pixel_size_m,
            },
        }

        summary = self._write_summary_files(results, manifest)
        return {
            "benchmark_root": str(self._benchmark_root()),
            "manifest": manifest,
            "summary": summary,
            "results": results,
        }

    def run_benchmark_file(
        self,
        benchmark_json_path: str | Path,
        hazard_models: List[HazardModelSpec],
    ) -> Dict[str, Any]:
        spec = json.loads(Path(benchmark_json_path).read_text())

        if "tile_id" in spec and spec["tile_id"] != self.cfg.tile_id:
            raise ValueError(
                f"Benchmark spec tile_id={spec['tile_id']} does not match cfg.tile_id={self.cfg.tile_id}"
            )
        if "roi" in spec and tuple(spec["roi"]) != tuple(self.cfg.roi):
            raise ValueError(
                f"Benchmark spec roi={spec['roi']} does not match cfg.roi={list(self.cfg.roi)}"
            )

        cases: List[BenchmarkCase] = []
        for tier_name, tier_cases in spec["tiers"].items():
            for case in tier_cases:
                case = dict(case)
                case["tier"] = tier_name
                cases.append(BenchmarkCase.from_dict(case, tier=tier_name))

        return self.run(cases=cases, hazard_models=hazard_models)