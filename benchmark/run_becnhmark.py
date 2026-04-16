from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# This path setup allows the script to run from the repo root with:
#   python benchmark/run_benchmarks.py
THIS_FILE = Path(__file__).resolve()
BENCHMARK_DIR = THIS_FILE.parent
REPO_ROOT = BENCHMARK_DIR.parent
SRC_DIR = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_DIR, BENCHMARK_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    from src.data.data_handler import DataHandler, GeoTiffTileSpec
    from src.evaluation.benchmark_runner import (
        BenchmarkRunner,
        BenchmarkConfig,
        BenchmarkCase,
        HazardModelSpec,
    )
except ImportError:
    # Fallback for slightly messy local layouts during cleanup.
    from data.data_handler import DataHandler, GeoTiffTileSpec
    from evaluation.benchmark_runner import (
        BenchmarkRunner,
        BenchmarkConfig,
        BenchmarkCase,
        HazardModelSpec,
    )


def log_step(message: str) -> None:
    print(f"[Benchmark] {message}", flush=True)


def log_path(label: str, value: str | Path | None) -> None:
    if value is None:
        print(f"[Benchmark] {label}: None", flush=True)
    else:
        print(f"[Benchmark] {label}: {value}", flush=True)


def _default_cases() -> List[BenchmarkCase]:
    """
    Small starter benchmark set.
    These are ROI-local coordinates, not full-raster coordinates.
    The user can later replace these with a JSON benchmark file or custom cases.
    """
    return [
        BenchmarkCase(
            case_id="easy_01",
            start_rc=(269, 499),
            goal_rc=(122, 558),
            tier="easy",
            notes="Short validated case used during runner testing.",
        ),
        BenchmarkCase(
            case_id="medium_01",
            start_rc=(569, 899),
            goal_rc=(122, 588),
            tier="medium",
            notes="Higher-risk start region; previously validated in the notebook flow.",
        ),
        BenchmarkCase(
            case_id="hard_01",
            start_rc=(850, 250),
            goal_rc=(180, 760),
            tier="hard",
            notes="Longer cross-ROI traversal for early benchmark comparison.",
        ),
    ]


def _build_hazard_models(args: argparse.Namespace) -> List[HazardModelSpec]:
    """
    Build the benchmark model registry.

    The first benchmark pass stays small on purpose:
    - terrain_only
    - terrain_plus_crater

    ML can be added later without changing the benchmark runner structure.
    """
    models: List[HazardModelSpec] = [
        HazardModelSpec(
            model_id=args.terrain_model_id,
            kind="terrain_only",
            params={
                "slope_deg_max": args.slope_deg_max,
                "roughness_rms_max": args.roughness_rms_max,
                "w_slope": args.w_slope,
                "w_roughness": args.w_roughness,
                "use_impassable_mask": True,
                "impassable_slope_deg": args.impassable_slope_deg,
            },
        )
    ]

    if not args.no_crater_model:
        models.append(
            HazardModelSpec(
                model_id=args.crater_model_id,
                kind="terrain_plus_crater",
                params={
                    "cache_product_name": args.cache_product_name,
                    "slope_deg_max": args.slope_deg_max,
                    "roughness_rms_max": args.roughness_rms_max,
                    "w_slope": args.w_slope,
                    "w_roughness": args.w_roughness,
                    "use_impassable_mask": True,
                    "impassable_slope_deg": args.impassable_slope_deg,
                    "max_distance_m": args.max_distance_m,
                    "w_crater_density": args.w_crater_density,
                    "w_crater_proximity": args.w_crater_proximity,
                    "w_terrain": args.w_terrain,
                    "w_crater": args.w_crater,
                },
            )
        )

    if args.include_ml_model:
        models.append(
            HazardModelSpec(
                model_id=args.ml_model_id,
                kind="terrain_plus_crater_ml",
                params={
                    "cache_product_name": args.cache_product_name,
                    "ml_product_name": args.ml_product_name,
                    "slope_deg_max": args.slope_deg_max,
                    "roughness_rms_max": args.roughness_rms_max,
                    "w_slope": args.w_slope,
                    "w_roughness": args.w_roughness,
                    "use_impassable_mask": True,
                    "impassable_slope_deg": args.impassable_slope_deg,
                    "max_distance_m": args.max_distance_m,
                    "w_crater_density": args.w_crater_density_ml,
                    "w_crater_proximity": args.w_crater_proximity_ml,
                    "w_crater_ml": args.w_crater_ml,
                    "w_terrain": args.w_terrain_ml,
                    "w_crater": args.w_crater_ml_outer,
                },
            )
        )

    return models


def _build_data_handler(args: argparse.Namespace) -> DataHandler:
    """
    Build the minimal DataHandler needed by BenchmarkRunner.

    Benchmarking assumes the derived ROI cache already exists.
    That means:
    - terrain rasters must already be saved
    - crater cache must already be saved if crater models are used
    """
    tile_spec = GeoTiffTileSpec(
        tile_id=args.tile_id,
        tif_filename=args.tif_filename,
        tif_url=args.tif_url,
    )

    return DataHandler(
        base_url=args.base_url,
        tiles=[tile_spec],
        persistent_dir=args.persistent_dir,
        runtime_dir=args.runtime_dir,
        allow_download=not args.disable_download,
        force_download=args.force_download,
        timeout_s=args.timeout_s,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Endymion benchmark comparisons over multiple cases and hazard models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / cache
    parser.add_argument("--persistent-dir", default="C:\\Endymion\\persistent")
    parser.add_argument("--runtime-dir", default="C:\\Endymion\\runtime")
    parser.add_argument("--base-url", default="https://pgda.gsfc.nasa.gov/data/LOLA_20mpp")
    parser.add_argument("--tile-id", default="ldem_80s_20m")
    parser.add_argument("--tif-filename", default="LDEM_80S_20MPP_ADJ.TIF")
    parser.add_argument("--tif-url", default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--disable-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")

    # ROI
    parser.add_argument("--roi-size", type=int, default=1024)
    parser.add_argument("--no-canonical-roi", action="store_true")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"))

    # Benchmark identity / fixed settings
    parser.add_argument("--benchmark-id", default="benchmark_phase2_cli")
    parser.add_argument("--pixel-size-m", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--hazard-block", type=float, default=0.95)
    parser.add_argument("--block-cost", type=float, default=1e6)
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--heuristic-weight", type=float, default=2.0)
    parser.add_argument(
        "--corridor-radii",
        nargs="+",
        type=int,
        default=[25, 50, 80, 120, 180, 260, 400],
        metavar="R",
    )

    # Benchmark cases
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Optional benchmark spec JSON file. If omitted, built-in starter cases are used.",
    )

    # Terrain model parameters
    parser.add_argument("--terrain-model-id", default="terrain_only_v1")
    parser.add_argument("--slope-deg-max", type=float, default=34.55)
    parser.add_argument("--roughness-rms-max", type=float, default=18.56)
    parser.add_argument("--w-slope", type=float, default=0.7)
    parser.add_argument("--w-roughness", type=float, default=0.3)
    parser.add_argument("--impassable-slope-deg", type=float, default=40.0)

    # Crater model parameters
    parser.add_argument("--no-crater-model", action="store_true")
    parser.add_argument("--crater-model-id", default="terrain_crater_v1")
    parser.add_argument("--cache-product-name", default="crater_mask")
    parser.add_argument("--max-distance-m", type=float, default=1000.0)
    parser.add_argument("--w-crater-density", type=float, default=0.5)
    parser.add_argument("--w-crater-proximity", type=float, default=0.5)
    parser.add_argument("--w-terrain", type=float, default=0.8)
    parser.add_argument("--w-crater", type=float, default=0.2)

    # Optional ML model parameters
    parser.add_argument("--include-ml-model", action="store_true")
    parser.add_argument("--ml-model-id", default="terrain_crater_ml_v1")
    parser.add_argument("--ml-product-name", default="crater_proba_ml_v1")
    parser.add_argument("--w-crater-density-ml", type=float, default=0.3)
    parser.add_argument("--w-crater-proximity-ml", type=float, default=0.3)
    parser.add_argument("--w-crater-ml", type=float, default=0.4)
    parser.add_argument("--w-terrain-ml", type=float, default=0.85)
    parser.add_argument("--w-crater-ml-outer", type=float, default=0.15)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        log_step("Initialising DataHandler...")
        dh = _build_data_handler(args)

        if args.no_canonical_roi:
            if args.roi is None:
                raise ValueError("Provide --roi R0 R1 C0 C1 when --no-canonical-roi is used.")
            roi = tuple(int(x) for x in args.roi)
        else:
            log_step("Resolving canonical ROI...")
            roi = dh.canonical_roi(args.tile_id, size=args.roi_size)

        log_step(f"Using ROI: {roi}")

        cfg = BenchmarkConfig(
            tile_id=args.tile_id,
            roi=roi,
            benchmark_id=args.benchmark_id,
            pixel_size_m=args.pixel_size_m,
            alpha=args.alpha,
            hazard_block=args.hazard_block,
            block_cost=args.block_cost,
            connectivity=args.connectivity,
            heuristic_weight=args.heuristic_weight,
            corridor_radii=tuple(args.corridor_radii),
        )

        runner = BenchmarkRunner(dh, cfg)

        hazard_models =