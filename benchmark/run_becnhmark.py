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

