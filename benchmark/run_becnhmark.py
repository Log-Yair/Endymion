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


