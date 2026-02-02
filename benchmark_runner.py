
# -*- coding: utf-8 -*-
"""
BenchmarkRunner for Endymion

Runs a fixed benchmark suite (start/goal pairs) against one or more hazard models.

Design goal:
-  NOT modify HazardAssessor/Pathfinder to "support benchmarking".
- Benchmarking is orchestration: loop cases, call components, save artifacts, evaluate.

Folder layout (inside DataHandler.derived_dir(tile_id, roi)):
  benchmarks/benchmark_v1/<case_id>/<hazard_model_id>/
    hazard.npy
    cost.npy
    path_rc.npy
    nav_meta.json
    metrics.json (optional, via Evaluator)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from data_handler import DataHandler, ROI
from hazard_assessor import HazardAssessor
from pathfinder import Pathfinder, build_cost_from_hazard


RC = Tuple[int, int]


@dataclass
class BenchmarkConfig:
    """Fixed settings to keep comparisons fair across hazard model versions."""
    tile_id: str
    roi: ROI
    benchmark_id: str = "benchmark_v1"
    pixel_size_m: float = 20.0

    # Cost model (keep fixed for fairness)
    alpha: float = 10.0
    hazard_block: float = 0.95
    block_cost: float = 1e6

    # Pathfinder (keep fixed for fairness)
    connectivity: int = 8
    heuristic_weight: float = 2.0
    corridor_radii: Tuple[int, ...] = (25, 50, 80, 120, 180, 260, 400)


class BenchmarkRunner:
    """
    Executes a benchmark suite for a given tile+ROI.

    It expects derived rasters to already exist:
      dem_m.npy, slope_deg.npy, roughness_rms.npy + meta.json
    """

    def __init__(self, dh: DataHandler, cfg: BenchmarkConfig):
        self.dh = dh
        self.cfg = cfg

        # Core components (your uploaded implementations)
        self.hazard_assessor = HazardAssessor()  # can override params per model if needed
        self.pathfinder = Pathfinder(
            connectivity=cfg.connectivity,
            block_cost=cfg.block_cost,
            heuristic_weight=cfg.heuristic_weight,
        )

    # -------------------------
    # Paths / storage
    # -------------------------

    def _benchmark_root(self) -> Path:
        """Root directory where all benchmark results for this ROI live."""
        return self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi) / "benchmarks" / self.cfg.benchmark_id

    def _case_dir(self, case_id: str, hazard_model_id: str) -> Path:
        """Directory for one case and one hazard model variant."""
        return self._benchmark_root() / case_id / hazard_model_id

    # -------------------------
    # Running one case
    # -------------------------

    def run_case(self, case: Dict[str, Any], hazard_model_id: str = "terrain_weighted_v1") -> Dict[str, Any]:
        """
        Runs one start/goal pair and saves artifacts.

        case example:
          {"id": "A01", "start_rc": [100,100], "goal_rc": [924,924]}
        """
        case_id = case["id"]
        start: RC = tuple(case["start_rc"])
        goal: RC = tuple(case["goal_rc"])

        # Load cached derived rasters (fast reload path)
        derived = self.dh.load_derived(self.cfg.tile_id, self.cfg.roi)
        if derived is None:
            raise RuntimeError(
                "Derived rasters not found. Run the pipeline once to generate "
                "dem_m, slope_deg, roughness_rms in the derived cache."
            )

        dem_m = derived["dem_m"]
        slope_deg = derived["slope_deg"]
        roughness_rms = derived["roughness_rms"]

        # 1) Hazard generation (baseline terrain model now; later swap/extend for crater-ML)
        haz_out = self.hazard_assessor.assess(slope_deg=slope_deg, roughness_rms=roughness_rms, dem_m=dem_m)
        hazard = haz_out["hazard"]
        hazard_meta = haz_out["meta"]

        # 2) Cost grid
        cost = build_cost_from_hazard(
            hazard,
            alpha=self.cfg.alpha,
            hazard_block=self.cfg.hazard_block,
            block_cost=self.cfg.block_cost,
        )

        # 3) Corridor search experiment + pick first success
        corr = self.pathfinder.find_path_corridor(
            cost=cost,
            start=start,
            goal=goal,
            corridor_radii=self.cfg.corridor_radii,
            hazard=hazard,  # enables per-path hazard mean/max in the experiment log
        )

        best = corr["best"]
        best_radius = corr["best_radius_px"]
        experiment = corr["experiment"]

        success = bool(best and best.get("success", False))
        path_rc = np.asarray(best["path_rc"], dtype=np.int32) if success else np.zeros((0, 2), dtype=np.int32)

        # 4) Save artifacts
        out_dir = self._case_dir(case_id, hazard_model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "hazard.npy", hazard.astype(np.float32))
        np.save(out_dir / "cost.npy", cost.astype(np.float32))
        np.save(out_dir / "path_rc.npy", path_rc)

        # 5) Build nav_meta.json (aligned with your existing schema)
        nav_meta = {
            "tile_id": self.cfg.tile_id,
            "roi": list(self.cfg.roi),
            "shape": list(hazard.shape),
            "run_name": f"{self.cfg.benchmark_id}/{case_id}/{hazard_model_id}",
            "products": ["hazard", "cost", "path_rc"],
            "nav_meta": {
                "start_rc": list(start),
                "goal_rc": list(goal),
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
                    "total_cost": float(best["total_cost"]) if success else float("inf"),
                    "path_len": int(path_rc.shape[0]),
                    "expansions": int(best["meta"]["expansions"]) if success else int(best.get("meta", {}).get("expansions", -1)),
                },
                "corridor_experiment": experiment,
            }
        }

        (out_dir / "nav_meta.json").write_text(json.dumps(nav_meta, indent=2))

        # Optional: if you have Evaluator implemented, call it here and write metrics.json
        # evaluator = Evaluator(out_dir, pixel_size_m=self.cfg.pixel_size_m)
        # evaluator.save("metrics.json")

        return {
            "case_id": case_id,
            "hazard_model_id": hazard_model_id,
            "success": success,
            "out_dir": str(out_dir),
        }

    # -------------------------
    # Run full benchmark spec
    # -------------------------

    def run_benchmark_file(self, benchmark_json_path: str | Path, hazard_model_id: str = "terrain_weighted_v1") -> List[Dict[str, Any]]:
        """
        Loads the benchmark spec file and runs all cases (tiers included).
        """
        spec = json.loads(Path(benchmark_json_path).read_text())

        results: List[Dict[str, Any]] = []

        # Run all tiers in order, preserving IDs
        for tier_name, cases in spec["tiers"].items():
            for case in cases:
                # Keep tier in case dict for traceability
                case = dict(case)
                case["tier"] = tier_name
                res = self.run_case(case, hazard_model_id=hazard_model_id)
                res["tier"] = tier_name
                results.append(res)

        # Optional: write a small run log
        root = self._benchmark_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / f"run_log_{hazard_model_id}.json").write_text(json.dumps(results, indent=2))

        return results
