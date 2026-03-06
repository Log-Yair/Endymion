# -*- coding: utf-8 -*-
"""
BenchmarkRunner for Endymion

Runs a fixed benchmark suite (start/goal pairs) against one or more hazard models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_handler import DataHandler, ROI
from hazard_assessor import HazardAssessor
from pathfinder import Pathfinder, build_cost_from_hazard
from evaluator import Evaluator

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

    Expects derived rasters to already exist:
      dem_m.npy, slope_deg.npy, roughness_rms.npy + meta.json
    """

    def __init__(self, dh: DataHandler, cfg: BenchmarkConfig):
        self.dh = dh
        self.cfg = cfg

        self.hazard_assessor = HazardAssessor()
        self.pathfinder = Pathfinder(
            connectivity=cfg.connectivity,
            block_cost=cfg.block_cost,
            heuristic_weight=cfg.heuristic_weight,
        )

    def _benchmark_root(self) -> Path:
        """Root directory where all benchmark results for this ROI live."""
        return self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi) / "benchmarks" / self.cfg.benchmark_id

    def _case_dir(self, case_id: str, hazard_model_id: str) -> Path:
        """Directory for one case and one hazard model variant."""
        return self._benchmark_root() / case_id / hazard_model_id

    def run_case(self, case: Dict[str, Any], hazard_model_id: str = "terrain_weighted_v1") -> Dict[str, Any]:
        """Runs one start/goal pair and saves artifacts."""
        case_id = case["id"]
        start: RC = tuple(case["start_rc"])
        goal: RC = tuple(case["goal_rc"])

        derived = self.dh.load_derived(self.cfg.tile_id, self.cfg.roi)
        if derived is None:
            raise RuntimeError(
                "Derived rasters not found. Run the pipeline once to generate "
                "dem_m, slope_deg, roughness_rms in the derived cache."
            )

        dem_m = derived["dem_m"]
        slope_deg = derived["slope_deg"]
        roughness_rms = derived["roughness_rms"]

        haz_out = self.hazard_assessor.assess(slope_deg=slope_deg, roughness_rms=roughness_rms, dem_m=dem_m)
        hazard = haz_out["hazard"]
        hazard_meta = haz_out["meta"]

        cost = build_cost_from_hazard(
            hazard,
            alpha=self.cfg.alpha,
            hazard_block=self.cfg.hazard_block,
            block_cost=self.cfg.block_cost,
        )

        corr = self.pathfinder.find_path_corridor(
            cost=cost,
            start=start,
            goal=goal,
            corridor_radii=self.cfg.corridor_radii,
            hazard=hazard,
        )

        best = corr.get("best", None)
        best_radius = corr.get("best_radius_px", None)  # robust if missing
        experiment = corr.get("experiment", [])

        success = bool(best and best.get("success", False))
        path_rc = np.asarray(best["path_rc"], dtype=np.int32) if success else np.zeros((0, 2), dtype=np.int32)

        out_dir = self._case_dir(case_id, hazard_model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "hazard.npy", hazard.astype(np.float32))
        np.save(out_dir / "cost.npy", cost.astype(np.float32))
        np.save(out_dir / "path_rc.npy", path_rc)

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
                    "total_cost": float(best["total_cost"]) if success else None,
                    "path_len": int(path_rc.shape[0]),
                    "expansions": int((best.get("meta") or {}).get("expansions", -1)) if success else int((best.get("meta") or {}).get("expansions", -1)),
                    "failure_reason": None if success else "no_path_found_in_any_corridor",
                },
                "benchmark": {
                    "benchmark_id": self.cfg.benchmark_id,
                    "case_id": case_id,
                    "tier": case.get("tier"),
                    "hazard_model_id": hazard_model_id,
                },
                "corridor_experiment": experiment,
            },
        }

        (out_dir / "nav_meta.json").write_text(json.dumps(nav_meta, indent=2, allow_nan=False))

        evaluator = Evaluator(out_dir, pixel_size_m=self.cfg.pixel_size_m)
        evaluator.save("metrics.json")

        return {
            "case_id": case_id,
            "hazard_model_id": hazard_model_id,
            "success": success,
            "out_dir": str(out_dir),
        }

    def run_benchmark_file(self, benchmark_json_path: str | Path, hazard_model_id: str = "terrain_weighted_v1") -> List[Dict[str, Any]]:
        """Loads the benchmark spec file and runs all cases (tiers included)."""
        spec = json.loads(Path(benchmark_json_path).read_text())

        # Optional safety check if spec provides these fields
        if "tile_id" in spec and spec["tile_id"] != self.cfg.tile_id:
            raise ValueError(f"Benchmark spec tile_id={spec['tile_id']} does not match cfg.tile_id={self.cfg.tile_id}")
        if "roi" in spec and tuple(spec["roi"]) != tuple(self.cfg.roi):
            raise ValueError(f"Benchmark spec roi={spec['roi']} does not match cfg.roi={list(self.cfg.roi)}")

        results: List[Dict[str, Any]] = []

        for tier_name, cases in spec["tiers"].items():
            for case in cases:
                case = dict(case)
                case["tier"] = tier_name
                res = self.run_case(case, hazard_model_id=hazard_model_id)
                res["tier"] = tier_name
                results.append(res)

        root = self._benchmark_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / f"run_log_{hazard_model_id}.json").write_text(json.dumps(results, indent=2, allow_nan=False))

        return results
    
