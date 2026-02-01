# -*- coding: utf-8 -*-
"""
Evaluator for Endymion (Prototype)

Evaluates a completed navigation run using:
- hazard.npy
- cost.npy
- path_rc.npy
- nav_meta.json

Outputs:
- metrics.json (safety + efficiency baseline)

Author: El yayo
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class Evaluator:
    """
    Baseline navigation evaluator.

    One instance == one navigation run directory, e.g.
    .../roi_xxxx_xxxx_xxxx_xxxx/navigation_v1/
    """

    def __init__(self, run_dir: str | Path, pixel_size_m: float = 20.0):
        self.run_dir = Path(run_dir)
        self.pixel_size_m = float(pixel_size_m)

        # --- Required artifacts ---
        self.hazard = np.load(self.run_dir / "hazard.npy")
        self.cost = np.load(self.run_dir / "cost.npy")
        self.path_rc = np.load(self.run_dir / "path_rc.npy")
        self.meta = self._load_nav_meta(self.run_dir / "nav_meta.json")

        self.nav = self.meta["nav_meta"]

    
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_nav_meta(self, path: Path) -> dict:
        """
        Loads nav_meta.json safely.
        Handles 'Infinity' values present in corridor experiments.
        """
        s = path.read_text()
        s = s.replace(": Infinity", ": 1e309")
        s = s.replace(": -Infinity", ": -1e309")
        s = s.replace(": NaN", ": null")
        return json.loads(s)

    # ------------------------------------------------------- 
    # Geometry
    # -----------------------------------------------------

    def _step_lengths_px(self) -> np.ndarray:
        """Pixel step lengths between consecutive path nodes."""
        d = np.diff(self.path_rc, axis=0)
        return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

    def _count_turns(self) -> int:
        """Counts direction changes as a proxy for path smoothness."""
        d = np.diff(self.path_rc, axis=0)
        d = np.clip(d, -1, 1)
        return int(np.sum(np.any(d[1:] != d[:-1], axis=1)))

    # ----------------------------------------------------------
    # Main evaluation
    # ---------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Computes safety + efficiency metrics for the run.
        """

        rr = self.path_rc[:, 0].astype(int)
        cc = self.path_rc[:, 1].astype(int)

        H = self.hazard[rr, cc].astype(float)
        C = self.cost[rr, cc].astype(float)

        Hf = H[np.isfinite(H)]
        Cf = C[np.isfinite(C)]

        # ---- geometry ----
        step_px = self._step_lengths_px()
        path_len_px = float(np.sum(step_px))
        path_len_m = path_len_px * self.pixel_size_m

        sr, sc = self.nav["start_rc"]
        gr, gc = self.nav["goal_rc"]

        straight_px = np.sqrt((gr - sr) ** 2 + (gc - sc) ** 2)
        straight_m = straight_px * self.pixel_size_m

        detour = path_len_m / straight_m if straight_m > 0 else np.nan

        turn_count = self._count_turns()
        turns_per_m = turn_count / path_len_m if path_len_m > 0 else np.nan

        # ---- safety ----
        haz_mean = float(np.mean(Hf))
        haz_max = float(np.max(Hf))
        haz_p95 = float(np.percentile(Hf, 95))
        haz_sum = float(np.sum(Hf))
        haz_per_m = haz_sum / path_len_m

        hazard_block = float(self.nav["cost_model"]["hazard_block"])
        frac_blocked = float(np.mean(Hf >= hazard_block))

        global_mean = float(self.nav["hazard_assessor"]["stats"]["hazard_mean"])
        haz_ratio = haz_mean / global_mean

        safety_score = 1.0 / (1.0 + haz_per_m)

        # ---- efficiency ----
        cost_sum = float(np.sum(Cf))
        cost_per_m = cost_sum / path_len_m

        result = self.nav["result"]

        return {
            "run": {
                "run_name": self.meta.get("run_name"),
                "tile_id": self.meta.get("tile_id"),
                "roi": self.meta.get("roi"),
            },
            "geometry": {
                "path_nodes": int(self.path_rc.shape[0]),
                "path_length_m": path_len_m,
                "straight_line_m": straight_m,
                "detour_ratio": detour,
                "turn_count": turn_count,
                "turns_per_m": turns_per_m,
            },
            "safety": {
                "path_hazard_mean": haz_mean,
                "path_hazard_max": haz_max,
                "path_hazard_p95": haz_p95,
                "haz_per_m": haz_per_m,
                "frac_hazard_ge_block": frac_blocked,
                "hazard_mean_ratio": haz_ratio,
                "safety_score": safety_score,
            },
            "efficiency": {
                "cost_sum_along_path": cost_sum,
                "cost_per_m": cost_per_m,
                "planner_total_cost": result["total_cost"],
                "expansions": result["expansions"],
            }
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, filename: str = "metrics.json") -> Path:
        """
        Writes metrics.json next to nav_meta.json.
        """
        metrics = self.evaluate()
        out_path = self.run_dir / filename
        out_path.write_text(json.dumps(metrics, indent=2))
        return out_path
