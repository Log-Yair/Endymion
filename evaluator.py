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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

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

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_nav_meta(self, path: Path) -> dict:
        """
        Loads nav_meta.json safely.
        Handles legacy 'Infinity' values present in corridor experiments.
        """
        s = path.read_text()
        s = s.replace(": Infinity", ": 1e309")
        s = s.replace(": -Infinity", ": -1e309")
        s = s.replace(": NaN", ": null")
        return json.loads(s)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _step_lengths_px(self) -> np.ndarray:
        """Pixel step lengths between consecutive path nodes."""
        if self.path_rc.shape[0] < 2:
            return np.zeros((0,), dtype=float)
        d = np.diff(self.path_rc, axis=0)
        return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)

    def _count_turns(self) -> int:
        """Counts direction changes as a proxy for path smoothness."""
        if self.path_rc.shape[0] < 3:
            return 0
        d = np.diff(self.path_rc, axis=0)
        d = np.clip(d, -1, 1)
        return int(np.sum(np.any(d[1:] != d[:-1], axis=1)))

    # ------------------------------------------------------------------
    # Small numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_or_none(x: float) -> Optional[float]:
        """Convert NaN/Inf to None so JSON stays strict."""
        return float(x) if np.isfinite(x) else None

    @staticmethod
    def _safe_percentile(arr: np.ndarray, q: float) -> Optional[float]:
        """Percentile that returns None for empty input."""
        if arr.size == 0:
            return None
        return float(np.percentile(arr, q))

    @staticmethod
    def _safe_mean(arr: np.ndarray) -> Optional[float]:
        return float(np.mean(arr)) if arr.size else None

    @staticmethod
    def _safe_max(arr: np.ndarray) -> Optional[float]:
        return float(np.max(arr)) if arr.size else None

    @staticmethod
    def _safe_sum(arr: np.ndarray) -> float:
        return float(np.sum(arr)) if arr.size else 0.0

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """
        Computes safety + efficiency metrics for the run.
        Safe for both successful and failed runs.
        """
        # Pull success/failure info from meta (benchmark runner adds this)
        result = self.nav.get("result", {})
        success = bool(result.get("success", False))
        failure_reason = result.get("failure_reason", None)

        # If no path, return minimal metrics safely.
        if self.path_rc.size == 0 or self.path_rc.shape[0] == 0:
            return {
                "run": {
                    "run_name": self.meta.get("run_name"),
                    "tile_id": self.meta.get("tile_id"),
                    "roi": self.meta.get("roi"),
                },
                "status": {
                    "success": success,
                    "failure_reason": failure_reason,
                },
                "geometry": {
                    "path_nodes": 0,
                    "path_length_m": 0.0,
                    "straight_line_m": None,
                    "detour_ratio": None,
                    "turn_count": 0,
                    "turns_per_m": None,
                },
                "safety": {
                    "path_hazard_mean": None,
                    "path_hazard_max": None,
                    "path_hazard_p95": None,
                    "haz_per_m": None,
                    "frac_hazard_ge_block": None,
                    "hazard_mean_ratio": None,
                    "safety_score": None,
                },
                "efficiency": {
                    "cost_sum_along_path": None,
                    "cost_per_m": None,
                    "planner_total_cost": result.get("total_cost", None),
                    "expansions": result.get("expansions", None),
                },
            }

        rr = self.path_rc[:, 0].astype(int)
        cc = self.path_rc[:, 1].astype(int)

        H = self.hazard[rr, cc].astype(float)
        C = self.cost[rr, cc].astype(float)

        Hf = H[np.isfinite(H)]
        Cf = C[np.isfinite(C)]

        # ---- geometry ----
        step_px = self._step_lengths_px()
        path_len_px = float(np.sum(step_px)) if step_px.size else 0.0
        path_len_m = path_len_px * self.pixel_size_m

        sr, sc = self.nav["start_rc"]
        gr, gc = self.nav["goal_rc"]

        straight_px = float(np.sqrt((gr - sr) ** 2 + (gc - sc) ** 2))
        straight_m = straight_px * self.pixel_size_m

        detour = (path_len_m / straight_m) if straight_m > 0 else np.nan

        turn_count = self._count_turns()
        turns_per_m = (turn_count / path_len_m) if path_len_m > 0 else np.nan

        # ---- safety ----
        haz_mean = self._safe_mean(Hf)
        haz_max = self._safe_max(Hf)
        haz_p95 = self._safe_percentile(Hf, 95)
        haz_sum = self._safe_sum(Hf)
        haz_per_m = (haz_sum / path_len_m) if path_len_m > 0 else np.nan

        hazard_block = float(self.nav["cost_model"]["hazard_block"])
        frac_blocked = float(np.mean(Hf >= hazard_block)) if Hf.size else np.nan

        global_mean = float(self.nav["hazard_assessor"]["stats"]["hazard_mean"])
        haz_ratio = (float(haz_mean) / global_mean) if (haz_mean is not None and global_mean > 0) else np.nan

        safety_score = (1.0 / (1.0 + haz_per_m)) if np.isfinite(haz_per_m) else np.nan

        # ---- efficiency ----
        cost_sum = self._safe_sum(Cf)
        cost_per_m = (cost_sum / path_len_m) if path_len_m > 0 else np.nan

        return {
            "run": {
                "run_name": self.meta.get("run_name"),
                "tile_id": self.meta.get("tile_id"),
                "roi": self.meta.get("roi"),
            },
            "status": {
                "success": success,
                "failure_reason": failure_reason,
            },
            "geometry": {
                "path_nodes": int(self.path_rc.shape[0]),
                "path_length_m": self._finite_or_none(path_len_m),
                "straight_line_m": self._finite_or_none(straight_m),
                "detour_ratio": self._finite_or_none(detour),
                "turn_count": int(turn_count),
                "turns_per_m": self._finite_or_none(turns_per_m),
            },
            "safety": {
                "path_hazard_mean": self._finite_or_none(haz_mean) if haz_mean is not None else None,
                "path_hazard_max": self._finite_or_none(haz_max) if haz_max is not None else None,
                "path_hazard_p95": self._finite_or_none(haz_p95) if haz_p95 is not None else None,
                "haz_per_m": self._finite_or_none(haz_per_m),
                "frac_hazard_ge_block": self._finite_or_none(frac_blocked),
                "hazard_mean_ratio": self._finite_or_none(haz_ratio),
                "safety_score": self._finite_or_none(safety_score),
            },
            "efficiency": {
                "cost_sum_along_path": self._finite_or_none(cost_sum),
                "cost_per_m": self._finite_or_none(cost_per_m),
                "planner_total_cost": result.get("total_cost", None),
                "expansions": result.get("expansions", None),
            },
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, filename: str = "metrics.json") -> Path:
        """
        Writes metrics.json next to nav_meta.json.
        Uses strict JSON: disallows NaN/Inf.
        """
        metrics = self.evaluate()
        out_path = self.run_dir / filename
        out_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
        return out_path
