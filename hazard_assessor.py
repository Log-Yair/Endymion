# -*- coding: utf-8 -*-
"""
HazardAssessor for Endymion (Phase 1)

Consumes cached terrain rasters and produces a unified hazard map.
No feature extraction or ML inference is performed here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class HazardAssessor:
    """
    Phase-1 hazard model based on terrain difficulty.

    Hazard is computed as a weighted combination of:
    - slope (degrees)
    - surface roughness (RMS)
    """

    # -------------------------
    # Normalisation ceilings (prototype defaults; override in pipeline)
    # -------------------------
    slope_deg_max: float = 34.55          
    roughness_rms_max: float = 18.56      

    # -------------------------
    # Weights (should sum to 1.0)
    # -------------------------
    w_slope: float = 0.7
    w_roughness: float = 0.3

    # -------------------------
    # Optional "impassable" mask (prototype safety rule)
    # -------------------------
    use_impassable_mask: bool = True
    impassable_slope_deg: float = 40.0

    def assess(
        self,
        slope_deg: np.ndarray,
        roughness_rms: np.ndarray,
        dem_m: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute a hazard map from cached terrain rasters.
        dem_m is unused in Phase 1 (kept for extensibility).
        """
        self._validate_inputs(slope_deg, roughness_rms)

        # Normalise to [0,1] using robust ceilings (vmax)
        slope_n = self._normalise(slope_deg, vmax=self.slope_deg_max)
        rough_n = self._normalise(roughness_rms, vmax=self.roughness_rms_max)

        # Weighted hazard index
        hazard = (self.w_slope * slope_n + self.w_roughness * rough_n).astype(np.float32)

        # Optional hard cutoff for extreme slopes
        if self.use_impassable_mask:
            hazard[slope_deg > self.impassable_slope_deg] = 1.0

        # Metadata for traceability (stats reflect FINAL hazard)
        meta = {
            "model": "terrain_weighted_v1",
            "weights": {"slope": self.w_slope, "roughness": self.w_roughness},
            "thresholds": {
                "slope_deg_max": self.slope_deg_max,
                "roughness_rms_max": self.roughness_rms_max,
                "use_impassable_mask": self.use_impassable_mask,
                "impassable_slope_deg": self.impassable_slope_deg,
            },
            "stats": {
                "hazard_min": float(np.nanmin(hazard)),
                "hazard_max": float(np.nanmax(hazard)),
                "hazard_mean": float(np.nanmean(hazard)),
                "nan_ratio": float(np.isnan(hazard).mean()),
            },
        }

        return {
            "hazard": hazard,
            "components": {"slope_norm": slope_n, "roughness_norm": rough_n},
            "meta": meta,
        }

    @staticmethod
    def _normalise(arr: np.ndarray, vmax: float) -> np.ndarray:
        """
        Clip and normalise array to [0,1].
        NaNs are preserved.
        """
        out = arr.astype(np.float32, copy=False) / float(vmax)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _validate_inputs(a: np.ndarray, b: np.ndarray) -> None:
        """Ensure arrays are compatible for elementwise hazard computation."""
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")
        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match: {a.shape} vs {b.shape}")
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Inputs must be 2D rasters. Got {a.ndim}D and {b.ndim}D.")