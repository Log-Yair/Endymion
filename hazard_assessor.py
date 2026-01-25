
# -*- coding: utf-8 -*-
"""
HazardAssessor for Endymion (Phase 1)

Consumes cached terrain rasters and produces a unified hazard map.
No feature extraction or ML inference is performed here.

Author: Endymion Project
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

    All inputs must be aligned and already standardised.
    """

    # -------------------------
    # Normalisation thresholds
    # -------------------------
    slope_deg_max: float = 30.0        # beyond this is effectively non-traversable
    roughness_rms_max: float = 1.0     # metres (tunable later)

    # -------------------------
    # Weights (must sum to 1.0)
    # -------------------------
    w_slope: float = 0.7
    w_roughness: float = 0.3

    def assess(
        self,
        slope_deg: np.ndarray,
        roughness_rms: np.ndarray,
        dem_m: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute a hazard map from cached terrain rasters.

        Parameters
        ----------
        slope_deg : np.ndarray
            Local slope in degrees.
        roughness_rms : np.ndarray
            Local RMS roughness in metres.
        dem_m : np.ndarray, optional
            Elevation raster (unused in Phase 1, kept for extensibility).

        Returns
        -------
        dict with:
            - hazard : np.ndarray (float32, [0,1])
            - components : dict of intermediate layers
            - meta : dict of parameters and statistics
        """

        self._validate_inputs(slope_deg, roughness_rms)

        # -------------------------
        # Normalise inputs to [0,1]
        # -------------------------
        slope_n = self._normalise(
            slope_deg, vmax=self.slope_deg_max
        )
        rough_n = self._normalise(
            roughness_rms, vmax=self.roughness_rms_max
        )

        # -------------------------
        # Weighted hazard index
        # -------------------------
        hazard = (
            self.w_slope * slope_n +
            self.w_roughness * rough_n
        ).astype(np.float32)

        # -------------------------
        # Metadata for traceability
        # -------------------------
        meta = {
            "model": "terrain_weighted_v1",
            "weights": {
                "slope": self.w_slope,
                "roughness": self.w_roughness,
            },
            "thresholds": {
                "slope_deg_max": self.slope_deg_max,
                "roughness_rms_max": self.roughness_rms_max,
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
            "components": {
                "slope_norm": slope_n,
                "roughness_norm": rough_n,
            },
            "meta": meta,
        }

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _normalise(arr: np.ndarray, vmax: float) -> np.ndarray:
        """
        Clip and normalise array to [0,1].
        NaNs are preserved.
        """
        out = arr.astype(np.float32, copy=False) / float(vmax)
        out = np.clip(out, 0.0, 1.0)
        return out

    @staticmethod
    def _validate_inputs(a: np.ndarray, b: np.ndarray) -> None:
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")
        if a.shape != b.shape:
            raise ValueError(
                f"Input shapes must match: {a.shape} vs {b.shape}"
            )