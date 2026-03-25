# References / notes:
# - Extends the existing Endymion HazardAssessor (terrain_weighted_v1).
# - Keeps backwards compatibility with Phase-1 usage.
# - Designed to accept crater products exposed by CraterPredictor:
#   crater_mask, crater_distance_m, crater_density.
# - Main idea:
#     terrain = weighted slope + roughness
#     crater  = weighted mask + distance-risk + density
#     hazard  = terrain_weight * terrain + crater_weight * crater


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class HazardAssessor:
    """
    Endymion HazardAssessor (Phase 2)

    Supports:
    - Phase 1 terrain-only hazard
    - Phase 2 terrain + crater-aware hazard

    Backwards compatibility:
    If crater inputs are not provided, the model behaves like the original
    terrain-only version.
    """

    # ------------------------------------------------------------------
    # Terrain normalisation ceilings
    # ------------------------------------------------------------------
    slope_deg_max: float = 34.55
    roughness_rms_max: float = 18.56

    # ------------------------------------------------------------------
    # Terrain weights (within terrain group)
    # ------------------------------------------------------------------
    w_slope: float = 0.7
    w_roughness: float = 0.3

    # ------------------------------------------------------------------
    # Crater normalisation / risk controls
    # ------------------------------------------------------------------
    crater_distance_safe_m: float = 200.0
    crater_density_max: float = 0.25

    # ------------------------------------------------------------------
    # Crater weights (within crater group)
    # ------------------------------------------------------------------
    w_crater_mask: float = 0.4
    w_crater_distance: float = 0.4
    w_crater_density: float = 0.2

    # ------------------------------------------------------------------
    # Group-level weights
    # ------------------------------------------------------------------
    terrain_weight: float = 0.7
    crater_weight: float = 0.3

    # ------------------------------------------------------------------
    # Optional hard safety rule
    # ------------------------------------------------------------------
    use_impassable_mask: bool = True
    impassable_slope_deg: float = 40.0

    def assess(
        self,
        slope_deg: np.ndarray,
        roughness_rms: np.ndarray,
        dem_m: Optional[np.ndarray] = None,
        crater_mask: Optional[np.ndarray] = None,
        crater_distance_m: Optional[np.ndarray] = None,
        crater_density: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute a unified hazard map.

        Required:
        - slope_deg
        - roughness_rms

        Optional crater inputs:
        - crater_mask
        - crater_distance_m
        - crater_density

        Returns
        -------
        dict with:
        - hazard
        - components
        - meta
        """
        self._validate_inputs(
            slope_deg = slope_deg,
            roughness_rms = roughness_rms,
            crater_mask = crater_mask,
            crater_distance_m = crater_distance_m,
            crater_density = crater_density,
        )

        # Terrain component
        slope_n = self._normalise(slope_deg, vmax=self.slope_deg_max)
        rough_n = self._normalise(roughness_rms, vmax=self.roughness_rms_max)

        terrain = (
            self.w_slope * slope_n +
            self.w_roughness * rough_n
        ).astype(np.float32)

        # --------------------------------------------------------------
        # Crater component
        # --------------------------------------------------------------
        crater_inputs_present = any(
            arr is not None
            for arr in (crater_mask, crater_distance_m, crater_density)
        )

        # If any crater input is provided, require all three for a complete crater component.
        if crater_inputs_present:
            crater_mask_n = self._prepare_crater_mask(crater_mask, slope_deg.shape)
            crater_distance_risk = self._distance_to_risk(crater_distance_m, slope_deg.shape)
            crater_density_n = self._prepare_crater_density(crater_density, slope_deg.shape)

            # Combine crater factors into a single crater risk score.
            crater = (
                self.w_crater_mask * crater_mask_n +
                self.w_crater_distance * crater_distance_risk +
                self.w_crater_density * crater_density_n
            ).astype(np.float32)

            # Combine terrain and crater into final hazard score.
            hazard = (
                self.terrain_weight * terrain +
                self.crater_weight * crater
            ).astype(np.float32)

            model_id = "terrain_crater_weighted_v2"
        else:
            crater_mask_n = np.zeros_like(slope_n, dtype=np.float32)
            crater_distance_risk = np.zeros_like(slope_n, dtype=np.float32)
            crater_density_n = np.zeros_like(slope_n, dtype=np.float32)
            crater = np.zeros_like(slope_n, dtype=np.float32)

            hazard = terrain.astype(np.float32)
            model_id = "terrain_weighted_v1_compatible"


        # --------------------------------------------------------------
        # Optional hard cutoff for extreme slopes
        # --------------------------------------------------------------
        if self.use_impassable_mask:
            hazard = hazard.copy()
            hazard[slope_deg > self.impassable_slope_deg] = 1.0

        hazard = np.clip(hazard, 0.0, 1.0).astype(np.float32)

        meta = {
            "model": model_id,
            "groups": {
                "terrain_weight": self.terrain_weight,
                "crater_weight": self.crater_weight if crater_inputs_present else 0.0,
            },
            "weights": {
                "terrain": {
                    "slope": self.w_slope,
                    "roughness": self.w_roughness,
                },
                "crater": {
                    "mask": self.w_crater_mask,
                    "distance": self.w_crater_distance,
                    "density": self.w_crater_density,
                },
            },
            "thresholds": {
                "slope_deg_max": self.slope_deg_max,
                "roughness_rms_max": self.roughness_rms_max,
                "crater_distance_safe_m": self.crater_distance_safe_m,
                "crater_density_max": self.crater_density_max,
                "use_impassable_mask": self.use_impassable_mask,
                "impassable_slope_deg": self.impassable_slope_deg,
            },
            "crater_inputs_present": {
                "crater_mask": crater_mask is not None,
                "crater_distance_m": crater_distance_m is not None,
                "crater_density": crater_density is not None,
            },
            "stats": {
                "hazard_min": float(np.nanmin(hazard)),
                "hazard_max": float(np.nanmax(hazard)),
                "hazard_mean": float(np.nanmean(hazard)),
                "nan_ratio": float(np.isnan(hazard).mean()),
                "terrain_mean": float(np.nanmean(terrain)),
                "crater_mean": float(np.nanmean(crater)) if crater_inputs_present else 0.0,
            },
        }

        return {
            "hazard": hazard,
            "components": {
                "slope_norm": slope_n,
                "roughness_norm": rough_n,
                "terrain_component": terrain,
                "crater_mask_norm": crater_mask_n,
                "crater_distance_risk": crater_distance_risk,
                "crater_density_norm": crater_density_n,
                "crater_component": crater,
            },
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(arr: np.ndarray, vmax: float) -> np.ndarray:
        """
        Clip and normalise array to [0,1].
        NaNs are preserved.
        """
        out = arr.astype(np.float32, copy=False) / float(vmax)
        return np.clip(out, 0.0, 1.0)

    def _distance_to_risk(
        self,
        crater_distance_m: Optional[np.ndarray],
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        
        """
        Convert crater distance to a risk score between 0 and 1
        Rule : 
        - distance = 0 -> risk = 1 (high risk)
        - distance >= safe -> risk = 0 (low risk)
        """
        if crater_distance_m is None:
            return np.zeros(expected_shape, dtype=np.float32)
        arr = crater_distance_m.astype(np.float32, copy=False)
        risk = 1.0 - np.clip(arr / float(self.crater_distance_safe_m), 0.0, 1.0)
        return risk.astype(np.float32)
    

    

    @staticmethod
    def _validate_inputs(a: np.ndarray, b: np.ndarray) -> None:
        """Ensure arrays are compatible for elementwise hazard computation."""
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")
        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match: {a.shape} vs {b.shape}")
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Inputs must be 2D rasters. Got {a.ndim}D and {b.ndim}D.")