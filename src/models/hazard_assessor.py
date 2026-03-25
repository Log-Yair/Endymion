# References / notes:
# - Extends the existing Endymion Phase-1 HazardAssessor into a Phase-2 pipeline-ready version.
# - Keeps backwards compatibility with terrain-only mode from the previous hazard_assessor.py.
# - Designed to work with:
#     * FeatureExtractor outputs: slope_deg, roughness_rms
#     * CraterPredictor outputs: crater_mask, crater_distance_m, crater_density
# - Main design idea:
#     terrain_component = weighted slope + roughness
#     crater_component  = weighted crater mask + crater distance risk + crater density
#     final_hazard      = terrain_weight * terrain_component + crater_weight * crater_component
# - This keeps the model simple and explainable for the FYP, while leaving room for future extensions such as ML crater probability, illumination, or other hazard factors.

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
    # Terrain weights (inside terrain group)
    # ------------------------------------------------------------------
    w_slope: float = 0.7
    w_roughness: float = 0.3

    # ------------------------------------------------------------------
    # Crater normalisation / influence controls
    # ------------------------------------------------------------------
    crater_distance_safe_m: float = 200.0
    crater_density_max: float = 0.25

    # ------------------------------------------------------------------
    # Crater weights (inside crater group)
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
        *,
        slope_deg: np.ndarray,
        roughness_rms: np.ndarray,
        dem_m: Optional[np.ndarray] = None,
        crater_mask: Optional[np.ndarray] = None,
        crater_distance_m: Optional[np.ndarray] = None,
        crater_density: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute a unified hazard map.

        Required inputs:
        
        slope_deg : np.ndarray
            Terrain slope in degrees.
        roughness_rms : np.ndarray
            Local terrain roughness.

        Optional crater inputs:
        
        crater_mask : np.ndarray
            Binary crater footprint raster aligned to the ROI grid.
        crater_distance_m : np.ndarray
            Distance-to-nearest-crater raster in meters.
        crater_density : np.ndarray
            Local crater density raster.

        Returns:
        
        dict with:
        - hazard
        - components
        - meta
        """
        self._validate_inputs(
            slope_deg=slope_deg,
            roughness_rms=roughness_rms,
            crater_mask=crater_mask,
            crater_distance_m=crater_distance_m,
            crater_density=crater_density,
        )

        # --------------------------------------------------------------
        # Terrain component
        # --------------------------------------------------------------
        slope_n = self._normalise(slope_deg, vmax=self.slope_deg_max)
        rough_n = self._normalise(roughness_rms, vmax=self.roughness_rms_max)

        terrain_component = (
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

        if crater_inputs_present:
            crater_mask_n = self._prepare_crater_mask(
                crater_mask=crater_mask,
                expected_shape=slope_deg.shape,
            )

            crater_distance_risk = self._distance_to_risk(
                crater_distance_m=crater_distance_m,
                expected_shape=slope_deg.shape,
            )

            crater_density_n = self._prepare_crater_density(
                crater_density=crater_density,
                expected_shape=slope_deg.shape,
            )

            crater_component = (
                self.w_crater_mask * crater_mask_n +
                self.w_crater_distance * crater_distance_risk +
                self.w_crater_density * crater_density_n
            ).astype(np.float32)

            hazard = (
                self.terrain_weight * terrain_component +
                self.crater_weight * crater_component
            ).astype(np.float32)

            model_id = "terrain_crater_weighted_v2"

        else:
            # Backwards-compatible terrain-only path
            crater_mask_n = np.zeros_like(slope_n, dtype=np.float32)
            crater_distance_risk = np.zeros_like(slope_n, dtype=np.float32)
            crater_density_n = np.zeros_like(slope_n, dtype=np.float32)
            crater_component = np.zeros_like(slope_n, dtype=np.float32)

            hazard = terrain_component.astype(np.float32)
            model_id = "terrain_weighted_v1_compatible"

        # --------------------------------------------------------------
        # Optional hard cutoff for extreme slopes
        # --------------------------------------------------------------
        if self.use_impassable_mask:
            hazard = hazard.copy()
            hazard[slope_deg > self.impassable_slope_deg] = 1.0

        # Final safety clamp
        hazard = np.clip(hazard, 0.0, 1.0).astype(np.float32)

        # dem_m is currently unused, but we keep a small trace of whether it was supplied
        dem_present = dem_m is not None

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
            "inputs_present": {
                "dem_m": dem_present,
                "crater_mask": crater_mask is not None,
                "crater_distance_m": crater_distance_m is not None,
                "crater_density": crater_density is not None,
            },
            "stats": {
                "hazard_min": float(np.nanmin(hazard)),
                "hazard_max": float(np.nanmax(hazard)),
                "hazard_mean": float(np.nanmean(hazard)),
                "nan_ratio": float(np.isnan(hazard).mean()),
                "terrain_mean": float(np.nanmean(terrain_component)),
                "crater_mean": float(np.nanmean(crater_component)) if crater_inputs_present else 0.0,
            },
        }

        return {
            "hazard": hazard,
            "components": {
                "slope_norm": slope_n,
                "roughness_norm": rough_n,
                "terrain_component": terrain_component,
                "crater_mask_norm": crater_mask_n,
                "crater_distance_risk": crater_distance_risk,
                "crater_density_norm": crater_density_n,
                "crater_component": crater_component,
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
        Convert crater distance into a crater-risk score in [0,1].

        Rule:
        - distance = 0                 -> risk = 1
        - distance >= safe distance    -> risk = 0
        """
        if crater_distance_m is None:
            return np.zeros(expected_shape, dtype=np.float32)

        arr = crater_distance_m.astype(np.float32, copy=False)
        risk = 1.0 - np.clip(arr / float(self.crater_distance_safe_m), 0.0, 1.0)
        return risk.astype(np.float32)

    def _prepare_crater_density(
        self,
        crater_density: Optional[np.ndarray],
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Normalise crater density into [0,1].
        """
        if crater_density is None:
            return np.zeros(expected_shape, dtype=np.float32)

        return self._normalise(crater_density, vmax=self.crater_density_max)

    @staticmethod
    def _prepare_crater_mask(
        crater_mask: Optional[np.ndarray],
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Convert crater mask to float32 in [0,1].
        """
        if crater_mask is None:
            return np.zeros(expected_shape, dtype=np.float32)

        arr = crater_mask.astype(np.float32, copy=False)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _validate_same_shape(
        ref: np.ndarray,
        arr: Optional[np.ndarray],
        name: str,
    ) -> None:
        if arr is None:
            return
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array.")
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D raster. Got {arr.ndim}D.")
        if arr.shape != ref.shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match reference shape {ref.shape}."
            )

    def _validate_inputs(
        self,
        *,
        slope_deg: np.ndarray,
        roughness_rms: np.ndarray,
        crater_mask: Optional[np.ndarray],
        crater_distance_m: Optional[np.ndarray],
        crater_density: Optional[np.ndarray],
    ) -> None:
        if not isinstance(slope_deg, np.ndarray) or not isinstance(roughness_rms, np.ndarray):
            raise TypeError("slope_deg and roughness_rms must be numpy arrays.")

        if slope_deg.ndim != 2 or roughness_rms.ndim != 2:
            raise ValueError(
                f"Inputs must be 2D rasters. Got {slope_deg.ndim}D and {roughness_rms.ndim}D."
            )

        if slope_deg.shape != roughness_rms.shape:
            raise ValueError(
                f"Input shapes must match: slope_deg {slope_deg.shape} vs roughness_rms {roughness_rms.shape}"
            )

        self._validate_same_shape(slope_deg, crater_mask, "crater_mask") # crater_mask is expected to be binary, but we still validate its shape if provided
        self._validate_same_shape(slope_deg, crater_distance_m, "crater_distance_m") # crater_distance_m is expected to be a float raster, but we still validate its shape if provided
        self._validate_same_shape(slope_deg, crater_density, "crater_density") # crater_density is expected to be a float raster, but we still validate its shape if provided