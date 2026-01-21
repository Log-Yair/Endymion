# -*- coding: utf-8 -*-
"""
feature_extractor for Endymion
"""

from __future__ import annotations
# --- Core ---
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

RoughnessMethod = Literal["rms", "tri"]


@dataclass
class FeatureExtractor:
    """
    Endymion FeatureExtractor (Phase 1):
    - Input: a DEM ROI (H x W) in meters.
    - Output: slope + roughness feature rasters aligned to the input grid.

    Assumptions for your canonical ROI:
    - shape == (1024, 1024)
    - units == meters
    - NaN ratio ideally ~0 (but we handle NaNs safely anyway)
    """
    pixel_size_m: float = 20.0
    slope_units: str = "degrees"
    expected_shape: Optional[Tuple[int, int]] = (1024, 1024)  # set to None to disable
    strict_shape: bool = False  # if True, mismatch raises; if False, just allows it


    def extract(
        self,
        dem_m: np.ndarray,
        roughness_method: RoughnessMethod = "rms",
        roughness_window: int = 5,
        pad_mode: Literal["edge", "reflect"] = "edge",
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dict of feature rasters (float32):
          - slope_deg or slope_rad
          - slope_rise_run
          - roughness_<method>
        """
        self._validate_dem(dem_m)
        dem = dem_m.astype(np.float32, copy=False)

        dzdx, dzdy = self._gradients_central(dem, self.pixel_size_m, pad_mode=pad_mode)
        slope_rise_run = np.sqrt(dzdx * dzdx + dzdy * dzdy, dtype=np.float32)

        if self.slope_units == "degrees":
            slope = np.degrees(np.arctan(slope_rise_run)).astype(np.float32)
            slope_key = "slope_deg"
        else:
            slope = np.arctan(slope_rise_run).astype(np.float32)
            slope_key = "slope_rad"

        if roughness_method == "rms":
            rough = self._roughness_rms(dem, window=roughness_window, pad_mode=pad_mode)
            rough_key = "roughness_rms"
        elif roughness_method == "tri":
            rough = self._roughness_tri(dem, pad_mode=pad_mode)
            rough_key = "roughness_tri"
        else:
            raise ValueError(f"Unknown roughness_method: {roughness_method}")

        return {
            slope_key: slope,
            "slope_rise_run": slope_rise_run.astype(np.float32),
            rough_key: rough.astype(np.float32),
        }

    # -------------------------
    # SLOPE
    # -------------------------
    def _gradients_central(
        self,
        dem: np.ndarray,
        pixel_size_m: float,
        pad_mode: str = "edge",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Central differences with padding so output matches input shape.
        dzdx, dzdy are in (meters / meter) == unitless slope.
        """
        # Pad by 1 so we can do central diff everywhere
        z = np.pad(dem, 1, mode=pad_mode)

        # Central diff: (z[x+1]-z[x-1])/(2*dx)
        dzdx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2.0 * pixel_size_m)
        dzdy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2.0 * pixel_size_m)

        # NaN-safe: if any NaNs exist, keep them local instead of poisoning whole arrays
        if np.isnan(dem).any():
            # If the local stencil includes NaN, set gradient NaN
            stencil_nan_x = np.isnan(z[1:-1, 2:]) | np.isnan(z[1:-1, :-2]) | np.isnan(z[1:-1, 1:-1])
            stencil_nan_y = np.isnan(z[2:, 1:-1]) | np.isnan(z[:-2, 1:-1]) | np.isnan(z[1:-1, 1:-1])
            dzdx = dzdx.astype(np.float32)
            dzdy = dzdy.astype(np.float32)
            dzdx[stencil_nan_x] = np.nan
            dzdy[stencil_nan_y] = np.nan

        return dzdx.astype(np.float32), dzdy.astype(np.float32)

    # -------------------------
    # ROUGHNESS (RMS)
    # -------------------------
    def _roughness_rms(
        self,
        dem: np.ndarray,
        window: int = 5,
        pad_mode: str = "edge",
    ) -> np.ndarray:
        """
        Local RMS roughness approximated as local standard deviation in a WxW window.
        Uses a fast integral-image approach (O(1) per pixel) for mean and mean-square.

        Notes:
        - If NaNs exist, we fall back to a slower but NaN-aware approach.
        """
        if window < 3 or window % 2 == 0:
            raise ValueError("roughness_window must be an odd integer >= 3 (e.g., 3,5,7).")

        if np.isnan(dem).any():
            return self._roughness_rms_nanaware(dem, window=window, pad_mode=pad_mode)

        r = window // 2
        z = np.pad(dem, r, mode=pad_mode).astype(np.float64, copy=False)

        # integral images for z and z^2
        S = z.cumsum(axis=0).cumsum(axis=1)
        S2 = (z * z).cumsum(axis=0).cumsum(axis=1)

        # sum over window via integral image lookups
        H, W = dem.shape
        y0 = np.arange(0, H)
        x0 = np.arange(0, W)
        Y0, X0 = np.meshgrid(y0, x0, indexing="ij")

        y1 = Y0 + window - 1
        x1 = X0 + window - 1
        y0i = Y0
        x0i = X0

        def box_sum(ii: np.ndarray) -> np.ndarray:
            A = ii[y1, x1]
            B = ii[y0i - 1, x1]
            C = ii[y1, x0i - 1]
            D = ii[y0i - 1, x0i - 1]
            return A - B - C + D

        # Shift indices because our Y0/X0 are for the padded grid start
        Y0p = Y0
        X0p = X0
        y1 = Y0p + window - 1
        x1 = X0p + window - 1
        y0i = Y0p
        x0i = X0p

        # Need guard rows/cols: easiest is pad integral with a leading zero row/col
        S0 = np.pad(S, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        S20 = np.pad(S2, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

        # Now indices in S0 are +1
        y0s = y0i + 1
        x0s = x0i + 1
        y1s = y1 + 1
        x1s = x1 + 1

        sum_z = S0[y1s, x1s] - S0[y0s - 1, x1s] - S0[y1s, x0s - 1] + S0[y0s - 1, x0s - 1]
        sum_z2 = S20[y1s, x1s] - S20[y0s - 1, x1s] - S20[y1s, x0s - 1] + S20[y0s - 1, x0s - 1]

        n = float(window * window)
        mean = sum_z / n
        var = (sum_z2 / n) - (mean * mean)
        var = np.maximum(var, 0.0)  # numerical safety
        std = np.sqrt(var)

        return std.astype(np.float32)

    def _roughness_rms_nanaware(
        self,
        dem: np.ndarray,
        window: int,
        pad_mode: str,
    ) -> np.ndarray:
        """
        NaN-aware local stddev using a sliding window.
        Slower than integral-image path, but safe.
        """
        r = window // 2
        z = np.pad(dem, r, mode=pad_mode)
        H, W = dem.shape
        out = np.full((H, W), np.nan, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                block = z[i:i + window, j:j + window]
                vals = block[~np.isnan(block)]
                if vals.size >= (window * window) * 0.5:  # require at least 50% valid
                    out[i, j] = float(np.std(vals, ddof=0))
        return out

    # -------------------------
    # ROUGHNESS (TRI)
    # -------------------------
    def _roughness_tri(self, dem: np.ndarray, pad_mode: str = "edge") -> np.ndarray:
        """
        Terrain Ruggedness Index (TRI):
        mean absolute difference between cell and its 8 neighbours.
        """
        z = np.pad(dem, 1, mode=pad_mode)
        c = z[1:-1, 1:-1]

        neigh = [
            z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:],
            z[1:-1, :-2],               z[1:-1, 2:],
            z[2:, :-2],  z[2:, 1:-1],  z[2:, 2:],
        ]

        diffs = [np.abs(n - c) for n in neigh]
        tri = np.mean(diffs, axis=0).astype(np.float32)

        # preserve NaNs
        if np.isnan(dem).any():
            tri[np.isnan(dem)] = np.nan
        return tri

    # -------------------------
    # VALIDATION
    # -------------------------
    def _validate_dem(self, dem: np.ndarray) -> None:
        if not isinstance(dem, np.ndarray):
            raise TypeError("dem_m must be a numpy array.")
        if dem.ndim != 2:
            raise ValueError(f"dem_m must be 2D (H,W). Got shape {dem.shape}.")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be a positive finite number.")

        if self.expected_shape is not None and dem.shape != self.expected_shape:
            msg = f"Expected shape {self.expected_shape}, got {dem.shape}."
            if self.strict_shape:
                raise ValueError(msg)
            # non-strict: allow it (useful for 512x512 tests, etc.)

# dem_roi_m: (H,W) float32/float64, in meters (e.g., canonical 1024×1024 or smaller test ROIs)
# --- DataHandler stage ---
roi_km = dh.read_roi(tile, 7072, 8096, 7072, 8096)  # (H,W)
terrain = standardise_lola(tile, roi=roi_km, representation="height")

# --- Bridge (THIS LINE WAS MISSING) ---
dem_roi_m = terrain.data_m   # numpy array, meters

# --- FeatureExtractor stage ---
fx = FeatureExtractor(
    pixel_size_m=20.0,
    slope_units="degrees",
    expected_shape=None
)

features = fx.extract(
    dem_roi_m,
    roughness_method="rms",
    roughness_window=5
)

slope_deg = features["slope_deg"]
rough_rms = features["roughness_rms"]

