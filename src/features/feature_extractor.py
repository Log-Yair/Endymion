# -*- coding: utf-8 -*-
"""
feature_extractor for Endymion
"""

from __future__ import annotations

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
    """
    pixel_size_m: float = 20.0
    slope_units: str = "degrees"  # "degrees" or "radians"
    expected_shape: Optional[Tuple[int, int]] = (1024, 1024)  # set None to disable
    strict_shape: bool = False

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
        slope_rise_run = np.sqrt(dzdx * dzdx + dzdy * dzdy).astype(np.float32)

        if self.slope_units == "degrees":
            slope = np.degrees(np.arctan(slope_rise_run)).astype(np.float32)
            slope_key = "slope_deg"
        else:
            slope = np.arctan(slope_rise_run).astype(np.float32)
            slope_key = "slope_rad"

        if roughness_method == "rms":
            rough = self._roughness_rms(dem, window=roughness_window, pad_mode=pad_mode).astype(np.float32)
            rough_key = "roughness_rms"
        elif roughness_method == "tri":
            rough = self._roughness_tri(dem, pad_mode=pad_mode).astype(np.float32)
            rough_key = "roughness_tri"
        else:
            raise ValueError(f"Unknown roughness_method: {roughness_method}")

        return {
            slope_key: slope,
            "slope_rise_run": slope_rise_run,
            rough_key: rough,
        }

    def extract_from_dem(self, dem_m: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        return self.extract(dem_m, **kwargs)

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
        z = np.pad(dem, 1, mode=pad_mode)

        dzdx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2.0 * pixel_size_m)
        dzdy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2.0 * pixel_size_m)

        if np.isnan(dem).any():
            stencil_nan_x = np.isnan(z[1:-1, 2:]) | np.isnan(z[1:-1, :-2]) | np.isnan(z[1:-1, 1:-1])
            stencil_nan_y = np.isnan(z[2:, 1:-1]) | np.isnan(z[:-2, 1:-1]) | np.isnan(z[1:-1, 1:-1])
            dzdx = dzdx.astype(np.float32, copy=False)
            dzdy = dzdy.astype(np.float32, copy=False)
            dzdx[stencil_nan_x] = np.nan
            dzdy[stencil_nan_y] = np.nan

        return dzdx.astype(np.float32, copy=False), dzdy.astype(np.float32, copy=False)

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

        Fast path (no NaNs): integral image (O(1) per pixel).
        NaN path: sliding-window fallback (slower but safe).
        """
        if window < 3 or window % 2 == 0:
            raise ValueError("roughness_window must be an odd integer >= 3 (e.g., 3,5,7).")

        if np.isnan(dem).any():
            return self._roughness_rms_nanaware(dem, window=window, pad_mode=pad_mode)

        r = window // 2
        z = np.pad(dem, r, mode=pad_mode).astype(np.float64, copy=False)

        # integral images with leading 0 row/col
        S = np.pad(z.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        S2 = np.pad((z * z).cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

        H, W = dem.shape
        y0 = np.arange(0, H)
        x0 = np.arange(0, W)
        Y0, X0 = np.meshgrid(y0, x0, indexing="ij")

        # in S/S2 coordinates (already +1 padded), use half-open [y0,y1) form
        y0s = Y0
        x0s = X0
        y1s = Y0 + window
        x1s = X0 + window

        sum_z = S[y1s, x1s] - S[y0s, x1s] - S[y1s, x0s] + S[y0s, x0s]
        sum_z2 = S2[y1s, x1s] - S2[y0s, x1s] - S2[y1s, x0s] + S2[y0s, x0s]

        n = float(window * window)
        mean = sum_z / n
        var = (sum_z2 / n) - (mean * mean)
        var = np.maximum(var, 0.0)

        return np.sqrt(var).astype(np.float32)

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

        min_required = int((window * window) * 0.5)

        for i in range(H):
            for j in range(W):
                block = z[i:i + window, j:j + window]
                vals = block[~np.isnan(block)]
                if vals.size >= min_required:
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
            # non-strict: allow it
