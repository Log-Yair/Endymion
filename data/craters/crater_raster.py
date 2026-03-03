# ============================================================================
# Endymion Phase-2
# Crater Catalogue -> ROI-Aligned Raster
# ============================================================================
"""
Purpose
-------
Convert a crater catalogue (Robbins CSV: lat, lon, diameter) into a *binary* crater
mask aligned with an Endymion DEM ROI grid.

Why this exists
---------------
Endymion's downstream modules (FeatureExtractor, HazardAssessor, Pathfinder)
operate in raster grid space, so crater information must be converted into the
same pixel grid.

Projection pipeline (handled by Rasterio)
----------------------------------------
WGS84 (EPSG:4326) -> DEM native CRS (polar stereographic) -> (row, col)

Outputs
-------
- crater_mask : np.ndarray uint8 (H, W), values {0,1}
- metadata    : dict with ROI bounds, counts, and parameters (traceability)

Requirements
------------
  pip install rasterio numpy pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------

PixelROI = Tuple[int, int, int, int]
# (row_start, row_end, col_start, col_end) in *full DEM pixel space*


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class CraterRasterConfig:
    """Configuration parameters for rasterising a crater catalogue."""

    # DEM resolution in meters per pixel (must match your ROI DEM resolution)
    pixel_size_m: float = 20.0

    # Size filters expressed in meters (easy to explain + independent of pixel size).
    # Default min is 2 pixels at 20 m/px.
    min_diameter_m: float = 40.0
    max_diameter_m: Optional[float] = None

    # CSV column names (Robbins)
    latitude_column: str = "LAT_CIRC_IMG"
    longitude_column: str = "LON_CIRC_IMG"
    diameter_km_column: str = "DIAM_CIRC_IMG"


# ----------------------------------------------------------------------------
# Geometry helper
# ----------------------------------------------------------------------------

def rasterise_filled_circle(mask: np.ndarray, center_row: int, center_col: int, radius_px: int) -> None:
    """Rasterise a filled circle into a binary mask (in-place).

    Notes
    -----
    - Uses a bounding box so we don't scan the full ROI grid.
    - Distance check enforces a circular footprint.
    """

    H, W = mask.shape

    # Bounding box around the circle (clipped to mask bounds)
    row_min = max(0, center_row - radius_px)
    row_max = min(H, center_row + radius_px + 1)
    col_min = max(0, center_col - radius_px)
    col_max = min(W, center_col + radius_px + 1)

    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            if (r - center_row) ** 2 + (c - center_col) ** 2 <= radius_px ** 2:
                mask[r, c] = 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_crater_mask_from_catalogue(
    *,
    robbins_csv_path: str | Path,
    dem_img_path: str | Path,
    roi_pixels: PixelROI,
    config: CraterRasterConfig = CraterRasterConfig(),
) -> Dict[str, object]:
    """Build an ROI-aligned crater mask from the Robbins crater catalogue.

    Steps
    -----
    1) Load crater catalogue (lat/lon/diameter).
    2) Reproject lat/lon (WGS84) -> DEM CRS.
    3) Convert projected coords -> pixel indices.
    4) Filter to craters whose *centres* fall inside ROI.
    5) Rasterise each crater as a filled circle in ROI-local pixel space.
    """

    robbins_csv_path = Path(robbins_csv_path)
    dem_img_path = Path(dem_img_path)

    # ---- Load catalogue ----
    crater_df = pd.read_csv(robbins_csv_path)

    required_columns = [
        config.latitude_column,
        config.longitude_column,
        config.diameter_km_column,
    ]
    for col in required_columns:
        if col not in crater_df.columns:
            raise ValueError(f"Required column '{col}' not found in crater catalogue CSV.")

    crater_df = crater_df[required_columns].dropna().copy()

    # ---- Open DEM to obtain CRS + pixel transform ----
    with rasterio.open(dem_img_path) as ds:
        if ds.crs is None:
            raise ValueError("DEM image does not have a defined CRS.")

        lats = crater_df[config.latitude_column].to_numpy(dtype=float)
        lons = crater_df[config.longitude_column].to_numpy(dtype=float)

        # Normalise lon into [-180, 180]
        lons = ((lons + 180.0) % 360.0) - 180.0

        # Reproject WGS84 -> DEM CRS
        xs, ys = transform(
            src_crs="EPSG:4326",
            dst_crs=ds.crs,
            xs=lons.tolist(),
            ys=lats.tolist(),
        )

        # Projected (x,y) -> pixel indices (row, col)
        pixel_rows, pixel_cols = ds.index(xs, ys)

    pixel_rows = np.asarray(pixel_rows)
    pixel_cols = np.asarray(pixel_cols)

    # ---- Filter to ROI ----
    row_start, row_end, col_start, col_end = roi_pixels

    inside_roi = (
        (pixel_rows >= row_start)
        & (pixel_rows < row_end)
        & (pixel_cols >= col_start)
        & (pixel_cols < col_end)
    )

    crater_df_roi = crater_df.loc[inside_roi].copy()
    pixel_rows_roi = pixel_rows[inside_roi]
    pixel_cols_roi = pixel_cols[inside_roi]

    # ---- Size filters (meters), then convert to pixels ----
    diameter_km = crater_df_roi[config.diameter_km_column].to_numpy(dtype=float)
    diameter_m = diameter_km * 1000.0

    valid = diameter_m >= float(config.min_diameter_m)
    if config.max_diameter_m is not None:
        valid &= diameter_m <= float(config.max_diameter_m)

    diameter_m = diameter_m[valid]
    pixel_rows_roi = pixel_rows_roi[valid]
    pixel_cols_roi = pixel_cols_roi[valid]

    diameter_px = diameter_m / float(config.pixel_size_m)

    # ---- Build ROI-local mask ----
    roi_height = row_end - row_start
    roi_width = col_end - col_start

    crater_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)

    crater_count = 0

    for full_row, full_col, d_px in zip(pixel_rows_roi, pixel_cols_roi, diameter_px):
        local_row = int(full_row - row_start)
        local_col = int(full_col - col_start)
        radius_px = int(max(1, round(float(d_px) / 2.0)))

        if 0 <= local_row < roi_height and 0 <= local_col < roi_width:
            rasterise_filled_circle(crater_mask, local_row, local_col, radius_px)
            crater_count += 1

    metadata = {
        "roi_pixels": {
            "row_start": int(row_start),
            "row_end": int(row_end),
            "col_start": int(col_start),
            "col_end": int(col_end),
        },
        "roi_shape": {"height": int(roi_height), "width": int(roi_width)},
        "pixel_size_m": float(config.pixel_size_m),
        "catalogue_total_rows": int(len(crater_df)),
        "craters_centres_in_roi": int(inside_roi.sum()),
        "craters_rasterised": int(crater_count),
        "config": {
            "min_diameter_m": float(config.min_diameter_m),
            "max_diameter_m": None if config.max_diameter_m is None else float(config.max_diameter_m),
            "latitude_column": config.latitude_column,
            "longitude_column": config.longitude_column,
            "diameter_km_column": config.diameter_km_column,
        },
    }

    return {"crater_mask": crater_mask, "metadata": metadata}