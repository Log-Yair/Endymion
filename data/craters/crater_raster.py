# ============================================================================
# Crater Catalogue → ROI-Aligned Raster
# ============================================================================
#
# Purpose:
#   Convert Robbins crater catalogue (lat, lon, diameter)
#   into a binary crater mask aligned with the DEM ROI grid.
#
# Design Principles:
#   - ROI is defined in pixel space (row_start, row_end, col_start, col_end).
#   - All outputs must match DEM grid dimensions exactly.
#   - Projection handled via Rasterio (no manual stereographic math).
#
# Projection Pipeline:
#   WGS84 (EPSG:4326) → DEM native CRS (polar stereographic)
#   → Projected coordinates (x,y)
#   → Pixel indices via dataset.index(x, y)
#
# Output:
#   crater_mask.npy  (uint8, 0/1)
#   crater_meta.json
#
# ============================================================================


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

#============================================================================
# Type DEFINITION for Region of Interest (ROI) in pixel coordinates
#============================================================================

ROI = Tuple[int, int, int, int]  # (r0,r1,c0,c1 = (row_start, row_end, col_start, col_end))


#===========================================================================
# CraterRaster configuration dataclass
#===========================================================================

@dataclass
class CraterRasterConfig:
    ''' Configuration parameters for rasterising crater catalogue '''

    pixel_size_m: float = 20.0 # DEM resolution on meters per pixel
    
    ''' Crater size limits (in meters) for rasterisation '''

    min_diameter: float = 2.0 # minimum crater diameter in meters

    max_diameter: Optional[float] = None # maximum crater diameter in meters (None = no limit)

    ''' Column names in the input CSV for crater properties '''

    latitude_column: str = "LAT_CIRC_IMG" # column name for crater latitude in the input CSV

    longitude_column: str = "LON_CIRC_IMG" # column name for crater longitude in the input CSV

    diameter_km_column: str = "DIAM_CIRC_IMG" # column name for crater diameter in kilometers in the input CSV



# ----------------------------------------------------------------------------
# Core Geometry Utility
# ----------------------------------------------------------------------------

def rasterise_filled_circle(
    mask: np.ndarray,
    center_row: int,
    center_col: int,
    radius_px: int
) -> None:
    """
    Rasterises a filled circle into the mask array.

    Parameters
    ----------
    mask : np.ndarray
        Binary output array (modified in-place).

    center_row, center_col : int
        Pixel coordinates within ROI-local grid.

    radius_px : int
        Radius in pixels.

    Notes
    -----
    - Uses bounding-box restriction to avoid scanning entire grid.
    - Distance check enforces circular shape.
    """

    grid_height, grid_width = mask.shape # dimensions of the ROI grid

    row_min = max(0, center_row - radius_px) # restrict to bounding box around the circle

    row_max = min(grid_height, center_row + radius_px + 1) # +1 because upper bound is exclusive in Python slicing

    col_min = max(0, center_col - radius_px) # restrict to bounding box around the circle

    col_max = min(grid_width, center_col + radius_px + 1) # +1 because upper bound is exclusive in Python slicing

    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            if (r - center_row) ** 2 + (c - center_col) ** 2 <= radius_px ** 2:
                mask[r, c] = 1

# ============================================================================
# Main function to rasterise crater catalogue into binary mask
# ============================================================================

def build_crater_mask_from_catalogue(
    *,
    robbins_csv_path: str | Path,
    dem_img_path: str | Path,
    roi_pixels: ROI,
    config: CraterRasterConfig = CraterRasterConfig(),
) -> Dict[str, object]:
    """
    Converts Robbins crater catalogue to ROI-aligned crater mask.

    Steps:
    1. Load crater catalogue.
    2. Reproject lat/lon to DEM CRS.
    3. Convert projected coords → pixel indices.
    4. Filter to ROI bounds.
    5. Rasterise crater circles into ROI-local mask.
    """

    # ------------------------------------------------------------------------
    # Load crater catalogue
    # ------------------------------------------------------------------------

    crater_df = pd.read_csv(robbins_csv_path)
