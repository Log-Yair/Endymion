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

ROI = Tuple[int, int, int, int]  # (r0,r1,c0,c1 = (row_start, row_end, col_start, col_end))

@dataclass
class CraterRaster:
    pixel_size_m: float = 20.0 # meters per pixel
    min_diameter: float = 2.0 # minimum crater diameter in meters
    max_diameter: Optional[float] = None # maximum crater diameter in meters (None = no limit)
    col_lat : str= "LAT_CIRC_IMG" # column name for latitude in input CSV
    col_lon : str= "LON_CIRC_IMG" # column name for longitude in input CSV
    col_diam_km : str= "DIAM_CIRC_IMG" # column name for diameter in input CSV

def raster_circle(mask :np.ndarray, r0 :int, c0 :int, radius_px :int) -> None:
    """Rasterise a circle into the mask at (r0,c0) with given radius in pixels."""
    rows, cols = mask.shape
    r_min = max(0, r0 - radius_px)
    r_max = min(rows, r0 + radius_px + 1)
    c_min = max(0, c0 - radius_px)
    c_max = min(cols, c0 + radius_px + 1)

    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            if (r - r0) ** 2 + (c - c0) ** 2 <= radius_px ** 2:
                mask[r, c] = 1