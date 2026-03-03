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
from importlib import metadata
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
    
    ''' Get grid dimensions for bounding box restriction '''
    grid_height, grid_width = mask.shape # dimensions of the ROI grid

    ''' Calculate bounding box limits, ensuring they stay within grid bounds '''

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

    required_columns = [
        config.latitude_column,
        config.longitude_column,
        config.diameter_km_column
    ]

    for col in required_columns:
        if col not in crater_df.columns:
            raise ValueError(f"Required column '{col}' not found in crater catalogue CSV.")
    
    crater_df = crater_df[required_columns].dropna.copy() # keep only required columns and drop rows with missing values

    # ------------------------------------------------------------------------
    # Open DEM with rasterio to get CRS and transform
    # ------------------------------------------------------------------------

    with rasterio.open(dem_img_path) as dataset:
        if dataset.crs is None:
            raise ValueError("DEM image does not have a defined CRS.")
        
        # extract geographic coordinates
        latituded = crater_df[config.latitude_column].to_numpy(dtype=float) # extract latitude column as numpy array of floats
        longituded = crater_df[config.longitude_column].to_numpy(dtype=float) # extract longitude column as numpy array of floats

        # normlasisee crater longitud into [-180, 180] range to avoid issues with dateline crossing
        longitudes = ((longituded + 180) % 360) - 180

        # reproject crater coordinates from WGS84 to DEM CRS
        projected_x, projected_y = transform(
            src_crs='EPSG:4326', # WGS84
            dst_crs=dataset.crs, # DEM native CRS
            xs=longitudes,
            ys=latituded
        )

        # convert projected coordinates to pixel indices
        pixel_rows, pixel_cols = dataset.index(projected_x, projected_y) # get pixel indices for each crater center

    pixel_cols = np.array(pixel_cols) # convert to numpy array for easier indexing
    pixel_rows = np.array(pixel_rows) # convert to numpy array for easier indexing

    # ------------------------------------------------------------------------
    # Filter craters to those whose centers fall within the ROI bounds
    # ------------------------------------------------------------------------
    row_start, row_end, col_start, col_end = roi_pixels # unpack ROI pixel bounds

    # Create boolean mask for craters whose centers are
    inside_roi = (
        (pixel_rows >= row_start) &
        (pixel_rows < row_end) &
        (pixel_cols >= col_start) &
        (pixel_cols < col_end)
    )

    crater_df_roi = crater_df.loc[inside_roi].copy() # filter crater dataframe to only include craters whose centers are inside the ROI

    pixel_rows_roi = pixel_rows[inside_roi] # filter pixel row indices to only include those whose centers are inside the ROI
    pixel_cols_roi = pixel_cols[inside_roi] # filter pixel column indices to only include those whose centers are inside the ROI

    # ------------------------------------------------------------------------
    # Convert crater diameter (km) → pixel radius
    # ------------------------------------------------------------------------

    diameter_km = crater_df_roi[config.diameter_km_column].to_numpy(dtype=float) # extract diameter column as numpy array of floats

    diameter_px = (diameter_km * 1000.0) / config.pixel_resolution_m # convert diameter from kilometers to meters, then divide by pixel size to get diameter in pixels

    # Apply size filters
    valid = diameter_px >= config.minimum_diameter_px # filter out craters smaller than minimum diameter
    if config.maximum_diameter_px is not None:
        valid &= diameter_px <= config.maximum_diameter_px      # filter out craters larger than maximum diameter (if specified)

    diameter_px = diameter_px[valid] # filter diameter array to only include valid craters
    pixel_rows_roi = pixel_rows_roi[valid] # filter pixel row indices to only include valid craters
    pixel_cols_roi = pixel_cols_roi[valid] # filter pixel column indices to only include valid craters

    # ------------------------------------------------------------------------
    # create a ROI locla mask
    # ------------------------------------------------------------------------

    # dimensions of the ROI grid
    roi_height = row_end - row_start # number of rows in the ROI grid
    roi_width = col_end - col_start # number of columns in the ROI grid 

    crater_mask = np.zeros((roi_height, roi_width), dtype=np.uint8) # initialize binary mask for the ROI grid
    crater_count = 0 # initialize crater count

    for full_row, full_col, diameter in zip(pixel_rows_roi, pixel_cols_roi, diameter_px):
        local_row = full_row - row_start # convert to ROI-local row index
        local_col = full_col - col_start # convert to ROI-local column index
        radius_px = int(max(1, round(diameter / 2))) # calculate radius in pixels, ensuring at least 1 pixel radius for very small craters

        if 0 <= local_row < roi_height and 0 <= local_col < roi_width:
            rasterise_filled_circle(crater_mask, local_row, local_col, radius_px) # rasterise crater circle into the mask
            crater_count += 1 # increment crater count


    # ------------------------------------------------------------------------
    # getting the metadata for the crater mask
    # ------------------------------------------------------------------------  

    metadata = {
    "roi_pixel_bounds": {
        # Note: these are the pixel bounds of the ROI in the full DEM grid, not the local ROI grid
        "row_start": row_start,

        "row_end": row_end,

        "col_start": col_start,

        "col_end": col_end
    },

    "roi_shape": {
        # These are the dimensions of the ROI grid (height and width in pixels)

        "height": roi_height,

        "width": roi_width
    },
    # The pixel resolution of the DEM, which is used to convert crater diameters from meters to pixels for rasterisation
    "pixel_size_m": config.pixel_resolution_m,

    # The number of craters that were rasterised into the mask (i.e. the number of craters whose centers fell within the ROI and passed the size filters)
    "crater_count": crater_count,

    # The total number of craters in the original catalogue (before filtering to ROI and size limits)
    "catalogue_total_rows": int(len(crater_df)),

    # The number of craters that had their centers within the ROI bounds (before applying size filters)
    "craters_in_roi": int(len(crater_df_roi)),

    # The number of craters that were actually rasterised into the mask (after applying size filters)
    "craters_rasterised": crater_count,
    
    # The configuration parameters used for rasterisation, which is important for reproducibility and understanding the context of the crater mask (e.g. what size limits were applied, which columns were used for lat/lon/diameter)
    "config": { 
        "min_diameter_m": config.minimum_diameter_m,

        "max_diameter_m": config.maximum_diameter_m,

        "latitude_column": config.latitude_column,

        "longitude_column": config.longitude_column,

        "diameter_km_column": config.diameter_km_column
    },

    }
