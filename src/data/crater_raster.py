# ============================================================================
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
  scipy is also required for distance transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import inv
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import rasterio # for reading DEM and coordinate reprojection
from scipy.ndimage import distance_transform_edt # for computing distance to nearest crater cell
from pyproj import CRS, Transformer # for coordinate reprojection from lunar geographic to DEM CRS

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
    """
    Configuration for rasterising crater catalogue data into ROI-aligned products.
    """
    pixel_size_m: float = 20.0

    # Size filters in meters
    min_diameter_m: float = 40.0
    max_diameter_m: Optional[float] = None

    # Density window (odd integer, in pixels)
    density_window_px: int = 21

    # Optional clip for distance map (helps avoid huge values dominating stats)
    max_distance_m: Optional[float] = 1000.0

    # CSV column names (Robbins)
    latitude_column: str = "LAT_CIRC_IMG"
    longitude_column: str = "LON_CIRC_IMG"
    diameter_km_column: str = "DIAM_CIRC_IMG"

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def rasterise_filled_circle(mask: np.ndarray, center_row: int, center_col: int, radius_px: int) -> None:
    """
    Rasterise a filled circle into a binary mask (in-place).

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


# 
def _validate_density_window_size(window: int) -> None:
    if window < 3 or window % 2 == 0:
        raise ValueError(f"Density window size must be an odd integer >= 3")
    return int(window)

# 
def _compute_local_density(binary_mask: np.ndarray, window: int) -> np.ndarray:
    """
    Local crater density as moving-window mean of the binary crater mask.
    Output range is [0, 1].
    """
    window = _validate_density_window_size(window)

    mask = binary_mask.astype(np.float32, copy=False) # ensure float32 for cumulative sums, but avoid unnecessary copy if already float32
    r = window // 2

    padded = np.pad(mask, r, mode="constant", constant_values=0.0) # pad with zeros so that the moving window can be applied at the borders without issues

    # Integral image with leading zero row/col
    S = np.pad(
        padded.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0.0,
    )

    H,W = mask.shape # original mask shape (not padded)
    y0 = np.arange(H) # row indices of top-left corner of window
    x0 = np.arange(W) # col indices of top-left corner of window
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij") # shape (H, W), each element is the row/col index of the top-left corner of the window for that pixel

    y0s = Y0 # row index of top-left corner of window
    x0s = X0 # col index of top-left corner of window
    y1s = Y0 + window # row index of bottom-right corner of window
    x1s = X0 + window # col index of bottom-right corner of window

    sum_values = S[y1s, x1s] - S[y0s, x1s] - S[y1s, x0s] + S[y0s, x0s] # shape (H, W), sum of values in the window for each pixel
    density = sum_values / float(window * window) # local density is mean value in the window, range [0, 1]

    return density.astype(np.float32)


#
def _compute_distance_to_crater(binary_mask: np.ndarray, pixel_size_m: float, max_distance_m: Optional[float]) -> np.ndarray:
    """
    Distance to nearest crater cell in meters.

    Convention:
    - crater cells have distance 0
    - non-crater cells have Euclidean distance to nearest crater cell
    """
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be 2D.")
    
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be 2D.")
    
    # Compute distance transform (distance in pixels to nearest crater cell)
    distance_px = distance_transform_edt(binary_mask == 0) # distance transform of the inverse mask gives distance to nearest crater cell
    distance_m = distance_px.astype(np.float32) * float(pixel_size_m) # convert distance from pixels to meters

    # Optionally clip distance to max_distance_m to avoid huge values dominating stats
    if max_distance_m is not None:
        distance_m = np.minimum(distance_m, np.float32(max_distance_m)) # clip distance to max_distance_m if specified

    return distance_m



# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_crater_products_from_catalogue(
    *,
    robbins_csv_path: str | Path,
    dem_img_path: str | Path,
    roi_pixels: PixelROI,
    config: CraterRasterConfig = CraterRasterConfig(),
) -> Dict[str, object]:
    """
    Build ROI-aligned crater raster products from the Robbins crater catalogue.

    Outputs:
    - crater_mask        : uint8  (0/1)
    - crater_distance_m  : float32
    - crater_density     : float32 in [0,1]
    - metadata           : dict
    """
    # Validate inputs
    robbins_csv_path = Path(robbins_csv_path) # ensure Path object for consistency
    dem_img_path = Path(dem_img_path)

    # Check input files exist before doing any processing

    if not robbins_csv_path.exists():
        raise FileNotFoundError(f"Robbins crater catalogue CSV not found: {robbins_csv_path}")
    
    if not dem_img_path.exists():
        raise FileNotFoundError(f"DEM image not found: {dem_img_path}")
    

    #  Load catalogue 
    # Notes:
    # - Current local Robbins file has no header row.
    # - First line appears malformed/incomplete (9 fields), while valid records have 21 fields.
    # - So we skip the first line and assign the expected Robbins column names manually.

    robbins_columns = [
        "CRATER_ID",
        "LAT_CIRC_IMG",
        "LON_CIRC_IMG",
        "LAT_ELLI_IMG",
        "LON_ELLI_IMG",
        "DIAM_CIRC_IMG",
        "DIAM_CIRC_SD_IMG",
        "DIAM_ELLI_MAJOR_IMG",
        "DIAM_ELLI_MINOR_IMG",
        "DIAM_ELLI_ECCEN_IMG",
        "DIAM_ELLI_ELLIP_IMG",
        "DIAM_ELLI_ANGLE_IMG",
        "LAT_ELLI_SD_IMG",
        "LON_ELLI_SD_IMG",
        "DIAM_ELLI_MAJOR_SD_IMG",
        "DIAM_ELLI_MINOR_SD_IMG",
        "DIAM_ELLI_ANGLE_SD_IMG",
        "DIAM_ELLI_ECCEN_SD_IMG",
        "DIAM_ELLI_ELLIP_SD_IMG",
        "ARC_IMG",
        "PTS_RIM_IMG",
    ]

    crater_df = pd.read_csv(
        robbins_csv_path,
        header=None,
        names=robbins_columns,
        skiprows=1,
        encoding="utf-8-sig",
    )

    crater_df.columns = crater_df.columns.str.strip()
    
    # Check required columns exist
    required_columns = [
        config.latitude_column,
        config.longitude_column,
        config.diameter_km_column,
    ]
    # Validate that required columns are present in the CSV
    for col in required_columns:
        if col not in crater_df.columns:
            raise ValueError(f"Required column '{col}' not found in crater catalogue CSV.")

    crater_df = crater_df[required_columns].dropna().copy() # keep only needed columns and drop rows with missing values

    #  Open DEM to obtain CRS + pixel transform 
    with rasterio.open(dem_img_path) as ds:
        if ds.crs is None:
            raise ValueError("DEM image does not have a defined CRS.")

        lats = crater_df[config.latitude_column].to_numpy(dtype=float) # convert to numpy array of floats
        lons = crater_df[config.longitude_column].to_numpy(dtype=float) # convert to numpy array of floats

        # Normalise lon into [-180, 180]
        lons = ((lons + 180.0) % 360.0) - 180.0

        # Reproject lunar geographic coordinates -> DEM CRS
        #
        # Important:
        # - Robbins catalogue coordinates are lunar lat/lon, not Earth WGS84.
        # - Therefore EPSG:4326 is wrong here.
        # - derive the matching lunar geographic CRS from the DEM's base CRS.

        dem_crs = CRS.from_user_input(ds.crs)
        lunar_geo_crs = dem_crs.geodetic_crs

        if lunar_geo_crs is None:
            raise ValueError("Could not derive lunar geographic CRS from DEM CRS.")

        transformer = Transformer.from_crs(
            lunar_geo_crs,
            dem_crs,
            always_xy=True,
        )

        xs, ys = transformer.transform(lons, lats) # reprojected coordinates in DEM CRS (x, y)

        xs =np.asarray(xs, dtype=np.float64) # ensure numpy array of floats
        ys = np.asarray(ys, dtype=np.float64) # ensure numpy array of floats

        inv_transform = ~ds.transform # inverse of the DEM's pixel transform for converting from DEM CRS coordinates to pixel space (projected x,y  ->  pixel col,row)
        pixel_cols, pixel_rows = inv_transform * (xs, ys) # convert from DEM CRS coordinates to pixel space (col, row)

        pixel_rows = np.floor(pixel_rows).astype(int) # convert to numpy array of ints
        pixel_cols = np.floor(pixel_cols).astype(int) # convert to numpy array of ints

    # ---- Filter to ROI ----
    row_start, row_end, col_start, col_end = roi_pixels # unpack ROI bounds in full DEM pixel space

    # Check which crater centres fall inside the ROI bounds (in pixel space)
    inside_roi = (
        (pixel_rows >= row_start)
        & (pixel_rows < row_end)
        & (pixel_cols >= col_start)
        & (pixel_cols < col_end)
    )

    crater_df_roi = crater_df.loc[inside_roi].copy() # keep only craters with centres inside ROI
    pixel_rows_roi = pixel_rows[inside_roi] # pixel rows of craters with centres inside ROI
    pixel_cols_roi = pixel_cols[inside_roi] # pixel cols of craters with centres inside ROI

    # ---- Size filters (meters), then convert to pixels ----
    diameter_km = crater_df_roi[config.diameter_km_column].to_numpy(dtype=float)
    diameter_m = diameter_km * 1000.0 # convert km to m

    # Apply size filters in meters (independent of pixel size)
    valid = diameter_m >= float(config.min_diameter_m)
    if config.max_diameter_m is not None:
        valid &= diameter_m <= float(config.max_diameter_m)

    # Keep only valid craters after size filtering
    diameter_m = diameter_m[valid]
    pixel_rows_roi = pixel_rows_roi[valid]
    pixel_cols_roi = pixel_cols_roi[valid]

    diameter_px = diameter_m / float(config.pixel_size_m) # convert diameter from meters to pixels using config pixel size

    # ---- Build ROI-local mask ----
    roi_height = row_end - row_start
    roi_width = col_end - col_start

    crater_mask = np.zeros((roi_height, roi_width), dtype=np.uint8) # binary mask for craters in ROI (0=no crater, 1=crater)

    crater_count = 0 # count of craters rasterised into the mask

    # Rasterise each crater as a filled circle in ROI-local pixel space
    
    for full_row, full_col, d_px in zip(pixel_rows_roi, pixel_cols_roi, diameter_px):

        # Convert from full DEM pixel space to ROI-local pixel space by subtracting the ROI start indices
        local_row = int(full_row - row_start)
        local_col = int(full_col - col_start)
        radius_px = int(max(1, round(float(d_px) / 2.0))) # radius in pixels, at least 1 pixel to ensure visibility of small craters

        # Check if the crater centre is within the ROI bounds in local pixel space before rasterising
        if 0 <= local_row < roi_height and 0 <= local_col < roi_width:
            rasterise_filled_circle(crater_mask, local_row, local_col, radius_px)
            crater_count += 1

    # Compute distance map to nearest crater cell in meters
    crater_distance_m = _compute_distance_to_crater(
        crater_mask, 
        pixel_size_m=config.pixel_size_m,
        max_distance_m=config.max_distance_m,
    )
    # Compute local crater density as moving-window mean of the binary crater mask
    crater_density = _compute_local_density(
        crater_mask,
        window=config.density_window_px,
    )                                                

    metadata = {
        
        "roi_pixels": {
            # Store ROI bounds in full DEM pixel space for traceability
            "row_start": int(row_start),
            "row_end": int(row_end),
            "col_start": int(col_start),
            "col_end": int(col_end),
        },

        "roi_shape": {"height": int(roi_height), "width": int(roi_width)}, # Store ROI shape for traceability
        "pixel_size_m": float(config.pixel_size_m), # Store pixel size for traceability
        "catalogue_total_rows": int(len(crater_df)), # Total rows in original catalogue (before filtering)
        "craters_centres_in_roi": int(inside_roi.sum()), # Number of craters with centres inside ROI (before size filtering)
        "craters_rasterised": int(crater_count), # Number of craters actually rasterised into the mask (after all filtering)
        "products":[
            "crater_mask",
            "crater_distance_m",
            "crater_density",
        ],

        # Statistics about the crater distribution in the ROI for traceability and potential filtering in downstream modules
        "stats": {
            "mask_coverage_ratio": float(crater_mask.mean()),
            "distance_min_m": float(np.min(crater_distance_m)),
            "distance_max_m": float(np.max(crater_distance_m)),
            "distance_mean_m": float(np.mean(crater_distance_m)),
            "density_min": float(np.min(crater_density)),
            "density_max": float(np.max(crater_density)),
            "density_mean": float(np.mean(crater_density)),
        },
        
        # Store config parameters for traceability
        "config": {
            "min_diameter_m": float(config.min_diameter_m),
            "max_diameter_m": None if config.max_diameter_m is None else float(config.max_diameter_m),
            "density_window_px": int(config.density_window_px),
            "max_distance_m": None if config.max_distance_m is None else float(config.max_distance_m),
            "latitude_column": config.latitude_column,
            "longitude_column": config.longitude_column,
            "diameter_km_column": config.diameter_km_column,
        },

    }

    
    return {
        "crater_mask": crater_mask,
        "crater_distance_m": crater_distance_m,
        "crater_density": crater_density, 
        "metadata": metadata,
    }

"""backwards compatibility wrapper for CraterPredictor"""
def build_crater_mask_from_catalogue(
    *,
    robbins_csv_path: str | Path,
    dem_img_path: str | Path,
    roi_pixels: PixelROI,
    config: CraterRasterConfig = CraterRasterConfig(),

) -> Dict[str, Any]:
    
    out = build_crater_products_from_catalogue(
        robbins_csv_path=robbins_csv_path,
        dem_img_path=dem_img_path,
        roi_pixels=roi_pixels,
        config=config,
    )
    return {
        "crater_proba": out["crater_mask"], # for pipeline consistency, even if it's binary in this mode
        "meta": out["metadata"],
    }
