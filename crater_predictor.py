# References / og notes:
# - Phase-1 version was a stub that returned all zeros.
# - Phase-2 adds an *optional* catalogue-backed mode that rasterises Robbins craters into an ROI-aligned mask using Rasterio.
# - Coordinate transform: WGS84 (EPSG:4326) -> DEM native CRS; pixel index via ds.index(x, y).

"""
CraterPredictor for Endymion.

What this module does (now):

- Keeps the pipeline modular (CraterPredictor exists even before ML).
- Supports TWO modes:

  1) "stub_v1" (default): returns all zeros (Phase-1 compatible)
  2) "catalogue_raster_v1": builds/loads a crater *binary mask* from the Robbins catalogue
     and returns it as a "crater_proba" map (0/1).

Why this is useful: 
It lets you integrate craters into the pipeline *today* (Phase-2 Step 1), without
changing HazardAssessor yet.

Later (Phase-2 ML):
Replace catalogue_raster_v1 with a real model inference that outputs probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np

@dataclass
class CraterPredictor:
    """
    Crater predictor.

    Parameters
    ----------
    model_id:
        - "stub_v1" (default): returns all zeros.
        - "catalogue_raster_v1": rasterises Robbins catalogue into ROI mask.

    cache_product_name:
        Filename base used when saving into the derived directory.
    """
    model_id: str = "stub_v1" # default to stub mode for Phase-1 compatibility
    cache_product_name: str = "crater_mask"  # crater_mask.npy + crater_meta.json

    def predict_proba_map(
        self,
        features: Dict[str, np.ndarray],
        dem_m: Optional[np.ndarray] = None,
        *,
        tile_id: Optional[str] = None, # for traceability in metadata
        roi: Optional[tuple[int, int, int, int]] = None, # (row_start, row_end, col_start, col_end) in pixel space
        dh: Any = None,  # DataHandler (kept as Any to avoid hard import cycles)
        dem_img_path: Optional[str | Path] = None, # for coordinate transforms in catalogue_raster mode
        robbins_csv_path: Optional[str | Path] = None, # path to Robbins crater catalogue CSV for catalogue_raster mode
        raster_config: Optional[Any] = None,  # crater_raster.CraterRasterConfig
        rebuild_if_missing: bool = False, # if True, will build the crater mask from catalogue if not found in cache (only for catalogue_raster mode)
    ) -> Dict[str, np.ndarray]:
        """
        Outputs is alawys "crater_proba" for pipeline consistency, even if it's binary in catalogue_raster mode.
        """
        # pick any raster as shape reference
        ref = self.get_reference_raster(features) # helper function to find a 2D raster in features to use as shape reference for the output probability map
        
        for v in features.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                ref = v
                break
        if ref is None:
            raise ValueError("No 2D feature rasters provided to CraterPredictor.")

        proba = np.zeros(ref.shape, dtype=np.float32)

        meta = {
            "model_id": self.model_id,
            "note": "Phase-1 stub: returns all zeros",
        }

        return {"crater_proba": proba, "meta": meta}
