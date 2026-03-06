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
It lets me integrate craters into the pipeline *today* without
changing HazardAssessor yet.

Later (Phase-2 ML):
Replace catalogue_raster_v1 with a real model inference that outputs probabilities.
"""

from __future__ import annotations

from curses import meta
from dataclasses import dataclass
from logging import config
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
        ref = self._get_reference_raster(features) # helper function to find a 2D raster in features to use as shape reference for the output probability map

        if self.model_id == "stub_v1":

            proba = np.zeros(ref.shape, dtype=np.float32) # return all zeros for stub mode
            return {"crater_proba": proba,
                    "meta":{"model_id": self.model_id, "note": "Phase-1 stub: returns all zeros"}} # include model_id in metadata for traceability
        if self.model_id != "catalogue_raster_v1":
            raise ValueError(f"Unknown model_id: {self.model_id}")
        
        # ----------------------------------------------------------------------
        # Catalogue raster mode:
        # ----------------------------------------------------------------------

        if dh is None or tile_id is None or roi is None:
            raise ValueError("catalogue_raster_v1 mode requires dh, tile_id, and roi for caching")
        
        # cache handling: check if we already have a rasterised crater mask for this tile and ROI in the derived directory

        derived_dir = Path(dh.derived_dir(tile_id, roi)) # get the derived directory for this tile and ROI
        derived_dir.mkdir(parents=True, exist_ok=True) # ensure it exists

        # Check if cached mask exists

        mask_path = derived_dir / f"{self.cache_product_name}.npy"      # e.g. crater_mask.npy
        meta_path = derived_dir / f"{self.cache_product_name}_meta.json" # e.g. crater_mask_meta.json
        
        # If the cache exists load it

        if mask_path.exists() and meta_path.exists() and not rebuild_if_missing:
            crater_mask = np.load(mask_path)
            if crater_mask.shape != ref.shape:
                raise ValueError(
                    f"Cached crater mask shape {crater_mask.shape} does not match expected {ref.shape}. "
                    f"(tile_id={tile_id}, roi={roi})"
                )
            
            meta = json.loads(meta_path.read_text())

            return {
                "crater_proba": crater_mask.astype(np.float32), # return as float32 for pipeline consistency, even though it's binary
                "meta": {
                    "model_id": self.model_id,
                    "source": "derived_cache",
                    "product": str(mask_path.name),
                    "catalogue": meta,  
                },
            }
        # If not, build it from the catalogue
        if dem_img_path is None or robbins_csv_path is None:
            raise ValueError(
                "Crater mask not found in derived cache. Provide dem_img_path and robbins_csv_path to build it "
                "(or set rebuild_if_missing=False and precompute it)."
            )
        
        # import lazily so that i don't have to install rasterio for Phase-1
        from crater_raster import build_crater_mask_from_catalogue, CraterRasterConfig

        cfg = raster_config if raster_config is not None else CraterRasterConfig() # use provided config or default config

        # build the crater mask from the catalogue (this will also handle coordinate transforms and rasterisation)
        out = build_crater_mask_from_catalogue(
            robbins_csv_paths=robbins_csv_path,
            dem_img_path=dem_img_path,
            roi=roi,
            config=cfg,
        )

    # Extract outputs
        crater_mask = out["crater_mask"] # binary mask of shape (rows, cols) where 1 indicates crater presence according to the catalogue
        crater_meta = out["metadata"] # e.g. number of craters rasterised, any warnings about craters outside the ROI, etc.

        # sanity check the output shape matches our reference raster

        if crater_mask.shape != ref.shape:
            raise ValueError(
                f"Built crater mask shape {crater_mask.shape} does not match expected {ref.shape}. "
                "This usually means your ROI / DEM file does not match the features grid."
            )
        
        # Save into derived cache (keeps HazardAssessor untouched)
        np.save(mask_path, crater_mask.astype(np.uint8, copy=False))
        meta_path.write_text(json.dumps(crater_meta, indent=2))

        return {
            "crater_proba": crater_mask.astype(np.float32, copy=False),
            "meta": {
                "model_id": self.model_id,
                "source": "built_from_catalogue",
                "product": str(mask_path.name),
                "saved_to": str(derived_dir),
                "catalogue": crater_meta,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_reference_raster(features: Dict[str, np.ndarray]) -> np.ndarray:
        """Pick a 2D raster from features to define output shape."""
        for key, v in features.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                return v
        raise ValueError("No 2D feature rasters provided to CraterPredictor.")


