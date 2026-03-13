# References / notes:
#     * crater_raster.py = build raster products from catalogue
#     * crater_predictor.py = load/cache/expose pipeline outputs
# - This version still supports:
#     1) stub_v1
#     2) catalogue_raster_v1
# - New outputs exposed:
#     * crater_mask
#     * crater_distance_m
#     * crater_density
#     * crater_proba  (kept for future ML compatibility)

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from turtle import distance
from typing import Any, Dict, Optional
import json

import numpy as np

from src.data.crater_raster import (build_crater_mask_from_catalogue, CraterRasterConfig )


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
        roi_pixels: Optional[tuple[int, int, int, int]] = None, # (row_start, row_end, col_start, col_end) in pixel space
        dh: Any = None,  # DataHandler (kept as Any to avoid hard import cycles)
        dem_img_path: Optional[str | Path] = None, # for coordinate transforms in catalogue_raster mode
        robbins_csv_path: Optional[str | Path] = None, # path to Robbins crater catalogue CSV for catalogue_raster mode
        raster_config: Optional[CraterRasterConfig] = None,  # crater_raster.CraterRasterConfig
        rebuild_if_missing: bool = False, # if True, will build the crater mask from catalogue if not found in cache (only for catalogue_raster mode)
    ) -> Dict[str, np.ndarray]:
        """
        Outputs is alawys "crater_proba" for pipeline consistency, even if it's binary in catalogue_raster mode.
        """
        # pick any raster as shape reference
        ref = self._get_reference_raster(features)

        if self.model_id == "stub_v1":
            zeros = np.zeros(ref.shape, dtype=np.float32)

            return {
                "crater_proba": zeros,
                "crater_mask": np.zeros(ref.shape, dtype=np.uint8),
                "crater_distance_m": zeros.copy(),
                "crater_density": zeros.copy(),
                "meta": {
                    "model_id": self.model_id,
                    "note": "Phase-1 stub: returns zero crater outputs.",
                },
            }

        if self.model_id != "catalogue_raster_v1":
            raise ValueError(f"Unknown model_id: {self.model_id}")

        if dh is None or tile_id is None or roi_pixels is None:
            raise ValueError(
                "catalogue_raster_v1 requires dh, tile_id, and roi_pixels for cache handling."
            )
        

        derived_dir = Path(dh.derived_dir(tile_id, roi_pixels)) # get the derived directory for this tile and ROI
        derived_dir.mkdir(parents=True, exist_ok=True) # ensure it exists

        # Check if cached mask exists

        mask_path = derived_dir / f"{self.cache_product_name}.npy"      # e.g. crater_mask.npy
        distance_path = derived_dir / f"{self.cache_product_name}_distance.npy" # e.g. crater_mask_distance.npy
        density_path = derived_dir / f"{self.cache_product_name}_density.npy" # e.g. crater_mask_density.npy
        meta_path = derived_dir / f"{self.cache_product_name}_meta.json" # e.g. crater_mask_meta.json
        
        # If the cache exists load it

        cache_exists = (
            mask_path.exists()
            and distance_path.exists()
            and density_path.exists()
            and meta_path.exists()
            
            )

        if cache_exists and not rebuild_if_missing:
            crater_mask = np.load(mask_path)
            crater_distance_m = np.load(distance_path)
            crater_density = np.load(density_path)
            meta = json.loads(meta_path.read_text())

            self._validate_cached_shape("crater_mask", crater_mask, ref.shape, tile_id, roi_pixels)
            self._validate_cached_shape("crater_distance_m", crater_distance_m, ref.shape, tile_id, roi_pixels)
            self._validate_cached_shape("crater_density", crater_density, ref.shape, tile_id, roi_pixels)

            return {
                "crater_proba": crater_mask.astype(np.float32, copy=False),
                "crater_mask": crater_mask.astype(np.uint8, copy=False),
                "crater_distance_m": crater_distance_m.astype(np.float32, copy=False),
                "crater_density": crater_density.astype(np.float32, copy=False),
                "meta": {
                    "model_id": self.model_id,
                    "source": "derived_cache",
                    "products": {
                        "crater_mask": str(mask_path.name),
                        "crater_distance_m": str(distance_path.name),
                        "crater_density": str(density_path.name),
                    },
                    "catalogue": meta,
                },
            }


        # If not, build it from the catalogue
        if dem_img_path is None or robbins_csv_path is None:
            raise ValueError(
                "Crater products not found in cache. Provide dem_img_path and robbins_csv_path "
                "to build them, or ensure the cache exists."
            )
        # Use provided raster_config or default if not provided
        cfg = raster_config if raster_config is not None else CraterRasterConfig() # use provided config or default config

        # build the crater mask from the catalogue (this will also handle coordinate transforms and rasterisation)
        out = build_crater_products_from_catalogue(
            robbins_csv_path=robbins_csv_path,
            dem_img_path=dem_img_path,
            roi_pixels=roi_pixels,
            config=cfg,
        )

    # Extract outputs

        crater_mask = out["crater_mask"] # binary mask of shape (rows, cols) where 1 indicates crater presence according to the catalogue
        crater_meta = out["metadata"] # e.g. number of craters rasterised, any warnings about craters outside the ROI, etc.
        crater_distance_m = out["crater_distance_m"] # raster of same shape where each pixel value is the distance in meters to the nearest crater rim
        crater_density = out["crater_density"] # raster of same shape where each pixel value

        # is the local crater density (e.g. number of craters per square km) calculated using a kernel density estimation approach
        self._validate_cached_shape("crater_mask", crater_mask, ref.shape, tile_id, roi_pixels)
        # validate shapes before saving to cache, to catch any issues early and avoid caching invalid products
        self._validate_cached_shape("crater_distance_m", crater_distance_m, ref.shape, tile_id, roi_pixels)
        # validate_shapes(crater_density, ref.shape)
        self._validate_cached_shape("crater_density", crater_density, ref.shape, tile_id, roi_pixels)

        np.save(mask_path, crater_mask.astype(np.uint8, copy=False)) # save binary mask as uint8 to save space (0 and 1 values only)
        np.save(distance_path, crater_distance_m.astype(np.float32, copy=False)) # save distance raster as float32
        np.save(density_path, crater_density.astype(np.float32, copy=False)) # save density raster as float32
        meta_path.write_text(json.dumps(crater_meta, indent=2)) # save metadata as JSON for human readability

        return {
            "crater_proba": crater_mask.astype(np.float32, copy=False), # save as float32 for probability values
            "crater_mask": crater_mask.astype(np.uint8, copy=False),
            "crater_distance_m": crater_distance_m.astype(np.float32, copy=False),
            "crater_density": crater_density.astype(np.float32, copy=False),
            "meta": {
                "model_id": self.model_id,
                "source": "built_from_catalogue",
                "saved_to": str(derived_dir),
                "products": {
                    "crater_mask": str(mask_path.name),
                    "crater_distance_m": str(distance_path.name),
                    "crater_density": str(density_path.name),
                },
                "catalogue": crater_meta,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_reference_raster(features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Pick any 2D feature raster to define the expected output shape.
        """
        for _, value in features.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return value
        raise ValueError("No 2D feature rasters provided to CraterPredictor.")

    @staticmethod
    def _validate_cached_shape(
        product_name: str,
        arr: np.ndarray,
        expected_shape: tuple[int, int],
        tile_id: str,
        roi_pixels: tuple[int, int, int, int],
    ) -> None:
        if arr.shape != expected_shape:
            raise ValueError(
                f"Cached/built {product_name} shape {arr.shape} does not match expected "
                f"{expected_shape} (tile_id={tile_id}, roi_pixels={roi_pixels})."
            )                                                                   