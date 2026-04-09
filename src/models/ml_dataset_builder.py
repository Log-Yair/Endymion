#  * data_handler.py -> loads derived terrain rasters from the ROI cache
#  * crater_predictor.py / crater_raster.py -> cache crater_mask.npy in the same ROI derived dir
#  * benchmark_runner.py -> already expects the ROI derived cache to contain terrain rasters
# - Design idea:
#   first ML stage = predict crater presence from terrain-derived features only.
# - Important leakage rule:
#   crater_mask is used as the LABEL, not as an input feature.
#   crater_distance_m and crater_density are also excluded as predictors in this first version.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Required, Tuple, Any

import json
import numpy as np
import pandas as pd

from src.data.data_handler import DataHandler, ROI


# machine learning builder for the first crater model, which predicts crater presence from terrain features only
@dataclass
class MLDatasetBuilder:
    """
    Build a per-pixel supervised ML dataset for the first Endymion crater model.

    Goal
    ----
    Predict crater presence from terrain-derived features.

    Inputs expected from the ROI derived cache
    -----------------------------------------
    Required:
    - dem_m.npy
    - slope_deg.npy
    - roughness_rms.npy
    - crater_mask.npy

    Optional:
    - any extra rasters passed manually in `extra_feature_rasters`

    Output
    ------
    A pandas DataFrame where each row is one pixel:
    row, col, block_row, block_col, block_id,
    elevation_m, slope_deg, roughness_rms, ...extra features..., label_crater

    Notes
    -----
    - This builder intentionally does NOT use crater-derived rasters
      (crater_distance_m, crater_density, crater_mask) as predictors.
    - crater_mask is only used as the supervised label.
    """
    # Block size used for later spatial train/test splits
    block_size_px: int = 64

    # Label column name
    label_column: str = "label_crater"

    # Whether to drop NaN / inf rows
    drop_invalid: bool = True

    # Optional random seed for sampling / balancing
    random_state: int = 42

    # Internal cache for the built dataset
    def build_from_cache(
        self,
        dh: DataHandler,
        tile_id: str,
        roi: ROI,
        *,
        include_rowcol: bool = True,
        include_block_ids: bool = True,
        balance_strategy: Optional[str] = None,
        negative_ratio: float = 1.0,
        extra_feature_rasters: Optional[Dict[str, np.ndarray]] = None,
        save_csv: bool = False,
        csv_name: str = "ml_dataset.csv",
    ) -> Dict[str, Any]:
        """
        Build dataset from the existing ROI cache.

        Parameters
        ----------
        dh : DataHandler
            Endymion DataHandler instance.
        tile_id : str
            Tile ID used by the current ROI cache.
        roi : ROI
            (r0, r1, c0, c1)
        include_rowcol : bool
            Include row and col columns.
        include_block_ids : bool
            Include block_row, block_col, and block_id for spatial splits.
        balance_strategy : Optional[str]
            None            -> keep all samples
            "undersample"   -> keep all positives + sample negatives
        negative_ratio : float
            If balance_strategy="undersample":
            negatives kept = positives * negative_ratio
        extra_feature_rasters : Optional[Dict[str, np.ndarray]]
            Additional terrain-only feature rasters aligned to the ROI.
        save_csv : bool
            If True, save the built dataset into the ROI derived directory.
        csv_name : str
            CSV filename if save_csv=True

        Returns
        -------
        Dict[str, Any]
            {
                "dataframe": pd.DataFrame,
                "meta": dict,
                "paths": dict,
            }
        """

        derived = dh.load_derived(tile_id, roi)
        if derived is None:
            raise ValueError(f"No derived data found in cache for tile {tile_id} and ROI {roi}")
        
        # required terrain inpouts
        required = ["dem_m", "slope_deg", "roughness_rms"]
        missing = [k for k in required if k not in derived]
        if missing:
            raise ValueError(f"Missing required rasters in derived cache: {missing}")
        
        dem_m = np.asarray(derived["dem_m"], dtype=np.float32) # ensure consistent dtype
        slope_deg = np.asarray(derived["slope_deg"], dtype=np.float32) # ensure consistent dtype
        roughness_rms = np.asarray(derived["roughness_rms"], dtype=np.float32) # ensure consistent dtype

        ref_shape = dem_m.shape # reference shape for consistency checks

        self._validate_same_shape("slope_deg", slope_deg, ref_shape)
        self._validate_same_shape("roughness_rms", roughness_rms, ref_shape)

        #load crater label fronm Roi derived cache

        derived_dir = Path(dh.get_derived_dir(tile_id, roi)) # get the derived directory path
        crater_mask_path = derived_dir / "crater_mask.npy" # construct the full path to crater_mask.npy

        if not crater_mask_path.exists():
            raise ValueError(f"Missing required raster 'crater_mask.npy' in derived cache at {crater_mask_path}")