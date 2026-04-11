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
            raise FileNotFoundError(
                f"Missing crater label raster: {crater_mask_path.name}. "
                "Build crater products first with CraterPredictor(catalogue_raster_v1)."
            )
        
        crater_mask = np.load(crater_mask_path).astype(np.uint8) # load and ensure binary uint8 type
        self._validate_same_shape("crater_mask", crater_mask, ref_shape)

        #core feature set ( just terrain )
        feature_rasters: Dict[str, np.ndarray] = {
            "elevation_m": dem_m,
            "slope_deg": slope_deg,
            "roughness_rms": roughness_rms,
        }

        # optional extra feaetures passed manually (but still terrain-derived, not crater-derived)
        if extra_feature_rasters is not None:
            for name, raster in extra_feature_rasters.items():
                arr = np.asarray(raster, dtype=np.float32) # ensure consistent dtype
                self._validate_same_shape(name, arr, ref_shape) # check shape consistency with reference
                feature_rasters[name] = arr.astype(np.float32) # store in consistent dtype

        
        df = self._rasters_to_dataframe(
            feature_rasters=feature_rasters,
            label=crater_mask,
            include_rowcol=include_rowcol,
            include_block_ids=include_block_ids,
        )

        # Remove invalid rows if requested
        if self.drop_invalid:
            df = self._drop_invalid_rows(df, feature_columns=list(feature_rasters.keys()))

        # Optional balancing
        if balance_strategy is not None:
            if balance_strategy != "undersample":
                raise ValueError(
                    f"Unsupported balance_strategy={balance_strategy}. "
                    "Currently supported: None, 'undersample'."
                )
            
            # For undersampling, keep all positive samples and randomly sample negative samples to achieve the desired negative_ratio.
            df = self._undersample_negatives(
                df=df,
                label_column=self.label_column,
                negative_ratio=negative_ratio,
            )

        df = df.reset_index(drop=True) # reset index after any filtering/sampling steps

        # Metadata
        positives = int((df[self.label_column] == 1).sum()) # count positive samples in the final dataset
        negatives = int((df[self.label_column] == 0).sum()) # count negative samples in the final dataset

        meta = {
            "tile_id": tile_id,
            "roi": list(roi),
            "shape": list(ref_shape),
            "num_rows": int(len(df)),
            "num_positive": positives,
            "num_negative": negatives,
            "positive_ratio": float(positives / len(df)) if len(df) > 0 else None,
            "feature_columns": list(feature_rasters.keys()),
            "label_column": self.label_column,
            "block_size_px": int(self.block_size_px),
            "balance_strategy": balance_strategy,
            "negative_ratio": float(negative_ratio),
            "source_paths": {
                "derived_dir": str(derived_dir),
                "crater_mask": str(crater_mask_path),
            },
        }

        paths = {} # for any output paths we want to return (e.g. saved CSV)

        #save the built dataset as a CSV in the ROI derived directory for easy access by benchmark_runner and others
        if save_csv:
            csv_path = derived_dir / csv_name
            df.to_csv(csv_path, index=False)
            paths["csv"] = str(csv_path)

        return {
            "dataframe": df,
            "meta": meta,
            "paths": paths,
        }
    
    # -------------------------------------------------------------
    # Core raster -> dataframe conversion
    # ---------------------------------------------------------
    # This is where we flatten the aligned rasters into a per-pixel DataFrame, and add row/col and block IDs if requested.
    def _rasters_to_dataframe(
        self,
        *,
        feature_rasters: Dict[str, np.ndarray],
        label: np.ndarray,
        include_rowcol: bool,
        include_block_ids: bool,
    ) -> pd.DataFrame:
        """
        Flatten aligned rasters into a per-pixel DataFrame.
        """
        ref_name, ref = next(iter(feature_rasters.items())) # take the first feature raster as reference for shape and alignment
        H, W = ref.shape

        rows, cols = np.indices((H, W), dtype=np.int32)

        data: Dict[str, np.ndarray] = {} # initialize data dict to collect columns for the DataFrame

        # Add row and col if requested
        if include_rowcol:
            data["row"] = rows.ravel()
            data["col"] = cols.ravel()

        # Add block IDs if requested (for spatial splitting later)
        if include_block_ids:
            block_row = (rows // self.block_size_px).astype(np.int32)
            block_col = (cols // self.block_size_px).astype(np.int32)
            block_id = (block_row * 100000 + block_col).astype(np.int32)

            data["block_row"] = block_row.ravel()
            data["block_col"] = block_col.ravel()
            data["block_id"] = block_id.ravel()

        # Add feature columns
        for name, arr in feature_rasters.items():
            self._validate_same_shape(name, arr, ref.shape)
            data[name] = arr.ravel()

        data[self.label_column] = label.astype(np.uint8, copy=False).ravel() # ensure label is binary uint8 and add to data dict

        return pd.DataFrame(data) # convert the collected data dict into a pandas DataFrame
    
        # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def _drop_invalid_rows(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Remove rows where any feature is NaN or infinite,since these would not be valid inputs for ML models.
        """
        valid_mask = np.ones(len(df), dtype=bool) #all true mask to start with

        # Check each feature column for finite values and update the valid_mask accordingly
        for col in feature_columns:
            vals = df[col].to_numpy()
            valid_mask &= np.isfinite(vals)

        return df.loc[valid_mask].copy() #copy any remaining valid rows into a new DataFrame and return it

    # ------------------------------------------------------------------
    # Class balancing
    # --------------------------------------------------------

    """
    Function to undersample negative samples while keeping all positive samples, based on the specified negative_ratio
    """
    def _undersample_negatives(
        self,
        *,
        df: pd.DataFrame,
        label_column: str,
        negative_ratio: float,
    ) -> pd.DataFrame:
        """
        Keep all positives, sample negatives.

        negatives_kept = positives * negative_ratio
        """
        if negative_ratio <= 0:
            raise ValueError("negative_ratio must be > 0.")

        positives = df[df[label_column] == 1] # filter the DataFrame to get all positive samples (where label_column == 1)
        negatives = df[df[label_column] == 0] # filter the DataFrame to get all negative samples (where label_column == 0)

        n_pos = len(positives) #count the number of positive samples in the dataset
        if n_pos == 0:
            raise ValueError("Cannot undersample negatives because there are no positive samples.")

        n_neg_keep = int(round(n_pos * negative_ratio)) # calculate how many negative samples to keep based on the desired negative_ratio relative to the number of positives
        n_neg_keep = min(n_neg_keep, len(negatives)) #min with the total number of available negatives to avoid trying to sample more than exist

        # sampling without replacement to get unique negative samples.
        negatives_sampled = negatives.sample(
            n=n_neg_keep,
            random_state=self.random_state,
            replace=False, # sample without replacement to get unique negative samples
        )

        out = pd.concat([positives, negatives_sampled], axis=0) #
        out = out.sample(frac=1.0, random_state=self.random_state)  # shuffle
        return out.copy() #copy and return the balanced dataset

    # ------------------------------------------------
    # Validation helpss
    # ------------------------------------------------

    @staticmethod
    def _validate_same_shape(name: str, arr: np.ndarray, expected_shape: Tuple[int, int]) -> None:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match expected shape {expected_shape}."
            )




    

