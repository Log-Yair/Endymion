# References / notes:
# - Built to match your current Endymion structure:
#   * data_handler.py -> loads derived terrain rasters from the ROI cache
#   * crater_predictor.py / crater_raster.py -> cache crater_mask.npy in the same ROI derived dir
#   * benchmark_runner.py -> already expects the ROI derived cache to contain terrain rasters
# - Design idea:
#   first ML stage = predict crater presence from terrain-derived features only.
# - Important leakage rule:
#   crater_mask is used as the LABEL, not as an input feature.
#   crater_distance_m and crater_density are also excluded as predictors in this first version.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import json
import numpy as np
import pandas as pd

from src.data.data_handler import DataHandler, ROI


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