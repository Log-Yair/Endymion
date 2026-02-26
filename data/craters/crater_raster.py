# References / origin notes:
# - Uses GDAL/OSR to avoid hand-deriving polar stereographic math:
#   lon/lat (EPSG:4326) -> dataset projection -> pixel (geoTransform).
# - Circle rasterisation approach is standard: bounding-box loop + distance check.
#
# Requirements:
#   pip install numpy pandas
#   and GDAL available in environment (often preinstalled in Colab via apt).
#
# Visual intuition:
#   crater circle fill: (r-r0)^2 + (c-c0)^2 <= radius_px^2

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass   

import numpy as np
import pandas as pd 

try:
    from osgeo import gdal, osr
except Exception as e:
    raise ImportError(
        "GDAL python bindings not found. "
        "In Colab you can usually install via apt: "
        "`!apt-get update -qq && apt-get install -y -qq gdal-bin python3-gdal`"
    ) from e
