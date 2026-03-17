# References / notes:
# - Extends your current Endymion data_handler.py design.
# - GeoTIFF support uses rasterio because crater_raster.py already expects
#   a geospatial raster with CRS + transform.
# - Idea based on your existing DataHandler flow plus Rasterio dataset access.
# - Keep IMG/LBL support for backward compatibility during migration.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
import json
import shutil
import requests
import numpy as np
import rasterio

# =========================
# Types
# =========================

ROI = Tuple[int, int, int, int]  # (r0, r1, c0, c1)


@dataclass(frozen=True)
class LOLATileSpec:
    tile_id: str
    img_filename: str
    lbl_filename: str


@dataclass(frozen=True)
class GeoTiffTileSpec:
    tile_id: str
    tif_filename: str


TileSpec = Union[LOLATileSpec, GeoTiffTileSpec]


@dataclass
class LBLMeta:
    lines: int
    line_samples: int
    sample_type: str
    sample_bits: int
    unit: Optional[str] = None
    scaling_factor: Optional[float] = None
    offset: Optional[float] = None
    missing_constant: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterMeta:
    tile_id: str
    shape: Tuple[int, int]
    dtype: str
    units: str = "m"
    nodata: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterPatch:
    data: np.ndarray
    meta: RasterMeta
    roi: ROI


@dataclass
class RasterTile:
    tile_id: str
    data: np.ndarray
    lbl: Optional[LBLMeta] = None


class DataHandlerError(Exception):
    pass


class TileNotFoundError(DataHandlerError):
    pass


class LabelParseError(DataHandlerError):
    pass


class ValidationError(DataHandlerError):
    pass


class DataHandler:
    """
    Endymion DataHandler with dual-format support:
    - LOLA IMG + LBL
    - GeoTIFF / COG GeoTIFF

    Strategy:
    - keep old prototype support
    - add GeoTIFF as the preferred Phase-2 path
    """

    def __init__(
        self,
        base_url: str,
        tiles: List[TileSpec],
        cache_dir: str | Path = "./data_cache/lola",
        persistent_dir: str | Path | None = None,
        runtime_dir: str | Path | None = None,
        allow_download: bool = True,
        force_download: bool = False,
        timeout_s: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.tiles: Dict[str, TileSpec] = {t.tile_id: t for t in tiles}

        self.persistent_dir = Path(persistent_dir) if persistent_dir else Path(cache_dir)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.persistent_dir

        self.persistent_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.allow_download = allow_download
        self.force_download = force_download
        self.timeout_s = timeout_s

        self.cache_dir = self.runtime_dir  # backward compatibility

