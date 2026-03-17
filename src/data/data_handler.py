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

    # =========================
    # Public API
    # =========================

    def get_patch(self, tile_id: str, roi: ROI, verbose: bool = False) -> RasterPatch:
        spec = self._get_tile(tile_id)

        if isinstance(spec, GeoTiffTileSpec):
            tif_path = self._ensure_local_geotiff(spec)
            return self._get_patch_from_geotiff(tile_id, tif_path, roi, verbose=verbose)

        if isinstance(spec, LOLATileSpec):
            img_path, lbl_path = self._ensure_local_lola(spec)
            lbl = self._parse_lbl(lbl_path)
            tile = self._load_float_img(tile_id, img_path, lbl)

            patch = self._extract_roi(tile.data, roi)
            patch_m = self._standardise_to_meters(patch, lbl)

            meta = RasterMeta(
                tile_id=tile_id,
                shape=patch_m.shape,
                dtype=str(patch_m.dtype),
                units="m",
                nodata=np.nan,
                extra={},
            )

            self._validate_patch(patch_m, roi, meta)

            if verbose:
                meta.extra.update({
                    "source_format": "img_lbl",
                    "lbl_unit": lbl.unit,
                    "lbl_scaling_factor": lbl.scaling_factor,
                    "lbl_offset": lbl.offset,
                    "tile_shape": (lbl.lines, lbl.line_samples),
                })

            return RasterPatch(data=patch_m, meta=meta, roi=roi)

        raise DataHandlerError(f"Unsupported tile spec for tile_id={tile_id}")

    # =========================
    # Tile resolution
    # =========================

    def _get_tile(self, tile_id: str) -> TileSpec:
        spec = self.tiles.get(tile_id)
        if spec is None:
            raise TileNotFoundError(
                f"Unknown tile_id '{tile_id}'. Available: {list(self.tiles.keys())}"
            )
        return spec

    def _ensure_local_lola(self, spec: LOLATileSpec) -> Tuple[Path, Path]:
        rt_img = self.runtime_dir / spec.img_filename
        rt_lbl = self.runtime_dir / spec.lbl_filename

        ps_img = self.persistent_dir / spec.img_filename
        ps_lbl = self.persistent_dir / spec.lbl_filename

        if rt_img.exists() and rt_lbl.exists() and not self.force_download:
            return rt_img, rt_lbl

        if ps_img.exists() and ps_lbl.exists() and not self.force_download:
            self._copy_if_needed(ps_img, rt_img)
            self._copy_if_needed(ps_lbl, rt_lbl)
            return rt_img, rt_lbl

        if not self.allow_download:
            raise TileNotFoundError(
                f"LOLA tile not found locally and downloads are disabled: {spec.tile_id}"
            )

        self._download_file(f"{self.base_url}/{spec.img_filename}", ps_img)
        self._download_file(f"{self.base_url}/{spec.lbl_filename}", ps_lbl)

        self._copy_if_needed(ps_img, rt_img)
        self._copy_if_needed(ps_lbl, rt_lbl)
        return rt_img, rt_lbl

    def _ensure_local_geotiff(self, spec: GeoTiffTileSpec) -> Path:
        rt_tif = self.runtime_dir / spec.tif_filename
        ps_tif = self.persistent_dir / spec.tif_filename

        if rt_tif.exists() and not self.force_download:
            return rt_tif

        if ps_tif.exists() and not self.force_download:
            self._copy_if_needed(ps_tif, rt_tif)
            return rt_tif

        if not self.allow_download:
            raise TileNotFoundError(
                f"GeoTIFF not found locally and downloads are disabled: {spec.tile_id}"
            )

        self._download_file(f"{self.base_url}/{spec.tif_filename}", ps_tif)
        self._copy_if_needed(ps_tif, rt_tif)
        return rt_tif

    def _copy_if_needed(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)

    def _download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not self.force_download:
            return

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=(15, self.timeout_s)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(out_path)

    # =========================
    # GeoTIFF loading
    # =========================

    def _get_patch_from_geotiff(
        self,
        tile_id: str,
        tif_path: Path,
        roi: ROI,
        verbose: bool = False,
    ) -> RasterPatch:
        r0, r1, c0, c1 = roi

        with rasterio.open(tif_path) as ds:
            if ds.count < 1:
                raise ValidationError(f"No raster band found in {tif_path}")

            if not (0 <= r0 < r1 <= ds.height and 0 <= c0 < c1 <= ds.width):
                raise ValidationError(
                    f"ROI {roi} out of bounds for GeoTIFF shape {(ds.height, ds.width)}"
                )

            patch = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32, copy=False)

            nodata = ds.nodata
            if nodata is not None and np.isfinite(nodata):
                patch = patch.copy()
                patch[patch == nodata] = np.nan

            meta = RasterMeta(
                tile_id=tile_id,
                shape=patch.shape,
                dtype=str(patch.dtype),
                units="m",
                nodata=np.nan,
                extra={},
            )

            self._validate_patch(patch, roi, meta)

            if verbose:
                meta.extra.update({
                    "source_format": "geotiff",
                    "tif_path": str(tif_path),
                    "crs": str(ds.crs),
                    "transform": tuple(ds.transform),
                    "full_shape": (ds.height, ds.width),
                    "resolution": ds.res,
                })

        return RasterPatch(data=patch, meta=meta, roi=roi)

    # =========================
    # LBL parsing + IMG loading
    # =========================

    def _parse_lbl(self, lbl_path: Path) -> LBLMeta:
        fields: Dict[str, str] = {}

        try:
            text = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            raise LabelParseError(f"Failed to read LBL: {lbl_path} ({e})")

        for raw in text:
            line = raw.strip()
            if (not line) or line.startswith("/*") or "=" not in line:
                continue

            key, val = [x.strip() for x in line.split("=", 1)]
            key_u = key.upper()

            if key_u in {
                "LINES", "LINE_SAMPLES", "SAMPLE_TYPE", "SAMPLE_BITS",
                "UNIT", "SCALING_FACTOR", "OFFSET", "MISSING_CONSTANT"
            }:
                fields[key_u] = val.strip().strip('"')

        def req_int(k: str) -> int:
            if k not in fields:
                raise LabelParseError(f"LBL missing required field: {k}")
            return int(fields[k])

        def req_str(k: str) -> str:
            if k not in fields:
                raise LabelParseError(f"LBL missing required field: {k}")
            return str(fields[k])

        return LBLMeta(
            lines=req_int("LINES"),
            line_samples=req_int("LINE_SAMPLES"),
            sample_type=req_str("SAMPLE_TYPE"),
            sample_bits=req_int("SAMPLE_BITS"),
            unit=fields.get("UNIT"),
            scaling_factor=float(fields["SCALING_FACTOR"]) if "SCALING_FACTOR" in fields else None,
            offset=float(fields["OFFSET"]) if "OFFSET" in fields else None,
            missing_constant=float(fields["MISSING_CONSTANT"]) if "MISSING_CONSTANT" in fields else None,
            extra={k: v for k, v in fields.items()},
        )

    def _numpy_dtype_from_sample_type(self, sample_type: str, sample_bits: int) -> np.dtype:
        st = (sample_type or "").upper()

        if ("REAL" not in st) and ("FLOAT" not in st):
            raise ValueError(f"Unsupported SAMPLE_TYPE: {sample_type}")

        if sample_bits == 32:
            if ("PC" in st) or ("LSB" in st):
                return np.dtype("<f4")
            if "MSB" in st:
                return np.dtype(">f4")
            return np.dtype("f4")

        if sample_bits == 64:
            if ("PC" in st) or ("LSB" in st):
                return np.dtype("<f8")
            if "MSB" in st:
                return np.dtype(">f8")
            return np.dtype("f8")

        raise ValueError(f"Unsupported SAMPLE_BITS: {sample_bits}")

    def _load_float_img(self, tile_id: str, img_path: Path, lbl: LBLMeta) -> RasterTile:
        dtype = self._numpy_dtype_from_sample_type(lbl.sample_type, lbl.sample_bits)

        expected_bytes = lbl.lines * lbl.line_samples * (lbl.sample_bits // 8)
        actual_bytes = img_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise IOError(f"IMG too small for tile '{tile_id}'")

        arr = np.memmap(
            img_path,
            dtype=dtype,
            mode="r",
            shape=(lbl.lines, lbl.line_samples),
        )
        return RasterTile(tile_id=tile_id, data=arr, lbl=lbl)

