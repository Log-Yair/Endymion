# - Extends the current Endymion data_handler.py design.
# - GeoTIFF support uses rasterio because crater_raster.py already expects
#   a geospatial raster with CRS + transform.
# - Idea based on your existing DataHandler flow plus Rasterio dataset access.
# - Keep IMG/LBL support for backward compatibility during migration.

# new notes:
# - Final Endymion canonical ROI policy for the GeoTIFF workflow.
# - Based on the selected south-polar COG product:
#   LDEM_80S_20MPP_ADJ.TIF (80°S to 90°S, 20 m/pixel).
# - The canonical ROI is intentionally defined from the raster grid itself, not copied from the old IMG/LBL prototype coordinates.
# - This keeps the ROI stable even if the storage format changes, as long as the GeoTIFF grid remains the same.

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

CANONICAL_PATCH_SIZE = 1024  # pixels (for the GeoTIFF workflow, this is the standard patch size we will use for processing and feature extraction)


@dataclass(frozen=True)  # immutable dataclass for tile specifications
class LOLATileSpec:
    tile_id: str
    img_filename: str
    lbl_filename: str


@dataclass(frozen=True)  # immutable dataclass for GeoTIFF tile specifications
class GeoTiffTileSpec:
    tile_id: str
    tif_filename: str
    tif_url: str | None = None


TileSpec = Union[LOLATileSpec, GeoTiffTileSpec]  # a tile can be specified as either an LOLA IMG/LBL pair or a single GeoTIFF file


# Metadata classes for LBL and raster data, plus container classes for patches and tiles.
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


# =========================
# ROI
# =========================
def centered_roi(height: int, width: int, size: int = CANONICAL_PATCH_SIZE) -> ROI:
    """
    Return a square ROI centred on the raster grid.
    This is the final canonical ROI rule for the GeoTIFF-based Endymion pipeline.
    For the south-polar stereographic LOLA GeoTIFF, the raster centre corresponds
    to the lunar south-pole reference area.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if size > height or size > width:
        raise ValueError(
            f"Requested size {size} exceeds raster shape {(height, width)}"
        )

    r0 = (height - size) // 2
    c0 = (width - size) // 2
    r1 = r0 + size
    c1 = c0 + size
    return (r0, r1, c0, c1)


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
    # Logging
    # =========================
    def _log(self, message: str) -> None:
        """Print simple progress messages for long-running file operations."""
        print(f"[DataHandler] {message}", flush=True)    

    # =========================
    # Tile metadata and raster shape
    # =========================
    def get_raster_shape(self, tile_id: str) -> Tuple[int, int]:
        """
        Return full raster shape (rows, cols) for the selected tile.
        This step may trigger cache lookup or download for GeoTIFF inputs.
        """
        spec = self._get_tile(tile_id)

        if isinstance(spec, GeoTiffTileSpec):
            self._log(f"Resolving GeoTIFF for tile '{tile_id}'...")
            tif_path = self._ensure_local_geotiff(spec)
            self._log(f"Opening GeoTIFF to read raster shape: {tif_path}")
            with rasterio.open(tif_path) as ds:
                shape = (ds.height, ds.width)
            self._log(f"Raster shape resolved: {shape}")
            return shape

        if isinstance(spec, LOLATileSpec):
            self._log(f"Resolving IMG/LBL tile '{tile_id}'...")
            img_path, lbl_path = self._ensure_local_lola(spec)
            lbl = self._parse_lbl(lbl_path)
            shape = (lbl.lines, lbl.line_samples)
            self._log(f"Raster shape resolved: {shape}")
            return shape

        raise TileNotFoundError(f"Unsupported tile spec type for tile_id={tile_id}")

    def canonical_roi(self, tile_id: str, size: int = CANONICAL_PATCH_SIZE) -> ROI:
        """
        Resolve the final canonical ROI for a tile.
        For the GeoTIFF workflow, this is the centred south-pole reference ROI.
        """
        self._log(f"Resolving canonical ROI for tile '{tile_id}' with size {size}...") # simple log message for ROI resolution start
        height, width = self.get_raster_shape(tile_id) # get raster shape (may trigger cache/download)
        roi = centered_roi(height, width, size=size) # compute centered ROI based on raster shape and requested size
        self._log(f"Canonical ROI resolved: {roi}") # simple log message for ROI resolution result
        return roi

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
        rt_tif = self.runtime_dir / spec.tif_filename # runtime cache path for the GeoTIFF
        ps_tif = self.persistent_dir / spec.tif_filename # persistent cache path for the GeoTIFF

        if rt_tif.exists() and not self.force_download:
            self._log(f"GeoTIFF found in runtime cache: {rt_tif}") # simple log message for cache hits
            return rt_tif

        if ps_tif.exists() and not self.force_download:
            self._log(f"GeoTIFF found in persistent cache: {ps_tif}") # simple log message for persistent cache hits
            self._copy_if_needed(ps_tif, rt_tif)
            return rt_tif

        # if it reaches this point, the GeoTIFF is not available locally and we need to download it (if allowed)
        if not self.allow_download:
            raise TileNotFoundError(
                f"GeoTIFF not found locally and downloads are disabled: {spec.tile_id}"
            )

        url = spec.tif_url if spec.tif_url else f"{self.base_url}/{spec.tif_filename}" # construct download URL
        self._log(f"GeoTIFF not found locally. Downloading from: {url}") # simple log message for download attempts
        self._download_file(url, ps_tif) # download to persistent cache first
        self._copy_if_needed(ps_tif, rt_tif) # copy to runtime cache for processing
        self._log(f"GeoTIFF ready in runtime cache: {rt_tif}") # simple log message for completion
        return rt_tif

    # copy file if it doesn't exist in runtime cache or is different size than source (simple heuristic to avoid redundant copies)
    def _copy_if_needed(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True) # ensure destination directory exists

        # assume file is already copied if it exists with the same size
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return

        self._log(f"Copying file into runtime cache: {src.name}") # simple log message for copy operations
        shutil.copy2(src, dst) 

    "download file with progress logging and timeout handling (using requests with streaming to handle large files and provide progress updates. truly needed for the large GeoTIFFs im working with, but also useful for the smaller LOLA files to keep the user informed.)"
    def _download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not self.force_download:
            self._log(f"Download skipped; file already exists: {out_path}") # simple log message for existing files when downloads are skipped
            return

        tmp = out_path.with_suffix(out_path.suffix + ".part") # temporary file path during download to avoid leaving incomplete files if interrupted
        self._log(f"Starting download: {out_path.name}") # simple log message for download start

        with requests.get(url, stream=True, timeout=(15, self.timeout_s)) as r:
            r.raise_for_status()
            # use Content-Length header for progress reporting if available, otherwise just report downloaded MB without percentage
            total_bytes = int(r.headers.get("Content-Length", "0"))
            downloaded = 0
            next_report_mb = 50.0 # report progress every 50 MB downloaded

            if total_bytes > 0:
                total_mb = total_bytes / (1024 * 1024) # convert to MB for logging
                self._log(f"Remote size: {total_mb:.1f} MB") # simple log message for remote file size

            # stream download to file with progress updates. this is important for large GeoTIFFs to avoid memory issues and provide user feedback during potentially long downloads.
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    downloaded_mb = downloaded / (1024 * 1024)
                    if downloaded_mb >= next_report_mb:
                        if total_bytes > 0:
                            pct = 100.0 * downloaded / total_bytes
                            self._log(
                                f"Download progress: {downloaded_mb:.1f} MB / "
                                f"{total_bytes / (1024 * 1024):.1f} MB ({pct:.1f}%)"
                            )
                        else:
                            self._log(f"Download progress: {downloaded_mb:.1f} MB")
                        next_report_mb += 50.0

        tmp.replace(out_path) # move temp file to final location atomically
        self._log(f"Download complete: {out_path}") # simple log message for download completion

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

    def _extract_roi(self, raster: np.ndarray, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= raster.shape[0] and 0 <= c0 < c1 <= raster.shape[1]):
            raise ValidationError(f"ROI {roi} out of bounds for raster shape {raster.shape}")
        return np.array(raster[r0:r1, c0:c1], dtype=np.float32, copy=True)

    def _standardise_to_meters(self, patch: np.ndarray, lbl: LBLMeta) -> np.ndarray:
        out = patch.astype(np.float32, copy=False)

        if lbl.missing_constant is not None:
            out[out == float(lbl.missing_constant)] = np.nan

        if lbl.scaling_factor is not None:
            out = (out * float(lbl.scaling_factor)).astype(np.float32, copy=False)

        if (lbl.unit or "").upper() in {"KILOMETER", "KILOMETRE", "KM"}:
            out = (out * 1000.0).astype(np.float32, copy=False)

        if lbl.offset is not None:
            out = (out + float(lbl.offset)).astype(np.float32, copy=False)

        return out

    def _validate_patch(self, patch_m: np.ndarray, roi: ROI, meta: RasterMeta) -> None:
        r0, r1, c0, c1 = roi
        expected_shape = (r1 - r0, c1 - c0)
        if patch_m.shape != expected_shape:
            raise ValidationError(f"Patch shape {patch_m.shape} != expected {expected_shape}")

        nan_ratio = float(np.isnan(patch_m).mean())
        if nan_ratio > 0.95:
            raise ValidationError(f"Patch NaN ratio too high: {nan_ratio:.3f}")

        vmin = float(np.nanmin(patch_m))
        vmax = float(np.nanmax(patch_m))
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValidationError("Patch has no finite values.")

        meta.extra.update({
            "nan_ratio": nan_ratio,
            "min_m": vmin,
            "max_m": vmax,
        })

    # =========================
    # Derived / navigation storage
    # =========================

    def _roi_key(self, roi: ROI) -> str:
        r0, r1, c0, c1 = roi
        return f"roi_{r0}_{r1}_{c0}_{c1}"

    def derived_dir(self, tile_id: str, roi: ROI) -> Path:
        return self.persistent_dir / "derived" / tile_id / self._roi_key(roi)

    def navigation_dir(self, tile_id: str, roi: ROI, run_name: str) -> Path:
        return self.derived_dir(tile_id, roi) / "navigation" / run_name

    def save_derived(
        self,
        tile_id: str,
        roi: ROI,
        dem_m: np.ndarray,
        features: Dict[str, np.ndarray],
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out_dir = self.derived_dir(tile_id, roi)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "dem_m.npy", dem_m.astype(np.float32, copy=False))

        if "slope_deg" in features:
            np.save(out_dir / "slope_deg.npy", features["slope_deg"].astype(np.float32, copy=False))
        if "roughness_rms" in features:
            np.save(out_dir / "roughness_rms.npy", features["roughness_rms"].astype(np.float32, copy=False))

        meta = {
            "tile_id": tile_id,
            "roi": list(roi),
            "shape": list(dem_m.shape),
            "units": "m",
            "products": [p for p in ["dem_m", "slope_deg", "roughness_rms"] if (out_dir / f"{p}.npy").exists()],
            "extra": meta_extra or {},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return out_dir

    def load_derived(self, tile_id: str, roi: ROI) -> Optional[Dict[str, Any]]:
        out_dir = self.derived_dir(tile_id, roi)
        meta_path = out_dir / "meta.json"
        dem_path = out_dir / "dem_m.npy"

        if not meta_path.exists() or not dem_path.exists():
            return None

        meta = json.loads(meta_path.read_text())

        out: Dict[str, Any] = {
            "meta": meta,
            "dem_m": np.load(dem_path),
        }

        slope_path = out_dir / "slope_deg.npy"
        rough_path = out_dir / "roughness_rms.npy"

        if slope_path.exists():
            out["slope_deg"] = np.load(slope_path)
        if rough_path.exists():
            out["roughness_rms"] = np.load(rough_path)

        return out

    def save_navigation(
        self,
        tile_id: str,
        roi: ROI,
        hazard: np.ndarray,
        cost: np.ndarray,
        path_rc: np.ndarray,
        run_name: str = "run_v1",
        nav_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out_dir = self.navigation_dir(tile_id, roi, run_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "hazard.npy", hazard.astype(np.float32, copy=False))
        np.save(out_dir / "cost.npy", cost.astype(np.float32, copy=False))
        np.save(out_dir / "path_rc.npy", np.asarray(path_rc, dtype=np.int32))

        meta = {
            "tile_id": tile_id,
            "roi": list(roi),
            "shape": list(hazard.shape),
            "run_name": run_name,
            "products": ["hazard", "cost", "path_rc"],
            "nav_meta": nav_meta or {},
        }
        (out_dir / "nav_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
        return out_dir