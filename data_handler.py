# -*- coding: utf-8 -*-
"""
Endymion DataHandler (LOLA + Derived Products + Navigation Runs)

Responsibilities:
- Resolve LOLA tiles (runtime cache <-> persistent cache; download if allowed)
- Parse .LBL metadata (minimal PDS3-like parsing)
- Load .IMG raster as memmap
- Extract ROI patches and standardise to Endymion convention:
    * float32
    * metres
    * NaNs for nodata
- Validate patches
- Persist derived products (dem/slope/roughness)
- Persist navigation runs (hazard/cost/path/meta)

Notes:
- 'persistent_dir' is the durable cache (e.g. Google Drive)
- 'runtime_dir' is the fast cache (e.g. local Colab runtime)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import json
import shutil
import requests
import numpy as np


# 
# Types
# 

ROI = Tuple[int, int, int, int]  # (r0, r1, c0, c1)


@dataclass(frozen=True)
class LOLATileSpec:
    """
    Defines where a tile lives and which files belong to it.
    """
    tile_id: str                      # e.g. "ldem_85s_20m"
    img_filename: str                 # e.g. "LDEM_85S_20M_FLOAT.IMG"
    lbl_filename: str                 # e.g. "LDEM_85S_20M_FLOAT.LBL"


@dataclass
class LBLMeta:
    """
    Minimal PDS label fields we actually need for loading the raster.
    """
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
    """
    What the rest of Endymion can rely on.
    """
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
    """
    Internal convenience type (full tile, usually memmap-backed).
    """
    tile_id: str
    data: np.ndarray
    lbl: LBLMeta


# 
# Exceptions
# 

class DataHandlerError(Exception):
    pass


class TileNotFoundError(DataHandlerError):
    pass


class LabelParseError(DataHandlerError):
    pass


class ValidationError(DataHandlerError):
    pass


# 
# DataHandler
# 

class DataHandler:
    """
    Endymion DataHandler:
    - resolves tiles (local cache; download hook optional)
    - parses .LBL
    - loads .IMG as memmap
    - extracts ROI
    - standardises to metres + float32 + NaNs
    - sanity checks
    - stores derived products + navigation runs under persistent_dir/derived/
    """

    def __init__(
        self,
        base_url: str,
        tiles: List[LOLATileSpec],
        cache_dir: str | Path = "./data_cache/lola",   # backwards compat
        persistent_dir: str | Path | None = None,      # Drive cache
        runtime_dir: str | Path | None = None,         # local/runtime cache
        allow_download: bool = True,
        force_download: bool = False,
        timeout_s: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.tiles: Dict[str, LOLATileSpec] = {t.tile_id: t for t in tiles}

        # Backwards-compatible default: behave like before
        self.persistent_dir = Path(persistent_dir) if persistent_dir else Path(cache_dir)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.persistent_dir

        self.persistent_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.allow_download = allow_download
        self.force_download = force_download
        self.timeout_s = timeout_s

        # For backwards compatibility with older notebooks/code
        self.cache_dir = self.runtime_dir

    # 
    # Public API
    # 

    def get_patch(self, tile_id: str, roi: ROI, verbose: bool = False) -> RasterPatch:
        """
        Main entry point used by your prototype runner/pipeline.
        """
        tile_spec = self._get_tile(tile_id)
        img_path, lbl_path = self._ensure_local(tile_spec)

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
                "lbl_unit": lbl.unit,
                "lbl_scaling_factor": lbl.scaling_factor,
                "lbl_offset": lbl.offset,
                "tile_shape": (lbl.lines, lbl.line_samples),
            })

        return RasterPatch(data=patch_m, meta=meta, roi=roi)

    # 
    # Tile resolution / local cache
    # 

    def _get_tile(self, tile_id: str) -> LOLATileSpec:
        spec = self.tiles.get(tile_id)
        if spec is None:
            raise TileNotFoundError(
                f"Unknown tile_id '{tile_id}'. Available: {list(self.tiles.keys())}"
            )
        return spec

    def _ensure_local(self, spec: LOLATileSpec) -> Tuple[Path, Path]:
        """
        Ensures (IMG, LBL) exist in runtime_dir (fast path).
        Uses persistent_dir as the durable store.
        """
        rt_img = self.runtime_dir / spec.img_filename
        rt_lbl = self.runtime_dir / spec.lbl_filename

        ps_img = self.persistent_dir / spec.img_filename
        ps_lbl = self.persistent_dir / spec.lbl_filename

        # 1) runtime cache hit
        if rt_img.exists() and rt_lbl.exists() and not self.force_download:
            return rt_img, rt_lbl

        # 2) persistent cache hit -> copy to runtime
        if ps_img.exists() and ps_lbl.exists() and not self.force_download:
            self._copy_if_needed(ps_img, rt_img)
            self._copy_if_needed(ps_lbl, rt_lbl)
            return rt_img, rt_lbl

        # 3) download
        if not self.allow_download:
            raise TileNotFoundError(
                f"Tile not found.\n"
                f"Runtime: {rt_img.name} / {rt_lbl.name} under {self.runtime_dir}\n"
                f"Persistent: {ps_img.name} / {ps_lbl.name} under {self.persistent_dir}\n"
                f"Downloads disabled (allow_download=False)."
            )

        # Download into persistent, then copy to runtime
        self._download_file(f"{self.base_url}/{spec.img_filename}", ps_img)
        self._download_file(f"{self.base_url}/{spec.lbl_filename}", ps_lbl)

        self._copy_if_needed(ps_img, rt_img)
        self._copy_if_needed(ps_lbl, rt_lbl)

        if not rt_img.exists():
            raise TileNotFoundError(f"Missing IMG file after download/copy: {rt_img}")
        if not rt_lbl.exists():
            raise TileNotFoundError(f"Missing LBL file after download/copy: {rt_lbl}")

        return rt_img, rt_lbl

    def _copy_if_needed(self, src: Path, dst: Path) -> None:
        """
        Copy src -> dst if missing or size mismatch.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)

    def _download_file(self, url: str, out_path: Path) -> None:
        """
        Streaming download with a temporary '.part' file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not self.force_download:
            return

        tmp = out_path.with_suffix(out_path.suffix + ".part")

        # Use conservative timeouts: (connect, read)
        with requests.get(url, stream=True, timeout=(15, self.timeout_s)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0"))
            done = 0

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)

        tmp.replace(out_path)

    # 
    # Label parsing (LBL)
    # 

    def _parse_lbl(self, lbl_path: Path) -> LBLMeta:
        """
        Minimal PDS3-ish parser:
        - reads KEY = VALUE lines
        - ignores OBJECT blocks
        - captures the fields we need for memmap loading + unit conversion
        """
        fields: Dict[str, str] = {}

        try:
            text = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            raise LabelParseError(f"Failed to read LBL: {lbl_path} ({e})")

        for raw in text:
            line = raw.strip()
            if (not line) or line.startswith("/*"):
                continue
            if "=" not in line:
                continue

            key, val = [x.strip() for x in line.split("=", 1)]
            key_u = key.upper()

            if key_u in {
                "LINES",
                "LINE_SAMPLES",
                "SAMPLE_TYPE",
                "SAMPLE_BITS",
                "UNIT",
                "SCALING_FACTOR",
                "OFFSET",
                "MISSING_CONSTANT",
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

        lines = req_int("LINES")
        line_samples = req_int("LINE_SAMPLES")
        sample_type = req_str("SAMPLE_TYPE")
        sample_bits = req_int("SAMPLE_BITS")

        unit = fields.get("UNIT")
        scaling_factor = float(fields["SCALING_FACTOR"]) if "SCALING_FACTOR" in fields else None
        offset = float(fields["OFFSET"]) if "OFFSET" in fields else None
        missing_constant = float(fields["MISSING_CONSTANT"]) if "MISSING_CONSTANT" in fields else None

        return LBLMeta(
            lines=lines,
            line_samples=line_samples,
            sample_type=sample_type,
            sample_bits=sample_bits,
            unit=unit,
            scaling_factor=scaling_factor,
            offset=offset,
            missing_constant=missing_constant,
            extra={k: v for k, v in fields.items()},
        )

    # 
    # IMG loading (float_img)
    # 

    def _numpy_dtype_from_sample_type(self, sample_type: str, sample_bits: int) -> np.dtype:
        """
        Maps PDS SAMPLE_TYPE/SAMPLE_BITS to numpy dtype.

        For LOLA float_img tiles typically:
          SAMPLE_TYPE = PC_REAL
          SAMPLE_BITS = 32  (little-endian float32)
        """
        st = (sample_type or "").upper()

        if ("REAL" not in st) and ("FLOAT" not in st):
            raise ValueError(f"Unsupported SAMPLE_TYPE (expected REAL/FLOAT): {sample_type}")

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

        raise ValueError(f"Unsupported SAMPLE_TYPE/SAMPLE_BITS: {sample_type}/{sample_bits}")

    def _load_float_img(self, tile_id: str, img_path: Path, lbl: LBLMeta) -> RasterTile:
        dtype = self._numpy_dtype_from_sample_type(lbl.sample_type, lbl.sample_bits)

        expected_bytes = lbl.lines * lbl.line_samples * (lbl.sample_bits // 8)
        actual_bytes = img_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise IOError(
                f"IMG too small: {actual_bytes} bytes < {expected_bytes} bytes for tile '{tile_id}'."
            )

        arr = np.memmap(
            img_path,
            dtype=dtype,
            mode="r",
            shape=(lbl.lines, lbl.line_samples),
        )

        return RasterTile(tile_id=tile_id, data=arr, lbl=lbl)

    # 
    # ROI + standardisation
    # 

    def _extract_roi(self, raster: np.ndarray, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= raster.shape[0] and 0 <= c0 < c1 <= raster.shape[1]):
            raise ValidationError(f"ROI {roi} out of bounds for raster shape {raster.shape}")
        return np.array(raster[r0:r1, c0:c1], dtype=np.float32, copy=True)

    def _standardise_to_meters(self, patch: np.ndarray, lbl: LBLMeta) -> np.ndarray:
        """
        Endymion convention:
          - float32
          - metres
          - nodata (if known) -> NaN
        """
        out = patch.astype(np.float32, copy=False)

        if lbl.missing_constant is not None:
            out[out == float(lbl.missing_constant)] = np.nan

        if lbl.scaling_factor is not None:
            out = (out * float(lbl.scaling_factor)).astype(np.float32, copy=False)

        if (lbl.unit or "").upper() in {"KILOMETER", "KILOMETRE", "KM"}:
            out = (out * 1000.0).astype(np.float32, copy=False)

        return out

    # 
    # Validation / sanity checks
    # 

    def _validate_patch(self, patch_m: np.ndarray, roi: ROI, meta: RasterMeta) -> None:
        r0, r1, c0, c1 = roi
        expected_shape = (r1 - r0, c1 - c0)
        if patch_m.shape != expected_shape:
            raise ValidationError(f"Patch shape {patch_m.shape} != expected {expected_shape} for roi={roi}")

        nan_ratio = float(np.isnan(patch_m).mean())
        if nan_ratio > 0.95:
            raise ValidationError(f"Patch NaN ratio too high: {nan_ratio:.3f}")

        vmin = float(np.nanmin(patch_m))
        vmax = float(np.nanmax(patch_m))
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValidationError("Patch has no finite values after standardisation.")

        meta.extra.update({
            "nan_ratio": nan_ratio,
            "min_m": vmin,
            "max_m": vmax,
        })

    # 
    # Derived products storage
    # 

    def _roi_key(self, roi: ROI) -> str:
        r0, r1, c0, c1 = roi
        return f"roi_{r0}_{r1}_{c0}_{c1}"

    def derived_dir(self, tile_id: str, roi: ROI) -> Path:
        """
        Base folder for any derived artifacts for this (tile_id, roi).
        """
        return self.persistent_dir / "derived" / tile_id / self._roi_key(roi)

    def save_derived(
        self,
        tile_id: str,
        roi: ROI,
        dem_m: np.ndarray,
        features: Dict[str, np.ndarray],
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save core derived rasters (DEM + slope/roughness) and a small meta.json.
        """
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
            "products": [p for p in ["dem_m", "slope_deg", "roughness_rms"]
                         if (out_dir / f"{p}.npy").exists()],
            "extra": meta_extra or {},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return out_dir

    def load_derived(self, tile_id: str, roi: ROI) -> Optional[Dict[str, Any]]:
        """
        Load derived rasters if they exist. Returns None if meta.json not found.
        """
        out_dir = self.derived_dir(tile_id, roi)
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        out: Dict[str, Any] = {"meta": meta}

        dem_path = out_dir / "dem_m.npy"
        if dem_path.exists():
            out["dem_m"] = np.load(dem_path)

        slope_path = out_dir / "slope_deg.npy"
        if slope_path.exists():
            out["slope_deg"] = np.load(slope_path)

        rough_path = out_dir / "roughness_rms.npy"
        if rough_path.exists():
            out["roughness_rms"] = np.load(rough_path)

        return out

    # 
    # Navigation runs storage (NEW)
    # 

    def save_navigation(
        self,
        tile_id: str,
        roi: ROI,
        hazard: np.ndarray,
        cost: np.ndarray,
        path_rc: np.ndarray,
        nav_meta: Dict[str, Any],
        run_name: str = "navigation_v1",
        save_json_path: bool = True,
    ) -> Path:
        """
        Save navigation artifacts next to derived rasters.

        Layout:
          derived/<tile_id>/<roi_key>/<run_name>/
            hazard.npy
            cost.npy
            path_rc.npy
            path_rc.json (optional)
            nav_meta.json
        """
        out_dir = self.derived_dir(tile_id, roi) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure stable dtypes for reproducibility + small disk size
        np.save(out_dir / "hazard.npy", hazard.astype(np.float32, copy=False))
        np.save(out_dir / "cost.npy", cost.astype(np.float32, copy=False))
        path_arr = np.asarray(path_rc, dtype=np.int32)
        np.save(out_dir / "path_rc.npy", path_arr)

        if save_json_path:
            (out_dir / "path_rc.json").write_text(json.dumps(path_arr.tolist(), indent=2))

        meta_out = {
            "tile_id": tile_id,
            "roi": list(roi),
            "shape": list(hazard.shape),
            "run_name": run_name,
            "products": ["hazard", "cost", "path_rc"],
            "nav_meta": nav_meta,
        }
        (out_dir / "nav_meta.json").write_text(json.dumps(meta_out, indent=2))
        return out_dir

    def load_navigation(
        self,
        tile_id: str,
        roi: ROI,
        run_name: str = "navigation_v1",
    ) -> Optional[Dict[str, Any]]:
        """
        Load navigation artifacts saved by save_navigation().
        Returns None if nav_meta.json doesn't exist.
        """
        out_dir = self.derived_dir(tile_id, roi) / run_name
        meta_path = out_dir / "nav_meta.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        out: Dict[str, Any] = {"meta": meta}

        hz = out_dir / "hazard.npy"
        cs = out_dir / "cost.npy"
        pr = out_dir / "path_rc.npy"

        if hz.exists():
            out["hazard"] = np.load(hz)
        if cs.exists():
            out["cost"] = np.load(cs)
        if pr.exists():
            out["path_rc"] = np.load(pr)

        return out

    def list_navigation_runs(self, tile_id: str, roi: ROI) -> List[str]:
        """
        Convenience: list run folders under derived/<tile>/<roi_key>/ that contain nav_meta.json.
        """
        base = self.derived_dir(tile_id, roi)
        if not base.exists():
            return []

        runs: List[str] = []
        for p in base.iterdir():
            if p.is_dir() and (p / "nav_meta.json").exists():
                runs.append(p.name)
        runs.sort()
        return runs


# Example canonical ROI for testing
CANONICAL_ROI: ROI = (7072, 8096, 7072, 8096)