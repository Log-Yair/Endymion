from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import shutil
import request

# ============================================================
# Types
# ============================================================

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
    unit: Optional[str] = None            # e.g. "KILOMETER"
    scaling_factor: Optional[float] = None
    offset: Optional[float] = None        # present in your LBL (1737.4) - not used now
    missing_constant: Optional[float] = None  # not present in your snippet, but keep slot
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterMeta:
    """
    What the rest of Endymion can rely on.
    """
    tile_id: str
    shape: Tuple[int, int]
    dtype: str
    units: str = "m"                      # Endymion convention: metres
    nodata: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterPatch:
    data: np.ndarray                      # 2D float32 with NaNs for nodata
    meta: RasterMeta
    roi: ROI


@dataclass
class RasterTile:
    """
    Internal convenience type (full tile, usually memmap-backed).
    """
    tile_id: str
    data: np.ndarray                      # memmap or ndarray
    lbl: LBLMeta


# ============================================================
# Exceptions
# ============================================================

class DataHandlerError(Exception):
    pass


class TileNotFoundError(DataHandlerError):
    pass


class LabelParseError(DataHandlerError):
    pass


class ValidationError(DataHandlerError):
    pass


# ============================================================
# DataHandler
# ============================================================

class DataHandler:
  """
    Endymion DataHandler:
      - resolves tiles (local cache; download hook optional)
      - parses .LBL
      - loads .IMG as memmap
      - extracts ROI
      - standardises to metres + float32 + NaNs
      - sanity checks
    """
    def __init__(
        self,
        base_url: str,
        tiles: list[LOLATileSpec],
        cache_dir: str | Path = "./data_cache/lola",   # keep for backwards compat
        persistent_dir: str | Path | None = None,      # NEW: Drive cache
        runtime_dir: str | Path | None = None,         # NEW: local cache
        allow_download: bool = True,                   # NEW
        force_download: bool = False,
        timeout_s: int = 60,                           # NEW
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

        # 3) need to download
        if not self.allow_download:
            # Keep your existing error style, but explain both locations
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        # copy if missing or size mismatch
        if (not dst.exists()) or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)

    def _download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If already present and not forcing, skip
        if out_path.exists() and not self.force_download:
            return

        tmp = out_path.with_suffix(out_path.suffix + ".part")

        with requests.get(url, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        tmp.replace(out_path)

    # ===========================
    # Public API
    # ==========================

    def get_patch(self, tile_id: str, roi: ROI, verbose: bool = False) -> RasterPatch:
        """
        Main entry point used by your prototype runner / pipeline (not System_Controller).
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

    # ----------------------------
    # Tile resolution / local cache
    # ----------------------------

    def _get_tile(self, tile_id: str) -> LOLATileSpec:
        spec = self.tiles.get(tile_id)
        if spec is None:
            raise TileNotFoundError(
                f"Unknown tile_id '{tile_id}'. Available: {list(self.tiles.keys())}"
            )
        return spec

    def _ensure_local(self, spec: LOLATileSpec) -> Tuple[Path, Path]:
        """
        Ensures (IMG, LBL) exist locally.
        Download hook can be added later without touching the rest.
        """
        img_path = self.cache_dir / spec.img_filename
        lbl_path = self.cache_dir / spec.lbl_filename

        if (img_path.exists() and lbl_path.exists()) and not self.force_download:
            return img_path, lbl_path

        # TODO: optional download implementation later
        # self._download_file(f"{self.base_url}/{spec.img_filename}", img_path)
        # self._download_file(f"{self.base_url}/{spec.lbl_filename}", lbl_path)

        if not img_path.exists():
            raise TileNotFoundError(f"Missing IMG file: {img_path}")
        if not lbl_path.exists():
            raise TileNotFoundError(f"Missing LBL file: {lbl_path}")

        return img_path, lbl_path

    # ----------------------------
    # Label parsing (LBL)
    # ----------------------------

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
            if not line or line.startswith("/*"):
                continue
            if "=" not in line:
                continue

            key, val = [x.strip() for x in line.split("=", 1)]
            key_u = key.upper()

            # Keep common scalar keys; ignore nested blocks safely
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

        # Required fields
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

    # ----------------------------
    # IMG loading (float_img)
    # ----------------------------

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

    # ----------------------------
    # ROI + standardisation
    # ----------------------------

    def _extract_roi(self, raster: np.ndarray, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= raster.shape[0] and 0 <= c0 < c1 <= raster.shape[1]):
            raise ValidationError(f"ROI {roi} out of bounds for raster shape {raster.shape}")

        # Copy so the patch is independent of the full tile memmap
        return np.array(raster[r0:r1, c0:c1], dtype=np.float32, copy=True)

    def _standardise_to_meters(self, patch: np.ndarray, lbl: LBLMeta) -> np.ndarray:
        """
        Endymion convention:
          - float32
          - metres
          - nodata (if known) -> NaN
        """
        out = patch.astype(np.float32, copy=False)

        # Apply missing constant -> NaN if label provides it
        if lbl.missing_constant is not None:
            out[out == float(lbl.missing_constant)] = np.nan

        # Apply scaling if present (your LBL has SCALING_FACTOR=1)
        if lbl.scaling_factor is not None:
            out = (out * float(lbl.scaling_factor)).astype(np.float32, copy=False)

        # Convert unit to metres (your LBL: UNIT = KILOMETER)
        if (lbl.unit or "").upper() in {"KILOMETER", "KILOMETRE", "KM"}:
            out = (out * 1000.0).astype(np.float32, copy=False)

        return out

    # ----------------------------
    # Validation / sanity checks
    # ----------------------------

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


# Example canonical ROI you gave
CANONICAL_ROI: ROI = (7072, 8096, 7072, 8096)
