from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np


# ============================================================
# 1) Types / Specs
# ============================================================

ROI = Tuple[int, int, int, int]  # (r0, r1, c0, c1) pixel window


@dataclass(frozen=True)
class LOLATileSpec:
    tile_id: str
    img_filename: str                 # e.g. "ldem_85n_20m_float.img"
    lbl_filename: Optional[str] = None  # e.g. "ldem_85n_20m_float.lbl"


@dataclass
class RasterMeta:
    """Minimal metadata the rest of the pipeline can rely on."""
    tile_id: str
    shape: Tuple[int, int]            # (lines, line_samples)
    dtype: str                        # numpy dtype str (e.g., '<f4')
    nodata: Optional[float] = None
    units: str = "m"                  # standard output should be metres
    scaling_factor: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterTile:
    """Memmap-backed full tile (do not materialise unless needed)."""
    tile_id: str
    data: np.ndarray                  # typically np.memmap
    meta: RasterMeta


@dataclass
class RasterPatch:
    """What DataHandler returns to the rest of Endymion."""
    data: np.ndarray                  # 2D float array (metres, NaNs for nodata)
    meta: RasterMeta
    roi: ROI


# ============================================================
# 2) Exceptions
# ============================================================

class DataHandlerError(Exception):
    pass


class TileNotFoundError(DataHandlerError):
    pass


class ValidationError(DataHandlerError):
    pass


# ============================================================
# 3) DataHandler
# ============================================================

class DataHandler:
    """
    Endymion DataHandler responsibilities:
      - ensure .img/.lbl present locally (optional remote download)
      - parse label (minimal required keys)
      - memmap open raster
      - extract ROI patches
      - standardise output (float32, metres, NaNs for nodata)
      - sanity checks

    It MUST NOT compute derived features (slope/roughness/etc).
    """

    def __init__(
        self,
        base_url: str,
        tiles: list[LOLATileSpec],
        cache_dir: str | Path = "./data_cache/lola",
        force_download: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.tiles = {t.tile_id: t for t in tiles}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_download = force_download

    # ----------------------------
    # Public API
    # ----------------------------

    def get_patch(self, tile_id: str, roi: ROI, verbose: bool = False) -> RasterPatch:
        """
        Main entry-point used by System_Controller.
        ROI-first: only materialises the ROI region (fast + memory safe).
        """
        tile = self.load_tile(tile_id, force_download=self.force_download, verbose=verbose)

        patch = self._read_roi(tile, roi)          # materialise ROI only
        patch = self._standardise_patch(patch, tile.meta)
        self._validate_patch(patch, roi, tile.meta)

        return RasterPatch(data=patch, meta=tile.meta, roi=roi)

    # ----------------------------
    # Tile resolution / local cache
    # ----------------------------

    def _get_spec(self, tile_id: str) -> LOLATileSpec:
        try:
            return self.tiles[tile_id]
        except KeyError:
            raise TileNotFoundError(
                f"Unknown tile_id '{tile_id}'. Available: {list(self.tiles.keys())}"
            )

    def ensure_downloaded(self, tile_id: str, force: bool = False) -> tuple[Path, Optional[Path]]:
        """
        Ensures IMG (and LBL if provided) exist locally.
        Download hook is intentionally a stub for now.
        """
        spec = self._get_spec(tile_id)
        img_path = self.cache_dir / spec.img_filename
        lbl_path = (self.cache_dir / spec.lbl_filename) if spec.lbl_filename else None

        if (img_path.exists() and (lbl_path is None or lbl_path.exists())) and not force:
            return img_path, lbl_path

        # TODO: Implement download when you're ready:
        # if not img_path.exists() or force:
        #     self._download_file(f"{self.base_url}/{spec.img_filename}", img_path)
        # if lbl_path and (not lbl_path.exists() or force):
        #     self._download_file(f"{self.base_url}/{spec.lbl_filename}", lbl_path)

        # Hard fail if still missing
        missing = []
        if not img_path.exists():
            missing.append(str(img_path))
        if lbl_path is not None and not lbl_path.exists():
            missing.append(str(lbl_path))

        if missing:
            raise TileNotFoundError(
                "Missing required tile files locally:\n"
                + "\n".join(f" - {m}" for m in missing)
                + "\nDownload not implemented; place files there or implement _download_file()."
            )

        return img_path, lbl_path

    # ----------------------------
    # Label parsing (minimal)
    # ----------------------------

    def parse_lbl(self, lbl_path: Path) -> dict[str, Any]:
        """
        Minimal PDS3-ish parser for the handful of keys we need.
        Assumes 'KEY = VALUE' per line. Good enough for LOLA LBLs in this project scope.
        """
        text = lbl_path.read_text(errors="ignore").splitlines()
        out: dict[str, Any] = {}

        def _clean(v: str) -> str:
            return v.strip().strip('"').strip("'")

        for line in text:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            # drop inline comments if present
            if "/*" in v:
                v = v.split("/*", 1)[0].strip()
            out[k] = _clean(v)

        return out

    # ----------------------------
    # Reading LOLA float_img
    # ----------------------------

    def _numpy_dtype_from_sample_type(self, sample_type: str, sample_bits: int) -> np.dtype:
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

    def load_tile(self, tile_id: str, force_download: bool = False, verbose: bool = False) -> RasterTile:
        """
        Ensures files are present, parses label, and returns a memmap-backed RasterTile.
        """
        img_path, lbl_path = self.ensure_downloaded(tile_id, force=force_download)

        if lbl_path is None:
            raise ValueError(f"No LBL provided for '{tile_id}'. Add lbl_filename in LOLATileSpec.")

        lbl = self.parse_lbl(lbl_path)

        # Required
        lines = int(lbl.get("LINES"))
        line_samples = int(lbl.get("LINE_SAMPLES"))
        sample_type = lbl.get("SAMPLE_TYPE")
        sample_bits = int(lbl.get("SAMPLE_BITS"))

        # Optional
        missing_constant = lbl.get("MISSING_CONSTANT")
        scaling_factor = lbl.get("SCALING_FACTOR")

        nodata = float(missing_constant) if missing_constant is not None else None
        scale = float(scaling_factor) if scaling_factor is not None else None

        dtype = self._numpy_dtype_from_sample_type(sample_type, sample_bits)

        expected_bytes = lines * line_samples * (sample_bits // 8)
        actual_bytes = img_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise IOError(
                f"IMG too small: {actual_bytes} < {expected_bytes} bytes for tile '{tile_id}'."
            )

        if verbose:
            print(f"[{tile_id}] shape=({lines},{line_samples}) dtype={dtype} nodata={nodata} scale={scale}")

        arr = np.memmap(img_path, dtype=dtype, mode="r", shape=(lines, line_samples))

        meta = RasterMeta(
            tile_id=tile_id,
            shape=(lines, line_samples),
            dtype=str(dtype),
            nodata=nodata,
            units="m",                 # we standardise to metres
            scaling_factor=scale,
            extra={"sample_type": sample_type, "sample_bits": sample_bits},
        )

        return RasterTile(tile_id=tile_id, data=arr, meta=meta)

    # ----------------------------
    # ROI extraction (ROI-first)
    # ----------------------------

    def _read_roi(self, tile: RasterTile, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        H, W = tile.meta.shape
        if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W):
            raise ValidationError(f"ROI {roi} out of bounds for raster shape {(H, W)}")

        # Materialise ROI only (keeps memmap benefit)
        roi_arr = np.asarray(tile.data[r0:r1, c0:c1], dtype=np.float32)
        return roi_arr

    def _standardise_patch(self, patch: np.ndarray, meta: RasterMeta) -> np.ndarray:
        """
        Endymion output convention:
          - float32
          - metres
          - nodata -> np.nan
          - apply scaling_factor if present
        """
        patch = np.asarray(patch, dtype=np.float32)

        if meta.nodata is not None:
            patch[patch == meta.nodata] = np.nan

        if meta.scaling_factor is not None:
            patch = patch * float(meta.scaling_factor)

        # If you ever discover a tile is in km, handle it ONCE here.
        # For LOLA elevation tiles you normally want metres.
        if meta.units.lower() in ("km", "kilometres", "kilometers"):
            patch = patch * 1000.0
            meta.units = "m"

        return patch

    # ----------------------------
    # Validation / sanity checks
    # ----------------------------

    def _validate_patch(self, patch: np.ndarray, roi: ROI, meta: RasterMeta) -> None:
        r0, r1, c0, c1 = roi
        expected = (r1 - r0, c1 - c0)
        if patch.shape != expected:
            raise ValidationError(f"Patch shape {patch.shape} != expected {expected} for roi={roi}")

        nan_ratio = float(np.isnan(patch).mean())
        meta.extra.update({"nan_ratio": nan_ratio})

        if nan_ratio > 0.95:
            raise ValidationError(f"Patch NaN ratio too high: {nan_ratio:.3f}")

        finite = patch[np.isfinite(patch)]
        if finite.size == 0:
            raise ValidationError("Patch has no finite values after standardisation.")

        meta.extra.update({
            "min_m": float(np.min(finite)),
            "max_m": float(np.max(finite)),
        })


# ============================================================
# 4) Example ROI ( canonical prototype)
# ============================================================

CANONICAL_ROI: ROI = (7072, 8096, 7072, 8096)
