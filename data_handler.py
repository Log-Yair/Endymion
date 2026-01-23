from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np


# --------------------------------------------------------
# 1) Types / Specs
# --------------------------------------------------------

ROI = Tuple[int, int, int, int]  # (r0, r1, c0, c1) pixel window


@dataclass(frozen=True)
class LOLATileSpec:
    tile_id: str                  # e.g. "ldem_85n_20m"
    filename: str                 # e.g. "ldem_85n_20m_float.img"
    label_filename: Optional[str] = None  # e.g. ".lbl" if needed


@dataclass
class RasterMeta:
    """Minimal metadata the rest of the pipeline can rely on."""
    tile_id: str
    shape: Tuple[int, int]
    dtype: str
    nodata: Optional[float] = None
    units: str = "m"              # keep consistent (recommend metres)
    extra: Dict[str, Any] = None


@dataclass
class RasterPatch:
    """What DataHandler returns."""
    data: np.ndarray              # 2D float array (metres, NaNs for nodata)
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
    Endymion DataHandler (loyal to the architecture):
      - fetch/open raw LOLA tiles (local cache + optional remote)
      - extract ROI patches
      - standardise output array (float32/float64, metres, NaNs for nodata)
      - run sanity checks

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

    def get_patch(self, tile_id: str, roi: ROI) -> RasterPatch:
        """
        Main entry-point used by System_Controller.
        """
        tile = self._get_tile(tile_id)
        img_path = self._ensure_local(tile)

        # 1) Read raw raster (full or memory-mapped)
        raw, meta = self._read_lola_float_img(img_path, tile)

        # 2) Extract ROI
        patch = self._extract_roi(raw, roi)

        # 3) Standardise dtype + nodata => NaN
        patch = self._standardise(patch, meta)

        # 4) Validate / sanity check
        self._validate_patch(patch, roi, meta)

        return RasterPatch(
            data=patch,
            meta=meta,
            roi=roi,
        )

    # ----------------------------
    # Tile resolution / I/O
    # ----------------------------

    def _get_tile(self, tile_id: str) -> LOLATileSpec:
        if tile_id not in self.tiles:
            raise TileNotFoundError(f"Unknown tile_id '{tile_id}'. Available: {list(self.tiles.keys())}")
        return self.tiles[tile_id]

    def _ensure_local(self, tile: LOLATileSpec) -> Path:
        """
        Ensure the tile .img exists locally. If not, download it.
        (Download impl can be added without changing the rest of the code.)
        """
        local = self.cache_dir / tile.filename
        if local.exists() and not self.force_download:
            return local

        # TODO: implement download if you're pulling from PDS
        # self._download_file(remote_url=f"{self.base_url}/{tile.filename}", local_path=local)

        if not local.exists():
            raise TileNotFoundError(
                f"Tile file not found locally: {local}\n"
                f"Download not implemented / not run. Either place the file there or implement _download_file()."
            )
        return local

    # ----------------------------
    # Reading LOLA float_img
    # ----------------------------

    def _read_lola_float_img(self, img_path: Path, tile: LOLATileSpec) -> tuple[np.ndarray, RasterMeta]:
        """
        Read LOLA .img (float grid).
        NOTE: This is where your current notebook code should live.
        Keep it isolated here so the rest stays clean.
        """
        # TODO: replace with your real reader:
        # - determine dtype (float32)
        # - determine shape (from label or known convention)
        # - nodata sentinel (if present)
        #
        # For now: raise clearly
        raise NotImplementedError("Implement LOLA float_img reader here (memory-map recommended).")

    # ----------------------------
    # ROI + Standardisation
    # ----------------------------

    def _extract_roi(self, raster: np.ndarray, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= raster.shape[0] and 0 <= c0 < c1 <= raster.shape[1]):
            raise ValidationError(f"ROI {roi} out of bounds for raster shape {raster.shape}")
        return raster[r0:r1, c0:c1].copy()  # copy keeps patch independent

    def _standardise(self, patch: np.ndarray, meta: RasterMeta) -> np.ndarray:
        """
        Output convention for Endymion:
          - float32 or float64
          - metres
          - nodata -> np.nan
        """
        if not np.issubdtype(patch.dtype, np.floating):
            patch = patch.astype(np.float32)

        if meta.nodata is not None:
            patch = np.where(patch == meta.nodata, np.nan, patch)

        # If units are km in raw, convert to metres here ONCE and set meta.units="m"
        if meta.units.lower() in ("km", "kilometres", "kilometers"):
            patch = patch * 1000.0
            meta.units = "m"

        return patch

    # ----------------------------
    # Validation / sanity checks
    # ----------------------------

    def _validate_patch(self, patch: np.ndarray, roi: ROI, meta: RasterMeta) -> None:
        r0, r1, c0, c1 = roi
        expected_shape = (r1 - r0, c1 - c0)

        if patch.shape != expected_shape:
            raise ValidationError(f"Patch shape {patch.shape} != expected {expected_shape} for roi={roi}")

        nan_ratio = float(np.isnan(patch).mean())
        if nan_ratio > 0.95:
            raise ValidationError(f"Patch NaN ratio too high: {nan_ratio:.3f}")

        # Optional: log-only stats (don’t raise unless obviously broken)
        _min = np.nanmin(patch)
        _max = np.nanmax(patch)
        if not np.isfinite(_min) or not np.isfinite(_max):
            raise ValidationError("Patch has no finite values after standardisation.")

        # You can store quick stats in meta.extra if you want
        if meta.extra is None:
            meta.extra = {}
        meta.extra.update({
            "nan_ratio": nan_ratio,
            "min_m": float(_min),
            "max_m": float(_max),
        })


# ============================================================
# 4) Example: your canonical ROI prototype
# ============================================================

CANONICAL_ROI: ROI = (7072, 8096, 7072, 8096)  # (1024, 1024)
