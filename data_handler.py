from __future__ import annotations

import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np

# ----------------------------
# Types
# ----------------------------

ROI = Tuple[int, int, int, int]  # (row0, row1, col0, col1)


@dataclass(frozen=True)
class LOLATileSpec:
    """
    Prototype-friendly spec: tile id + remote URLs.
    tile_id example: "ldem_85s_20m_float"
    """
    tile_id: str
    img_url: str
    lbl_url: str


@dataclass
class PDSImageMeta:
    """
    Minimal metadata parsed from a PDS3 .LBL label file.
    Tailored for LOLA polar float_img tiles.
    """
    tile_id: str

    lines: int
    line_samples: int
    sample_type: str
    sample_bits: int

    unit: Optional[str] = None           # e.g. "KILOMETER"
    scaling_factor: Optional[float] = 1.0
    offset: Optional[float] = None       # reference radius offset (km) if needed later
    missing_constant: Optional[float] = None

    raw: Optional[Dict[str, Any]] = None


@dataclass
class RasterTile:
    tile_id: str
    data: np.ndarray  # np.memmap or ndarray
    meta: PDSImageMeta


@dataclass
class RasterPatch:
    """
    What the prototype pipeline should pass into FeatureExtractor/HazardAssessor.
    Always DEM heights in metres with NaNs for nodata.
    """
    tile_id: str
    roi: ROI
    dem_m: np.ndarray          # float32, metres
    meta: PDSImageMeta


# ----------------------------
# Exceptions
# ----------------------------

class DataHandlerError(Exception):
    pass


class TileNotFoundError(DataHandlerError):
    pass


class LabelParseError(DataHandlerError):
    pass


class ValidationError(DataHandlerError):
    pass


# ----------------------------
# DataHandler
# ----------------------------

class DataHandler:
    """
    Endymion DataHandler (prototype-ready):

    - download/cache PDS .IMG/.LBL
    - parse LBL (minimal keys)
    - memmap the IMG
    - read ROI (materialise only ROI into RAM)
    - standardise to metres (km -> m) and apply missing_constant -> NaN
    - basic sanity checks

    NO derived features here (slope/roughness belong in FeatureExtractor).
    """

    _kv_re = re.compile(r"^\s*([A-Z0-9_\-^]+)\s*=\s*(.+?)\s*$")

    def __init__(
        self,
        cache_dir: str | Path = "/content/endymion_cache/lola",
        user_agent: str = "Endymion-DataHandler/0.3",
        timeout_sec: int = 60,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.tiles: Dict[str, LOLATileSpec] = {}

    # -------- Registration --------

    def register_tile(self, spec: LOLATileSpec) -> None:
        self.tiles[spec.tile_id] = spec

    def register_tiles(self, specs: List[LOLATileSpec]) -> None:
        for s in specs:
            self.register_tile(s)

    # -------- Public API --------

    def get_patch(
        self,
        tile_id: str,
        roi: ROI,
        *,
        force_download: bool = False,
        representation: Literal["height"] = "height",
        verbose: bool = False,
    ) -> RasterPatch:
        """
        Main entry point for the prototype pipeline.

        representation currently fixed to "height" for Endymion’s terrain features.
        (Keep "radius" for later if you truly need absolute radius; not required for slope.)
        """
        if representation != "height":
            raise ValueError("Prototype v1 supports representation='height' only.")

        tile = self.load_tile(tile_id, force_download=force_download, verbose=verbose)
        roi_km = self.read_roi_km(tile, roi)
        dem_m = self.standardise_height_to_m(tile.meta, roi_km)

        self._validate_patch(dem_m, roi, tile.meta)
        return RasterPatch(tile_id=tile_id, roi=roi, dem_m=dem_m, meta=tile.meta)

    def load_tile(self, tile_id: str, *, force_download: bool = False, verbose: bool = False) -> RasterTile:
        img_path, lbl_path = self.ensure_downloaded(tile_id, force=force_download)
        meta = self.parse_lbl(tile_id, lbl_path)

        dtype = self._numpy_dtype_from_sample_type(meta.sample_type, meta.sample_bits)

        expected_bytes = meta.lines * meta.line_samples * (meta.sample_bits // 8)
        actual_bytes = img_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise IOError(
                f"IMG too small: {actual_bytes} bytes < {expected_bytes} bytes for '{tile_id}'."
            )

        if verbose:
            print("Tile:", tile_id, "shape:", (meta.lines, meta.line_samples), "dtype:", dtype)

        arr = np.memmap(img_path, dtype=dtype, mode="r", shape=(meta.lines, meta.line_samples))
        return RasterTile(tile_id=tile_id, data=arr, meta=meta)

    # -------- Download/cache --------

    def ensure_downloaded(self, tile_id: str, *, force: bool = False) -> Tuple[Path, Path]:
        spec = self.tiles.get(tile_id)
        if spec is None:
            raise TileNotFoundError(f"Tile '{tile_id}' not registered.")

        img_path = self.cache_dir / f"{tile_id}.img"
        lbl_path = self.cache_dir / f"{tile_id}.lbl"

        if force or not lbl_path.exists():
            self._download_file(spec.lbl_url, lbl_path)

        if force or not img_path.exists():
            self._download_file(spec.img_url, img_path)

        if not img_path.exists() or not lbl_path.exists():
            raise TileNotFoundError(f"Missing tile files for '{tile_id}' in {self.cache_dir}")

        return img_path, lbl_path

    def _download_file(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent}, method="GET")
        tmp = out_path.with_suffix(out_path.suffix + ".part")

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)  # 1MB chunks

        tmp.replace(out_path)

    # -------- Label parsing --------

    def parse_lbl(self, tile_id: str, lbl_path: str | Path) -> PDSImageMeta:
        """
        Minimal PDS3-ish parser. Supports nested OBJECT blocks by namespacing keys.
        """
        lbl_path = Path(lbl_path)
        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        raw: Dict[str, Any] = {}
        stack: List[str] = []

        def set_key(k: str, v: str) -> None:
            if stack:
                raw[".".join(stack + [k])] = v
            else:
                raw[k] = v

        for line in lines:
            s = line.strip()
            if not s or s.startswith("/*"):
                continue

            if s.startswith("OBJECT"):
                m = self._kv_re.match(s)
                if m:
                    obj_name = m.group(2).strip().strip('"')
                    stack.append(obj_name)
                continue

            if s.startswith("END_OBJECT"):
                if stack:
                    stack.pop()
                continue

            m = self._kv_re.match(s)
            if m:
                k, v = m.group(1), m.group(2)
                set_key(k, v)

        def pick_any(*keys: str) -> Optional[str]:
            # exact
            for k in keys:
                if k in raw:
                    return str(raw[k])
            # suffix fallback
            for k in keys:
                for rk, rv in raw.items():
                    if rk.endswith("." + k):
                        return str(rv)
            return None

        def to_int(x: Optional[str]) -> Optional[int]:
            if x is None:
                return None
            return int(str(x).strip().strip('"'))

        def to_float(x: Optional[str]) -> Optional[float]:
            if x is None:
                return None
            s = str(x).strip().strip('"')
            return float(s)

        def to_str(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            return str(x).strip().strip('"')

        lines_i = to_int(pick_any("LINES"))
        samples_i = to_int(pick_any("LINE_SAMPLES"))
        st = to_str(pick_any("SAMPLE_TYPE"))
        sb = to_int(pick_any("SAMPLE_BITS"))

        if lines_i is None or samples_i is None or st is None or sb is None:
            raise LabelParseError(
                f"LBL missing required keys for '{tile_id}'. "
                f"Got LINES={lines_i}, LINE_SAMPLES={samples_i}, SAMPLE_TYPE={st}, SAMPLE_BITS={sb}"
            )

        meta = PDSImageMeta(
            tile_id=tile_id,
            lines=lines_i,
            line_samples=samples_i,
            sample_type=st,
            sample_bits=sb,
            unit=to_str(pick_any("UNIT")),
            scaling_factor=to_float(pick_any("SCALING_FACTOR")) or 1.0,
            offset=to_float(pick_any("OFFSET")),
            missing_constant=to_float(pick_any("MISSING_CONSTANT")),
            raw=raw,
        )
        return meta

    # -------- IMG interpretation --------

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

        raise ValueError(f"Unsupported SAMPLE_BITS for float: {sample_bits}")

    # -------- ROI + standardisation --------

    def read_roi_km(self, tile: RasterTile, roi: ROI) -> np.ndarray:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= tile.meta.lines and 0 <= c0 < c1 <= tile.meta.line_samples):
            raise ValidationError(f"ROI {roi} out of bounds for tile shape {(tile.meta.lines, tile.meta.line_samples)}")

        # Materialise ROI only (keeps RAM safe)
        return np.array(tile.data[r0:r1, c0:c1], dtype=np.float32, copy=True)

    def standardise_height_to_m(self, meta: PDSImageMeta, height_native: np.ndarray) -> np.ndarray:
        """
        LOLA polar float_img uses UNIT=KILOMETER in the label (your snippet).
        We standardise to metres for the rest of Endymion.
        """
        z = height_native.astype(np.float32, copy=False)

        # missing constant -> NaN (only if present)
        if meta.missing_constant is not None:
            z = z.copy()
            z[z == meta.missing_constant] = np.nan

        # scaling factor (usually 1)
        if meta.scaling_factor is not None and meta.scaling_factor != 1:
            z = z * float(meta.scaling_factor)

        unit = (meta.unit or "").strip().upper()
        if "KILOMETER" in unit or unit == "KM":
            z = z * 1000.0
        elif unit in ("METER", "METRE", "METERS", "METRES", "M"):
            pass
        elif unit == "":
            # if label is missing unit, assume km for LOLA float_img (but warn via meta.raw if needed)
            z = z * 1000.0
        else:
            raise ValueError(f"Unrecognised unit in label: {meta.unit}")

        return z.astype(np.float32, copy=False)

    # -------- Validation --------

    def _validate_patch(self, dem_m: np.ndarray, roi: ROI, meta: PDSImageMeta) -> None:
        r0, r1, c0, c1 = roi
        expected = (r1 - r0, c1 - c0)

        if dem_m.shape != expected:
            raise ValidationError(f"Patch shape {dem_m.shape} != expected {expected} for roi={roi}")

        nan_ratio = float(np.isnan(dem_m).mean())
        if nan_ratio > 0.95:
            raise ValidationError(f"Patch NaN ratio too high: {nan_ratio:.3f}")

        mn = float(np.nanmin(dem_m))
        mx = float(np.nanmax(dem_m))
        if not np.isfinite(mn) or not np.isfinite(mx):
            raise ValidationError("Patch has no finite values after standardisation.")
