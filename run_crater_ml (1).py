"""
- This file is intentionally aligned to the ML section already used in the Endymion experimental notebook, not to a more elaborate standalone ML pipeline.

- Main source of structure / logic:
    Endymion_Exp_pipeline.ipynb
      1) MLDatasetBuilder.build_from_cache(...)
      2) block-based spatial split using block_id
      3) RandomForestClassifier
      4) metrics + feature importances
      5) save crater_proba_ml_v1.npy

- Main ideas reused from the project files and notes:
    DataHandler -> resolve canonical ROI / derived dir
    MLDatasetBuilder -> build per-pixel dataset from cached rasters
    benchmark_runner.py -> consume crater_proba_ml_v1.npy later

- Design goal for this file:
  keep one clean, official, reproducible pipeline runner separate from the benchmark runner and separate from the presentation notebook.

Visual flow:

  ROI derived cache
    dem_m.npy
    slope_deg.npy
    roughness_rms.npy
    crater_mask.npy
         |
         v
  MLDatasetBuilder
         |
         v
  DataFrame with:
    row, col, block_id,
    elevation_m, slope_deg, roughness_rms,
    label_crater
         |
         v
  simple block split (train/test)
         |
         v
  Random Forest baseline
         |
         +--> held-out metrics
         +--> feature importances
         +--> crater_proba_ml_v1.npy
"""


# IMPORTS
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import joblib
except ImportError:  
    joblib = None


# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
HERE = THIS_FILE.parent
PARENT = HERE.parent

for p in (HERE, PARENT, HERE / "src", PARENT / "src"):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)


# -----------------------------------------------------------------------------
# Import strategy
# -----------------------------------------------------------------------------
# Supports either cleaned package imports or a flatter local layout.
def _install_flat_src_aliases() -> None:
    src_pkg = sys.modules.setdefault("src", types.ModuleType("src"))

    for pkg_name in ["data", "models"]:
        full_name = f"src.{pkg_name}"
        pkg = sys.modules.get(full_name)
        if pkg is None:
            pkg = types.ModuleType(full_name)
            sys.modules[full_name] = pkg
        setattr(src_pkg, pkg_name, pkg)

    alias_map = {
        "src.data.data_handler": "data_handler",
        "src.models.ml_dataset_builder": "ml_dataset_builder",
    }

    for alias, flat_name in alias_map.items():
        if alias in sys.modules:
            continue

        module = importlib.import_module(flat_name)
        sys.modules[alias] = module

        parent_name, attr_name = alias.rsplit(".", 1)
        parent = sys.modules[parent_name]
        setattr(parent, attr_name, module)


try:
    from src.data.data_handler import DataHandler, GeoTiffTileSpec, ROI
    from src.models.ml_dataset_builder import MLDatasetBuilder
except ImportError:
    _install_flat_src_aliases()
    from src.data.data_handler import DataHandler, GeoTiffTileSpec, ROI
    from src.models.ml_dataset_builder import MLDatasetBuilder


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class MLRunConfig:
    # Data / cache
    base_url: str = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp"
    tile_id: str = "ldem_80s_20m"
    tif_filename: str = "LDEM_80S_20MPP_ADJ.TIF"
    tif_url: Optional[str] = None
    tif_path: Optional[str] = None
    persistent_dir: str = "./persistent"
    runtime_dir: Optional[str] = None
    allow_download: bool = True
    force_download: bool = False
    timeout_s: int = 120

    # ROI
    roi_size_px: int = 1024
    use_canonical_roi: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None

    # Dataset
    block_size_px: int = 64
    negative_ratio: float = 1.0
    random_state: int = 42
    test_fraction: float = 0.30
    save_dataset_csv: bool = True
    dataset_csv_name: str = "ml_dataset_rf_v1.csv"

    # Model (kept aligned to notebook baseline)
    n_estimators: int = 100
    max_depth: int = 12
    min_samples_leaf: int = 5
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1

    # Outputs
    product_name: str = "crater_proba_ml_v1"
    model_filename: str = "rf_crater_model_v1.joblib"
    feature_importance_filename: str = "crater_proba_ml_v1_feature_importances.json"
    summary_filename: str = "crater_proba_ml_v1_summary.json"


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def log_step(message: str) -> None:
    print(f"[CraterML] {message}", flush=True)


def log_path(label: str, value: str | Path | None) -> None:
    if value is None:
        print(f"[CraterML] {label}: None", flush=True)
    else:
        print(f"[CraterML] {label}: {value}", flush=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _prepare_tif_source(cfg: MLRunConfig) -> tuple[str, Optional[str], bool]:
    """
    Resolve the GeoTIFF source for DataHandler.

    This script mostly works from the ROI cache, but canonical ROI resolution
    still needs the tile shape, so we keep the same local-file / download logic.
    """
    tif_filename = cfg.tif_filename
    tif_url = cfg.tif_url
    allow_download = cfg.allow_download

    if cfg.tif_path is None:
        return tif_filename, tif_url, allow_download

    src = Path(cfg.tif_path)
    if not src.exists():
        raise FileNotFoundError(f"Local GeoTIFF not found: {src}")

    persistent_dir = Path(cfg.persistent_dir)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    cached_tif = persistent_dir / src.name

    if src.resolve() != cached_tif.resolve():
        if (not cached_tif.exists()) or (cached_tif.stat().st_size != src.stat().st_size):
            cached_tif.write_bytes(src.read_bytes())

    log_path("Local GeoTIFF", src)
    log_path("Cached GeoTIFF", cached_tif)
    return src.name, None, False

# resolve ROI either from canonical logic or from CLI args, validating as needed
def _resolve_roi(dh: DataHandler, cfg: MLRunConfig) -> ROI:
    if cfg.use_canonical_roi:
        return dh.canonical_roi(cfg.tile_id, size=cfg.roi_size_px)

    if cfg.roi is None:
        raise ValueError("Provide --roi when --no-canonical-roi is used.")

    return tuple(int(x) for x in cfg.roi)  # type: ignore[return-value]

# Save the model using joblib if available, otherwise fall back to pickle.
def _save_model(model: RandomForestClassifier, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if joblib is not None:
        joblib.dump(model, out_path)
        return str(out_path)

    pickle_path = out_path.with_suffix(".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    return str(pickle_path)

# Convert sklearn classification report to dict with zero_division=0 to avoid NaNs.
def _classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------
def run_bridge(cfg: MLRunConfig) -> Dict[str, Any]:
    """
    Run the notebook-aligned crater ML baseline.

    The logic mirrors the experimental notebook on purpose:
    1) build dataset from cache
    2) split by block_id
    3) train one Random Forest
    4) evaluate on held-out blocks
    5) predict full ROI probabilities
    6) save crater_proba_ml_v1.npy
    """
    log_step("Initialising DataHandler...")
    tif_filename, tif_url, allow_download = _prepare_tif_source(cfg)

    dh = DataHandler(
        base_url=cfg.base_url,
        tiles=[
            GeoTiffTileSpec(
                tile_id=cfg.tile_id,
                tif_filename=tif_filename,
                tif_url=tif_url,
            )
        ],
        persistent_dir=cfg.persistent_dir,
        runtime_dir=cfg.runtime_dir,
        allow_download=allow_download,
        force_download=cfg.force_download,
        timeout_s=cfg.timeout_s,
    )

    roi = _resolve_roi(dh, cfg)
    derived_dir = Path(dh.derived_dir(cfg.tile_id, roi))
    derived_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"Using ROI: {roi}")
    log_path("Derived dir", derived_dir)

    # ------------------------------------------------------------------
    # 1) Build dataset from existing ROI cache
    # ------------------------------------------------------------------
    builder = MLDatasetBuilder(
        block_size_px=cfg.block_size_px,
        random_state=cfg.random_state,
    )

    log_step("Building ML dataset from ROI cache...")
    dataset_out = builder.build_from_cache(
        dh=dh,
        tile_id=cfg.tile_id,
        roi=roi,
        include_rowcol=True,
        include_block_ids=True,
        balance_strategy="undersample",
        negative_ratio=cfg.negative_ratio,
        save_csv=cfg.save_dataset_csv,
        csv_name=cfg.dataset_csv_name,
    )

    df = dataset_out["dataframe"].copy()
    meta = dataset_out["meta"]

    feature_cols = ["elevation_m", "slope_deg", "roughness_rms"]
    target_col = "label_crater"

    required_cols = ["row", "col", "block_id", *feature_cols, target_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Built dataset is empty. Cannot train crater model.")

    H, W = tuple(meta["shape"])

    log_step(f"Dataset rows: {len(df):,}")
    log_step(
        "Class counts: "
        f"positive={(df[target_col] == 1).sum():,}, "
        f"negative={(df[target_col] == 0).sum():,}"
    )

    # ------------------------------------------------------------------
    # 2) Spatial split using block_id (same spirit as notebook)
    # ------------------------------------------------------------------
    unique_blocks = np.array(sorted(df["block_id"].unique()))
    rng = np.random.default_rng(cfg.random_state)
    rng.shuffle(unique_blocks)

    n_test_blocks = max(1, int(len(unique_blocks) * cfg.test_fraction))
    test_blocks = set(unique_blocks[:n_test_blocks])
    train_blocks = set(unique_blocks[n_test_blocks:])

    train_df = df[df["block_id"].isin(train_blocks)].copy()
    test_df = df[df["block_id"].isin(test_blocks)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train/test split produced an empty subset. Adjust block size or test fraction.")

    log_step(f"Train rows: {len(train_df):,}")
    log_step(f"Test rows: {len(test_df):,}")
    log_step(f"Train positive ratio: {float(y_train.mean()):.6f}")
    log_step(f"Test positive ratio: {float(y_test.mean()):.6f}")

    # ------------------------------------------------------------------
    # 3) Train Random Forest baseline
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    log_step("Training Random Forest baseline...")
    rf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4) Evaluate on held-out blocks
    # ------------------------------------------------------------------
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    class_report = _classification_report_dict(y_test.to_numpy(), y_pred)

    log_step(
        f"Held-out metrics: accuracy={acc:.4f}, precision={prec:.4f}, "
        f"recall={rec:.4f}, f1={f1:.4f}"
    )

    # ------------------------------------------------------------------
    # 5) Feature importances
    # ------------------------------------------------------------------
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # ------------------------------------------------------------------
    # 6) Predict crater probability for the full ROI
    # ------------------------------------------------------------------
    log_step("Predicting full ROI crater probabilities...")
    X_all = df[feature_cols]
    all_proba = rf.predict_proba(X_all)[:, 1]

    crater_proba_ml = np.full((H, W), np.nan, dtype=np.float32)
    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    crater_proba_ml[rows, cols] = all_proba.astype(np.float32)

    # ------------------------------------------------------------------
    # 7) Save outputs
    # ------------------------------------------------------------------
    product_path = derived_dir / f"{cfg.product_name}.npy"
    np.save(product_path, crater_proba_ml)

    model_path = derived_dir / cfg.model_filename
    saved_model_path = _save_model(rf, model_path)

    feature_importance_path = derived_dir / cfg.feature_importance_filename
    feature_importance_path.write_text(
        json.dumps(importance_df.to_dict(orient="records"), indent=2)
    )

    summary = {
        "config": asdict(cfg),
        "tile_id": cfg.tile_id,
        "roi": list(roi),
        "dataset": {
            "rows": int(len(df)),
            "shape": [int(H), int(W)],
            "feature_columns": feature_cols,
            "label_column": target_col,
            "positive_count": int((df[target_col] == 1).sum()),
            "negative_count": int((df[target_col] == 0).sum()),
            "positive_ratio": float(df[target_col].mean()),
            "builder_meta": meta,
            "builder_paths": dataset_out.get("paths", {}),
        },
        "split": {
            "test_fraction": float(cfg.test_fraction),
            "num_unique_blocks": int(len(unique_blocks)),
            "num_test_blocks": int(n_test_blocks),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_positive_ratio": float(y_train.mean()),
            "test_positive_ratio": float(y_test.mean()),
        },
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": int(cfg.n_estimators),
            "max_depth": int(cfg.max_depth),
            "min_samples_leaf": int(cfg.min_samples_leaf),
            "class_weight": cfg.class_weight,
        },
        "evaluation": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        },
        "feature_importances": importance_df.to_dict(orient="records"),
        "probability_raster_stats": {
            "min": float(np.nanmin(crater_proba_ml)),
            "max": float(np.nanmax(crater_proba_ml)),
            "mean": float(np.nanmean(crater_proba_ml)),
        },
        "outputs": {
            "probability_raster": str(product_path),
            "model": saved_model_path,
            "feature_importances": str(feature_importance_path),
        },
    }

    summary_path = derived_dir / cfg.summary_filename
    summary_path.write_text(json.dumps(summary, indent=2))

    log_path("Saved probability raster", product_path)
    log_path("Saved model", saved_model_path)
    log_path("Saved feature importances", feature_importance_path)
    log_path("Saved summary", summary_path)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "product_path": product_path,
        "derived_dir": derived_dir,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the notebook-aligned Endymion crater RF bridge and save "
            "crater_proba_ml_v1.npy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / cache
    parser.add_argument("--persistent-dir", default="./persistent")
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--base-url", default="https://pgda.gsfc.nasa.gov/data/LOLA_20mpp")
    parser.add_argument("--tile-id", default="ldem_80s_20m")
    parser.add_argument("--tif-filename", default="LDEM_80S_20MPP_ADJ.TIF")
    parser.add_argument("--tif-url", default=None)
    parser.add_argument("--tif-path", default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--disable-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")

    # ROI
    parser.add_argument("--roi-size", type=int, default=1024)
    parser.add_argument("--no-canonical-roi", action="store_true")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"))

    # Dataset
    parser.add_argument("--block-size-px", type=int, default=64)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--no-save-dataset-csv", action="store_true")
    parser.add_argument("--dataset-csv-name", default="ml_dataset_rf_v1.csv")

    # Model
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-jobs", type=int, default=-1)

    # Outputs
    parser.add_argument("--product-name", default="crater_proba_ml_v1")
    parser.add_argument("--model-filename", default="rf_crater_model_v1.joblib")
    parser.add_argument(
        "--feature-importance-filename",
        default="crater_proba_ml_v1_feature_importances.json",
    )
    parser.add_argument("--summary-filename", default="crater_proba_ml_v1_summary.json")

    return parser

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = MLRunConfig(
        base_url=args.base_url,
        tile_id=args.tile_id,
        tif_filename=args.tif_filename,
        tif_url=args.tif_url,
        tif_path=args.tif_path,
        persistent_dir=args.persistent_dir,
        runtime_dir=args.runtime_dir,
        allow_download=not args.disable_download,
        force_download=args.force_download,
        timeout_s=args.timeout_s,
        roi_size_px=args.roi_size,
        use_canonical_roi=not args.no_canonical_roi,
        roi=tuple(args.roi) if args.roi is not None else None,
        block_size_px=args.block_size_px,
        negative_ratio=args.negative_ratio,
        random_state=args.random_state,
        test_fraction=args.test_fraction,
        save_dataset_csv=not args.no_save_dataset_csv,
        dataset_csv_name=args.dataset_csv_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
        product_name=args.product_name,
        model_filename=args.model_filename,
        feature_importance_filename=args.feature_importance_filename,
        summary_filename=args.summary_filename,
    )

    try:
        out = run_bridge(cfg)
        evaluation = out["summary"]["evaluation"]
        log_step(
            "Done. Held-out metrics: "
            f"accuracy={evaluation['accuracy']:.4f}, "
            f"precision={evaluation['precision']:.4f}, "
            f"recall={evaluation['recall']:.4f}, "
            f"f1={evaluation['f1']:.4f}"
        )
        return 0
    except Exception as e:
        log_step(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
