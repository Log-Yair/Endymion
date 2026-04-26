
# Endymion – Phase 2: ML & Crater Integration

## Overview
This branch represents **Phase 2** of the Endymion project. Building upon the initial terrain-only prototype, this version introduces a modular, multi-factor hazard model that integrates learned crater probabilities and geomorphological crater signals into the navigation pipeline.

The primary focus of this branch is the implementation of a **Random Forest** baseline for crater prediction and the formalization of a **fixed comparison contract** for benchmarking different hazard-modeling strategies.

---

## New Features in Phase 2

### 1. Multi-Factor Hazard Fusion
The `HazardAssessor` has been extended to support group-level weighted fusion of terrain and crater signals.
- **Terrain Drivers:** Normalized slope and RMS roughness.
- **Crater Drivers:** Crater presence (binary mask), crater proximity (distance-to-risk), and local crater density.
- **Safety Policy:** Implementation of a hard "impassable-slope" rule to explicitly block non-traversable terrain before pathfinding.

### 2. Machine Learning Bridge
A dedicated supervised learning stage has been added to provide predictive hazard assessments in data-poor regions:
- **Random Forest Baseline:** Trained on per-pixel terrain features (elevation, slope, roughness) to estimate crater likelihood.
- **Spatial Validation:** Uses a block-based spatial split to prevent data leakage and provide a defensible evaluation of model generalization.
- **Probability Raster:** Outputs an ROI-aligned probability map (`crater_proba_ml_v1.npy`) compatible with the main navigation pipeline.

### 3. Reproducible Benchmarking Framework
The system now includes a `BenchmarkRunner` designed for fair model evaluation:
- **Fixed Execution Contract:** The ROI, pathfinder settings (Weighted A*), and cost conversion parameters are held constant across runs.
- **Comparative Analysis:** Directly compares **Terrain-only**, **Handcrafted Crater-aware**, and **ML-assisted** variants.
- **Automated Metrics:** Produces summary logs including success rates, mean path hazard, cost per meter, and node expansions.

---

## Phase 2 Pipeline Flow
The augmented pipeline adds dedicated stages for crater processing and ML artifact generation:

```text
Lunar DEM (GeoTIFF) + Robbins Catalogue
                  ↓
             DataHandler
                  ↓
          FeatureExtractor
          ↙               ↘
 CraterPredictor     ML Bridge (RF Model)
          ↘               ↙
            HazardAssessor
                  ↓
          Pathfinder (Weighted A*)
                  ↓
            Evaluator (Metrics)
```

---

## Branch-Specific Entry Points

### Run the ML Training Bridge
To build the supervised dataset, train the Random Forest baseline, and save the probability raster:
```bash
python -m src.models.ml_dataset_builder
# Followed by
python src/run_crater_ml.py
```

### Run Phase 2 Benchmarks
To execute the multi-case comparison between the three hazard variants:
```bash
python benchmark/run_benchmark.py
```

---

## Implementation Notes
- **GeoTIFF Workflow:** This branch prioritizes the analysis-ready GeoTIFF DEM workflow over earlier legacy formats.
- **Artifact Reuse:** Derived terrain and crater products are cached in the `persistent/derived/` directory to ensure high performance during iterative benchmarking.
- **Experimental Status:** While the ML bridge provides a stable integration path, the current results suggest that the terrain-only baseline remains the most robust safety model under current weighting schemes.

---
