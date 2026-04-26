***

# Endymion Core Source (`src/`)

## Overview
This directory contains the modular implementation of the **Endymion** pipeline. Each package is responsible for a distinct stage of the hazard-aware navigation workflow, transforming raw lunar remote-sensing data into safety-weighted navigation paths through a reproducible and traceable process.

The pipeline is orchestrated via the main runner:
```bash
python src/run_endymion.py
```

## Pipeline Flow
The system follows a linear execution model where each stage consumes artifacts from the previous one and saves its own outputs for reuse:

**Lunar DEM** $\rightarrow$ **DataHandler** $\rightarrow$ **FeatureExtractor** $\rightarrow$ **CraterPredictor** $\rightarrow$ **HazardAssessor** $\rightarrow$ **Pathfinder** $\rightarrow$ **Evaluator**

---

## 1. Data Management (`src/data/`)
This module handles the ingestion and spatial standardization of lunar datasets.

* **`data_handler.py`**: Responsible for loading **LOLA south-polar GeoTIFFs**, resolving the **1024 × 1024 canonical ROI**, and managing persistent and runtime caching. It ensures all subsequent modules operate on the same standardized spatial contract.
* **`crater_raster.py`**: Processes the **Robbins Crater Catalogue** (vector data) and rasterizes it into ROI-aligned binary masks, distance-to-crater maps, and local density maps.

## 2. Feature Extraction (`src/features/`)
* **`feature_extractor.py`**: Derives quantitative geomorphological descriptors from elevation data. 
    * **Slope**: Computed using central difference gradients.
    * **Roughness**: Estimated through local **RMS (Root Mean Square)** surface variation in a moving window.

## 3. Modeling and Prediction (`src/models/`)
This package contains the logic for defining surface risk and predicting hazards.

* **`hazard_assessor.py`**: Fuses terrain and crater signals into a unified [0, 1] hazard raster using an explainable weighted sum . It also enforces a minimum safety policy by applying an **impassable-slope rule** before clipping the hazard map.
* **`crater_predictor.py`**: Provides crater-related signals to the assessor. It supports a baseline **Random Forest** extension that predicts crater probability based on terrain features.
* **`ml_dataset_builder.py`**: Constructs supervised per-pixel datasets for ML training while preventing "data leakage" by ensuring crater-derived features are strictly used as labels, not predictors.

## 4. Path Planning (`src/planning/`)
* **`pathfinder.py`**: Implements a **Weighted A*** algorithm operating over a traversal-cost grid.
    * **Cost Construction**: Converts hazards into penalties using the formula $Cost = 1 + \alpha H$.
    * **Constraint Handling**: Supports 8-connectivity grids, hazard-based blocking (threshold $\ge 0.95$), and **corridor-restricted search** to improve computational tractability.

## 5. Evaluation (`src/evaluation/`)
* **`evaluator.py`**: Analyzes single navigation runs, generating `metrics.json` to summarize path geometry (length, detour ratio), safety (mean/max hazard), and search efficiency (expansions).
* **`benchmark_runner.py`**: Orchestrates **controlled comparisons** across multiple cases. It freezes planning and cost settings while varying only the hazard model to provide a fair assessment of different risk-modeling strategies.

---

## Design Philosophy
The source code is governed by five core principles derived from best practices in scientific computing:
1.  **Modularity**: Each package has a clearly bounded responsibility
2.  **Reproducibility**: Strong use of saved artifacts and structured metadata ensures identical results across reruns.
3.  **Determinism**: Elimination of hidden randomness in pipeline execution
4.  **Traceability**: Every execution stores its configuration and intermediate products
5.  **Extensibility**: The modular architecture allows new models (e.g., illumination) to be added without redesigning the planning logic

## Developer Notes
* **Spatial Consistency**: All generated rasters must match the shape and alignment of the canonical ROI resolved by the `DataHandler`
* **JSON Integrity**: Output metadata and metrics must be sanitized for standard JSON compatibility (avoiding `NaN` or `Inf` values)
