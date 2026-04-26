
# Endymion Models Module

## Overview

The `src/models/` module contains the logic for transforming terrain and crater-derived features into hazard representations, as well as supporting machine learning extensions.

This module is responsible for answering the key question:

> **Given terrain and crater information, how hazardous is each location?**

---

## Files

```text
src/models/
│
├── hazard_assessor.py
├── crater_predictor.py
└── ml_dataset_builder.py
````

---

## Role in the Pipeline

```text
Terrain features (slope, roughness)
        ↓
Crater features (mask, distance, density)
        ↓
HazardAssessor
        ↓
Hazard map [0, 1]
```

Optional ML extension:

```text
Terrain features
        ↓
MLDatasetBuilder
        ↓
ML model (e.g. Random Forest)
        ↓
Crater probability map
        ↓
HazardAssessor
```

---

## Design Philosophy

The models module follows three key principles:

### Interpretability

Hazard is constructed explicitly using weighted components, not black-box models.

### Modularity

Crater and ML signals can be added or removed without breaking the system.

### Reproducibility

All model outputs are deterministic and saved as artefacts.

---

## 1. hazard_assessor.py

## Purpose

The `HazardAssessor` generates a unified hazard map by combining terrain and crater-derived signals.

---

## Inputs

* `slope_deg`
* `roughness_rms`
* (optional) `crater_mask`
* (optional) `crater_distance_m`
* (optional) `crater_density`

---

## Output

* `hazard` → float32 raster in range `[0, 1]`
* `components` → intermediate contributions
* `meta` → model configuration and statistics

---

## Hazard Model Structure

### Terrain Component

$$
H_{terrain} = w_s \cdot S_n + w_r \cdot R_n
$$

Where:

- $S_n$ = normalised slope
- $R_n$ = normalised roughness
- $w_s$ = slope weight
- $w_r$ = roughness weight
---

### Crater Component

$$
H_{crater} = w_m \cdot M + w_d \cdot D_r + w_\rho \cdot C_n
$$

Where:

- $M$ = crater mask
- $D_r$ = distance-based crater risk
- $C_n$ = normalised crater density
- $w_m$ = crater mask weight
- $w_d$ = crater distance weight
- $w_\rho$ = crater density weight

---

### Final Hazard

$$
H = w_t \cdot H_{terrain} + w_c \cdot H_{crater}
$$

Where:

- $H$ = final hazard score
- $H_{terrain}$ = terrain hazard component
- $H_{crater}$ = crater hazard component
- $w_t$ = terrain group weight
- $w_c$ = crater group weight

---

## Normalisation

All inputs are normalised before weighting:

```text
value_normalised = value / max_value
```

Examples:

* slope → divided by `slope_deg_max`
* roughness → divided by `roughness_rms_max`
* crater distance → converted into risk (inverse relationship)

---

## Impassable Terrain

A hard constraint is applied:

```text
slope_deg > threshold → hazard = 1.0
```

This ensures extremely steep terrain is always treated as unsafe.

---

## Key Idea

The hazard model is:

* simple
* explainable
* tunable

It avoids overfitting and allows clear reasoning about navigation behaviour.

---

## 2. crater_predictor.py

## Purpose

Provides crater-related raster products used by the hazard model.

---

## Modes

### `stub_v1`

* returns zero-valued crater outputs
* used for terrain-only baseline

---

### `catalogue_raster_v1`

* builds crater rasters from Robbins catalogue
* supports caching in ROI directory

---

## Outputs

```text
crater_mask
crater_distance_m
crater_density
crater_proba (for ML compatibility)
```

---

## Caching Behaviour

Crater products are saved as:

```text
crater_mask.npy
crater_mask_distance.npy
crater_mask_density.npy
crater_mask_meta.json
```

If present, they are loaded instead of recomputed.

---

## Key Idea

CraterPredictor acts as a **bridge between raw catalogue data and hazard modelling**.

---

## 3. ml_dataset_builder.py

## Purpose

Builds a supervised dataset for crater prediction.

---

## Goal

Predict crater presence using **terrain features only**.

---

## Inputs

* elevation (`dem_m`)
* slope
* roughness
* crater_mask (label only)

---

## Output

A tabular dataset:

```text
row, col, block_id,
elevation, slope, roughness,
label_crater
```

---

## Important Constraint

No data leakage:

```text
crater_mask → label ONLY
crater_distance → NOT used
crater_density → NOT used
```

---

## Why This Matters

Using crater-derived features as inputs would make the model trivial and invalid.

The goal is to learn crater likelihood from terrain characteristics.

---

## ML Role in Endymion

The ML model:

* is a **baseline extension**
* generates crater probability maps
* integrates into hazard modelling

It is **not the core of the system**, but an enhancement.

---

## Integration with Hazard Model

ML output can be used as:

```text
additional crater signal → fused into hazard
```

This allows comparison between:

* terrain-only
* terrain + crater
* terrain + ML

---

## Limitations of ML Stage

* uses simple models (e.g. Random Forest)
* no deep learning
* limited generalisation testing
* trained on ROI-scale data only

---

## Relationship to Other Modules

```text
FeatureExtractor
    → provides terrain features

CraterPredictor
    → provides crater rasters

HazardAssessor
    → fuses everything into hazard

BenchmarkRunner
    → compares model variants
```

---

## Summary

The `src/models/` module defines how terrain becomes hazard.

It provides:

* an interpretable hazard model
* crater-aware extensions
* a bridge to machine learning
* a framework for comparing different hazard representations

This module is central to Endymion’s ability to perform **hazard-aware navigation rather than simple pathfinding**.

