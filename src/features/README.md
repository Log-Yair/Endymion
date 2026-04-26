
# Endymion Feature Extraction Module

## Overview

The `src/features/` module is responsible for transforming raw elevation data (DEM) into terrain-derived features used for hazard modelling.

This stage converts geometric terrain information into **interpretable descriptors** such as slope and roughness, which directly influence navigation safety.

---

## Files

```text
src/features/
│
└── feature_extractor.py
````

---

## Role in the Pipeline

```text
DEM (elevation in metres)
        ↓
FeatureExtractor
        ↓
Terrain features (slope, roughness)
        ↓
HazardAssessor
```

---

## Purpose

Raw elevation alone is not sufficient to determine traversal risk.

Instead, Endymion derives features that better represent terrain difficulty:

* slope → steepness (risk of tipping/sliding)
* roughness → local terrain variability (risk of instability)

These features form the foundation of the hazard model.

---

## FeatureExtractor

## Class: `FeatureExtractor`

### Responsibilities

* Compute terrain gradients
* Derive slope from elevation changes
* Estimate local roughness
* Ensure outputs are aligned to the ROI grid

---

## Inputs

* `dem_m`
  DEM raster in metres (2D NumPy array)

---

## Outputs

The extractor returns a dictionary containing:

```text
slope_deg
slope_rise_run
roughness_rms
```

All outputs:

* are `float32`
* match the shape of the input DEM
* are aligned to the same ROI grid

---

## 1. Slope

### Computation

Slope is computed using central differences:

s = \sqrt{(\frac{\partial z}{\partial x})^2 + (\frac{\partial z}{\partial y})^2}

Where:

* ( z ) = elevation
* ( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} ) = gradients in x and y

---

### Units

* `slope_rise_run` → unitless gradient
* `slope_deg` → slope in degrees:

[
\theta = \tan^{-1}(s)
]

---

### Interpretation

* low slope → flat, safer terrain
* high slope → steep, higher hazard

---

## 2. Roughness (RMS)

### Concept

Roughness measures how much elevation varies locally within a window.

It is approximated using the standard deviation of elevation values:

\sigma = \sqrt{\frac{1}{N} \sum (z_i - \bar{z})^2}

---

### Implementation

* sliding window (default: 5×5)
* fast path using integral images
* fallback method if NaNs are present

---

### Interpretation

* low roughness → smooth terrain
* high roughness → uneven, potentially unstable terrain

---

## Window Size

Typical configuration:

```text
roughness_window = 5
```

At 20 m/pixel resolution:

```text
5 × 5 window ≈ 100 m × 100 m local area
```

---

## Output Alignment

All feature rasters must satisfy:

```text
slope.shape == roughness.shape == dem.shape
```

This is critical for:

* hazard modelling
* cost computation
* pathfinding

---

## Handling Edge Cases

### Boundary Handling

Padding is applied to ensure output size matches input:

* `edge` padding (default)
* `reflect` padding (optional)

---

### Missing Data (NaNs)

If NaNs exist:

* gradients avoid invalid neighbours
* roughness switches to a slower but safe method

---

## Why Feature Extraction Matters

The hazard model does **not** operate directly on elevation.

Instead:

```text
DEM → features → hazard
```

This separation provides:

* better interpretability
* clearer control over hazard behaviour
* easier extension (e.g. adding new features)

---

## Design Principles

### Interpretability

Each feature has a clear physical meaning:

* slope → steepness
* roughness → terrain variability

---

### Determinism

Same DEM → same features → same hazard

---

### Efficiency

* vectorised NumPy operations
* integral-image optimisation for RMS

---

### Compatibility

Outputs are designed to integrate directly with:

* `HazardAssessor`
* `CraterPredictor`
* `BenchmarkRunner`

---

## Common Issues

### Shape mismatch

All inputs and outputs must match the ROI shape.

---

### Incorrect units

DEM must be in **metres**, not kilometres or raw radius values.

---

### Window size errors

Roughness window must be:

* odd
* ≥ 3

---

## Relationship to Other Modules

```text
DataHandler
    → provides DEM

FeatureExtractor
    → computes slope + roughness

HazardAssessor
    → consumes these features
```

---

## Summary

The `src/features/` module transforms elevation data into meaningful terrain descriptors.

It is a critical step because:

* it defines how terrain is interpreted
* it directly influences hazard modelling
* it enables safe and explainable navigation decisions
