
# Endymion Documentation

## System Design & Architecture

This directory contains documentation for the internal design, architecture, and reasoning behind the Endymion pipeline.

The goal of this document is to explain **how the system works**, not just how to run it.

---

## 1. System Overview

Endymion is a modular pipeline that transforms lunar terrain data into hazard-aware navigation outputs.

The system is designed around a **data-driven processing flow**, where each stage produces reusable artefacts that can be cached, inspected, and reused.

### Core Idea

Navigation is treated as a **hazard-constrained optimisation problem**, not a shortest-path problem.

---

## 2. High-Level Pipeline

```text
DEM / GeoTIFF
    ↓
DataHandler
    ↓
FeatureExtractor
    ↓
CraterPredictor (optional)
    ↓
HazardAssessor
    ↓
Cost Grid (hazard → cost)
    ↓
Pathfinder (Weighted A*)
    ↓
Evaluator
````

Each module has a single responsibility and communicates through saved artefacts rather than direct coupling.

---

## 3. Design Principles

### Reproducibility

* All intermediate outputs are saved to disk
* Runs can be reloaded without recomputation
* Benchmarking uses fixed inputs and settings

### Modularity

* Each component is independent
* Modules can be replaced or extended without breaking the pipeline

### Traceability

* Each run produces metadata (`nav_meta.json`)
* Parameters, inputs, and outputs are recorded

### Determinism

* Same inputs produce the same outputs
* No hidden randomness in the pipeline

### Extensibility

* Designed to support:

  * machine learning models
  * illumination models
  * additional hazard signals

---

## 4. Core Modules

---

### 4.1 DataHandler

**Role:**

* Load and validate DEM data
* Resolve canonical ROI
* Handle caching and storage
* Save derived artefacts

**Key concepts:**

* GeoTIFF-first workflow
* Canonical ROI (1024 × 1024)
* Persistent vs runtime storage

---

### 4.2 FeatureExtractor

**Role:**

* Convert DEM into terrain descriptors

**Outputs:**

* `slope_deg`
* `slope_rise_run`
* `roughness_rms`

**Key idea:**
Terrain hazard is not derived from elevation directly, but from **derived terrain features**.

---

### 4.3 CraterPredictor

**Role:**

* Generate crater-related raster products aligned to the ROI

**Outputs:**

* crater mask
* crater distance map
* crater density map

**Modes:**

* `stub_v1` → no crater data
* `catalogue_raster_v1` → derived from Robbins catalogue

---

### 4.4 HazardAssessor

**Role:**

* Convert terrain and crater signals into a unified hazard map

**Hazard Model Structure:**

```text
terrain_component = f(slope, roughness)

crater_component = f(mask, distance, density)

final_hazard = terrain_weight * terrain_component
             + crater_weight  * crater_component
```

**Output:**

* Hazard map in range [0, 1]

**Key idea:**
Hazard is an **interpretable weighted model**, not a black-box prediction.

---

### 4.5 Cost Model

**Conversion:**

```text
cost = 1 + alpha * hazard
```

Cells above a threshold are treated as blocked:

```text
hazard >= hazard_block → blocked cell
```

**Key idea:**
Transforms hazard into something usable by pathfinding.

---

### 4.6 Pathfinder

**Algorithm:**

* Weighted A* (8-connectivity)

**Features:**

* hazard-based blocking
* corridor-restricted search (optional)
* heuristic weighting for efficiency

**Output:**

* path coordinates
* total cost
* search expansions

**Key idea:**
The planner does not ensure safety —
**the hazard model does.**

---

### 4.7 Evaluator

**Role:**

* Analyse navigation results

**Metrics include:**

#### Geometry

* path length
* detour ratio
* turn count

#### Safety

* mean hazard
* maximum hazard
* hazard exposure

#### Efficiency

* total cost
* cost per metre
* expansions

---

### 4.8 BenchmarkRunner

**Role:**

* Run controlled comparisons across models

**Keeps constant:**

* ROI
* start/goal cases
* pathfinding settings

**Varies:**

* hazard model

**Purpose:**
To determine how hazard modelling affects navigation behaviour.

---

## 5. Data Flow & Artefacts

Each stage produces saved outputs:

```text
Derived (terrain stage)
├── dem_m.npy
├── slope_deg.npy
├── roughness_rms.npy

Navigation run
├── hazard.npy
├── cost.npy
├── path_rc.npy
├── nav_meta.json
├── metrics.json
```

**Key idea:**
Everything is stored → everything is reproducible.

---

## 6. Hazard Modelling Philosophy

Endymion avoids black-box decision making.

Instead:

* hazard is explicitly defined
* each component is interpretable
* weights can be tuned and analysed

This makes the system suitable for:

* research
* benchmarking
* controlled experimentation

---

## 7. Prototype vs Final System

### Prototype

```text
DEM → Features → Hazard → Path
```

* terrain-only
* notebook-based
* limited structure

### Final System

```text
DEM → Features → Crater → Hazard → Cost → Path → Evaluation → Benchmark
```

* modular architecture
* saved artefacts
* multi-model comparison
* reproducible runs

---

## 8. Limitations

* ROI-based (not global navigation)
* no illumination modelling (yet)
* ML integration is basic
* not real-time
* depends on DEM quality

---

## 9. Future Extensions

* illumination-aware hazard modelling
* improved ML crater prediction
* multi-ROI / large-scale traversal
* mission-specific constraints
* integration with real robotic systems

---

## 10. Relationship to Report

This documentation complements the Final Year Project report.

* Report → explains methodology, theory, and evaluation
* This file → explains implementation structure and system design

Both together provide a complete understanding of Endymion.

```
