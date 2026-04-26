
# Endymion Evaluation Module

## Overview

The `src/evaluation/` module is responsible for analysing navigation results and supporting controlled benchmarking across different hazard models.

It provides:

- Per-run evaluation (via `Evaluator`)
- Multi-case, multi-model benchmarking (via `BenchmarkRunner`)

This module is critical for turning Endymion from a prototype into a **testable and comparable system**.

---

## Files

```text
src/evaluation/
│
├── evaluator.py
└── benchmark_runner.py
````

---

## Role in the Pipeline

### Single Run

```text
Pathfinder output
    ↓
Evaluator
    ↓
metrics.json
```

### Benchmark Mode

```text
Multiple runs
    ↓
BenchmarkRunner
    ↓
summary.json / summary.csv
```

---

## 1. evaluator.py

## Purpose

The `Evaluator` analyses a single navigation run and computes quantitative metrics describing:

* path geometry
* hazard exposure
* planning efficiency

---

## Inputs

The evaluator expects a completed navigation run containing:

```text
hazard.npy
cost.npy
path_rc.npy
nav_meta.json
```

These are produced by the pipeline (`run_endymion.py`).

---

## Outputs

The evaluator produces:

```text
metrics.json
```

This file contains structured metrics grouped by category.

---

## Metric Categories

### Geometry Metrics

Describe the shape and length of the path:

* **path_length_m**
  Total path length in metres

* **straight_line_m**
  Euclidean distance between start and goal

* **detour_ratio**
  Ratio of actual path vs straight-line distance

* **turn_count** *(if enabled)*
  Number of directional changes along the path

---

### Safety Metrics

Describe how hazardous the path is:

* **mean_path_hazard**
  Average hazard value along the path

* **max_path_hazard**
  Highest hazard encountered

* **hazard_exposure (haz_per_m)**
  Hazard accumulated per metre

* **safety_score**
  Composite indicator derived from hazard exposure

---

### Efficiency Metrics

Describe the performance of the planner:

* **planner_total_cost**
  Total cost reported by the pathfinder

* **cost_sum_along_path**
  Sum of traversal cost along the path

* **cost_per_m**
  Cost normalised by path length

* **expansions**
  Number of nodes explored by the search

---

## Key Idea

The evaluator separates **what the path looks like** from **how it was computed**.

* Geometry → path shape
* Safety → hazard exposure
* Efficiency → search performance

This allows meaningful comparison between different hazard models.

---

## 2. benchmark_runner.py

## Purpose

The `BenchmarkRunner` executes controlled experiments across:

* multiple start/goal cases
* multiple hazard models

and aggregates results into comparable summaries.

---

## Core Concept

Only the **hazard model changes**.

Everything else remains fixed:

* ROI
* terrain data
* start/goal pairs
* cost model parameters
* pathfinding settings

This ensures a fair comparison.

---

## Benchmark Structure

### BenchmarkConfig

Defines fixed parameters:

```text
- tile_id
- ROI
- cost model (alpha, hazard_block)
- pathfinder settings (connectivity, heuristic_weight)
```

---

### BenchmarkCase

Defines a navigation scenario:

```text
case_id
start_rc
goal_rc
tier (easy / medium / hard)
```

---

### HazardModelSpec

Defines a hazard variant:

```text
model_id
kind (terrain_only, terrain_plus_crater, etc.)
parameters
```

---

## Workflow

```text
Load terrain rasters
        ↓
Build hazard model
        ↓
Convert hazard → cost
        ↓
Run pathfinding for each case
        ↓
Evaluate each run
        ↓
Aggregate results
```

---

## Outputs

Each benchmark run produces:

### Per-run folders

```text
benchmarks/<benchmark_id>/<case_id>/<model_id>/
```

Containing:

```text
hazard.npy
cost.npy
path_rc.npy
nav_meta.json
metrics.json
```

---

### Benchmark summaries

```text
summary.json
summary.csv
manifest.json
```

---

## Summary Metrics

Typical aggregated values include:

* success rate
* mean path hazard
* maximum hazard
* safety score
* cost per metre
* path length
* expansions

These allow comparison across models such as:

```text
terrain_only vs terrain_plus_crater vs ML-assisted
```

---

## Deterministic Testing

The benchmark system is designed to be:

* deterministic
* reproducible
* comparable

This means:

* same inputs → same outputs
* no random variation between runs
* results can be traced back to exact configurations

---

## Design Rationale

### Why Evaluation Matters

A path alone is not meaningful without context.

For example:

* A shorter path may be more dangerous
* A safer path may require more search effort
* A model may reduce hazard but increase cost

The evaluation module allows these trade-offs to be measured explicitly.

---

### Why Benchmarking Matters

Benchmarking turns Endymion into a **comparative system**, not just a pipeline.

It enables questions such as:

* Does crater awareness improve safety?
* Does ML reduce hazard exposure?
* What is the cost of safer navigation?

---

## Common Failure Cases

### start_or_goal_blocked

Occurs when hazard exceeds the blocking threshold.

### no_path_found

Occurs when no valid path exists under constraints.

### missing artefacts

If required files are not present, evaluation cannot proceed.

---

## Relationship to Other Modules

```text
Pathfinder
    → produces path + cost

Evaluator
    → computes metrics

BenchmarkRunner
    → runs multiple Evaluator instances
    → aggregates results
```

---

## Summary

The `src/evaluation/` module provides the analytical backbone of Endymion.

It ensures that:

* navigation outputs are measurable
* hazard models are comparable
* results are reproducible
* system behaviour can be analysed objectively
