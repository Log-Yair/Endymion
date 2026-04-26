
***

# Endymion

## A Data-Driven Framework for Lunar Terrain Hazard Assessment and Hazard-Aware Navigation

**Endymion** is a research prototype designed to investigate how lunar south-polar terrain data can be transformed into explicit hazard representations and safe navigation paths 
Rather than treating terrain analysis and route planning as isolated tasks, the system integrates them into a modular, reproducible analytical pipeline

---

## Project Overview

Developed as a Computer Science Final Year Project, Endymion addresses the challenges of navigating the extreme topography and permanently shadowed regions (PSRs) of the lunar south pole. The framework supports:

- **Automated Data Ingestion:** Standardizing LOLA Gridded DEMs (20m/pixel) and Robbins Crater Catalogue data
- **Feature Engineering:** Deriving slope (degrees) and RMS roughness from elevation patches
- **Hazard Fusion:** Fusing terrain-only and crater-aware signals into a unified [0,1] hazard index
- **Risk-Aware Navigation:** Computing optimal routes via **Weighted A*** with corridor-restricted search for scalability
- **ML Integration:** Generating crater-probability outputs using a **Random Forest** baseline trained on terrain features
- **Reproducible Benchmarking:** Comparing multiple hazard variants under a fixed evaluation contract

---

## Final Pipeline

The system transforms raw remote-sensing data into navigation outputs through the following flow:

```text
       Lunar DEM / GeoTIFF + Crater Catalogue
                       ↓
                  DataHandler (ROI Extraction & Caching)
                       ↓
                FeatureExtractor (Slope & Roughness)
                       ↓
               CraterPredictor (ML & Vector Rasterization)
                       ↓
                 HazardAssessor (Weighted Fusion)
                       ↓
                   Cost Grid (Traversal Penalties)
                       ↓
                Pathfinder (Weighted A* / Corridors)
                       ↓
            Evaluator & BenchmarkRunner (Metrics & Summaries)
```

The core objective is to generate routes that prioritize safety by responding dynamically to terrain risk rather than simply minimizing geometric distance

---

## Repository Structure

```text
Endymion/
├── src/
│   ├── data/            # DataHandler & Crater rasterization
│   ├── features/        # Topographic feature extraction (Slope, RMS)
│   ├── models/          # HazardAssessor, CraterPredictor, & ML Dataset Builder
│   ├── planning/        # Pathfinder (Weighted A* logic)
│   ├── evaluation/      # Evaluator & BenchmarkRunner logic
│   └── run_endymion.py  # Main pipeline entry point
├── benchmark/           # Scripts for multi-case benchmark runs
├── notebooks/           # Experimental and demo notebooks
├── persistent/          # Default directory for cached derived products
├── requirements.txt     # Project dependencies
└── README.md
```

---

## Getting Started

### Installation
Python 3.10+ is recommended.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Log-Yair/Endymion.git
   cd Endymion
   ```

2. **Setup virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
- **Execute a single run:**
  ```bash
  python src/run_endymion.py
  ```
  This resolves the canonical ROI, generates terrain features, builds the hazard map, and plans a path

- **Execute benchmarks:**
  ```bash
  python benchmark/run_benchmark.py
  ```
  This runs controlled comparisons across multiple start-goal cases and hazard model variants

---

## Implementation Details

### Machine Learning Bridge
The system includes a machine-learning extension that uses a **Random Forest Classifier** to predict crater likelihood from terrain descriptors
- **Features:** Elevation, Slope, and Roughness
- **Performance:** Achieved **79.41% accuracy** and an **F1-score of 0.803** on spatially held-out test data

### Pathfinding Logic
The `Pathfinder` uses Weighted A* where the cell cost is defined as Cost = 1 + alpha x H . To ensure computational efficiency in large grids, it utilizes a corridor mask that expands progressively until a path is found

---

## Academic Context

This project was submitted as part of the **BSc (Hons) Computer Science** degree at the **University of Westminster**

- **Author:** Yair Cortes Torres (W20475998) 
- **Supervisor:** Dr. Alan Immanuel Benjamin Vallavaraj 
- **Disclaimer:** Endymion is a research prototype intended for academic analysis. It is not an operational mission-planning or rover-control system

---

## References
- **NASA LRO/LOLA:** Elevation and altimetry products
- **Smith et al. (2009) & Barker et al. (2016):** Digital Elevation Modeling foundations
- **Bickel et al. (2021):** Deep learning for PSR interpretation
- **Pohl (1970):** Heuristic pathfinding theory
