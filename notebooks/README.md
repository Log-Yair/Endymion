# Notebooks

# Endymion – Experimental & Demo Notebooks

This directory contains the Jupyter Notebooks used throughout the research and development lifecycle of the **Endymion** project. These files served as the primary environment for early prototyping, data exploration, and visual validation before the logic was refactored into the modular `src/` pipeline.

---

## Key Notebooks

### 1. Prototype Phase 1 Demo
**File:** `Endymion_Prototype_Phase1_Demo.ipynb`
This is the core demonstration notebook for the initial project phase. It validates the complete terrain-to-route chain, including:
- Loading LOLA south-polar DEM patches.
- Deriving slope and roughness descriptors.
- Generating terrain-only hazard maps and safe navigation paths.

### 2. Experimental Pipelines
**Files:** `Endymion_Exp_pipeline (version).ipynb` 
These notebooks document the iterative refinement of the system architecture. They were used to:
- Test the transition from a monolithic notebook workflow to a file-based caching system.
- Experiment with different hazard fusion weights and cost construction logic.
- Conduct early comparative tests between terrain-only and crater-aware models.

### 3. Crater & ML Development
**Files:** `crater.ipynb`, `Endymion_experimental_proto.ipynb`
Specialized environments for developing:
- The Robbins Crater Catalogue rasterization workflow.
- The baseline Random Forest classifier for crater probability prediction.
- Visual overlays for checking spatial alignment between DEMs and crater masks.

---

## Usage Guidelines

### Cloud Execution (Recommended)
Most notebooks are designed to be run in **Google Colab** to take advantage of high-performance CPU/GPU resources and seamless Google Drive integration for large lunar datasets.

### Transition to Source Code
Please note that these notebooks are intended for **exploration and demonstration**. The final, reproducible research framework resides in the `src/` directory. For formal benchmarking or production runs, use the command-line entry points:
- `python src/run_endymion.py`
- `python benchmark/run_benchmark.py`

---

## Notes on Reproducibility
To ensure these notebooks execute correctly:
1. Ensure the required data files (e.g., Robbins CSV and LOLA GeoTIFF) are accessible in your persistent directory.
2. Install the dependencies listed in the root `requirements.txt`.
3. Fixed random seeds are used throughout these notebooks to preserve deterministic behavior for research evaluation.
