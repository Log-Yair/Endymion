# Endymion
This repo contains my FYP project, Endymion 
## Prototype Pipeline (Reproducible Demo)

This repository contains the backend implementation of **Endymion** (DataHandler → FeatureExtractor → HazardAssessor → Pathfinder → Evaluator)
and a runnable **prototype pipeline** that reproduces the demo outputs.

### Quickstart (Recommended: Google Colab)
1. Open the notebook: - `notebooks/w20475998_Endymion_Prototype_Demo_v6_2.ipynb`
2. Run cells top-to-bottom.
3. Outputs (derived rasters + navigation runs + metrics) will be saved under:
   `derived/<tile_id>/roi_<...>/`


🔹 Prototype Demonstration  
The prototype pipeline notebook is available on the `notebooks` branch:
https://github.com/<user>/<repo>/tree/notebooks/notebooks/w20475998_Endymion_Prototype_Demo_v6_2.ipynb

### Quickstart (Local Python)
> Python 3.10+ recommended

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python scripts/run_prototype_pipeline.py
