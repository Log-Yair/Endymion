
## Prototype Demo (Runnable Notebook)

This repository contains the backend implementation of **Endymion (Phase-1 Prototype)** and an **official demo notebook** that runs the complete prototype pipeline end-to-end:

- Load lunar DEM data (LOLA)
- Extract a canonical ROI (1024×1024)
- Compute slope and roughness features
- Generate a terrain hazard map
- Run hazard-aware navigation (corridor-based Weighted A*)
- Save artefacts + evaluation metrics (`metrics.json`)

The official demo notebook is:
- `w20475998_Endymion_Prototype_Demo_v6_2.ipynb`

---

## Recommended: Run in Google Colab. 
Works on other IDEs too, but looses in performance

1. Open the demo notebook in Colab (upload it, or open directly from GitHub).
2. Run the notebook **top-to-bottom**.

## Drive vs Local Storage (important)
Near the top of the notebook you will see:

USE_DRIVE = True

**Change to False if you dont wish to install it on your drive**
