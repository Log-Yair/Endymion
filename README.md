# Endymion – Lunar Polar Navigation Aid

Endymion is a research prototype designed to support hazard-aware navigation
for future lunar missions operating near the lunar south pole.

The system analyses terrain data, estimates hazards, and generates safe
navigation paths using a modular pipeline architecture.

---

## System Pipeline

The navigation pipeline follows this structure:

DataHandler → FeatureExtractor → CraterPredictor → HazardAssessor → Pathfinder → Evaluator

1. Terrain data is loaded and cached
2. Surface features such as slope and roughness are extracted
3. Crater hazards are predicted or rasterised
4. A hazard map is generated
5. A navigation path is planned
6. Results are evaluated and stored

---

## Repository Structure


src/ → core system modules
notebooks/ → prototype and demonstration notebooks
data/ → datasets and cached outputs
docs/ → diagrams and project report


---

## Running the Prototype

Activate the virtual environment:


..venv\Scripts\Activate.ps1


Example module test:


python -c "from src.data.data_handler import DataHandler"


The full prototype pipeline is demonstrated in:


notebooks/Endymion_Prototype_Phase1_Demo.ipynb

---

## Author - Yair Cortes

Final Year Project   
Lunar Navigation and Hazard Mapping System
