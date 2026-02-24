
# References / origin notes:
# - It does NOT implement ML yet (Phase-2 work).
# - It is a stub to keep the pipeline modular. i don't want to hardcode "no crater" everywhere.

# -*- coding: utf-8 -*-
"""
CraterPredictor (Phase 1 stub)

Phase-1 does not require crater prediction. This placeholder:
- keeps your pipeline modular
- provides a consistent API for Phase-2

Later replace `predict_proba_map()` with a real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class CraterPredictor:
    """
    Stub crater predictor.

    Future (Phase-2):
    - load/train model
    - infer crater probability map from features (slope/roughness/illumination/etc.)
    """
    model_id: str = "stub_v1"

    def predict_proba_map(
        self,
        features: Dict[str, np.ndarray],
        dem_m: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a crater probability map aligned to the feature rasters.
        For the stub, returns zeros (no craters predicted).
        """
        # pick any raster as shape reference
        ref = None
        for v in features.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                ref = v
                break
        if ref is None:
            raise ValueError("No 2D feature rasters provided to CraterPredictor.")

        proba = np.zeros(ref.shape, dtype=np.float32)

        meta = {
            "model_id": self.model_id,
            "note": "Phase-1 stub: returns all zeros",
        }

        return {"crater_proba": proba, "meta": meta}
