# -*- coding: utf-8 -*-
"""
BenchmarkRunner for Endymion
# - Redesign based on your current BenchmarkRunner, Evaluator, HazardAssessor,
#   Pathfinder, DataHandler, and crater integration direction in Endymion.
# - Keeps compatibility with the existing per-run Evaluator contract:
#   hazard.npy, cost.npy, path_rc.npy, nav_meta.json -> metrics.json
# - Main design change:
#   benchmark_runner now supports:
#       1) multiple benchmark cases
#       2) multiple hazard model variants
#       3) benchmark-level summary aggregation
# - Idea source:
#   your current benchmark_runner.py only labels hazard_model_id in outputs,
#   but still always calls the same HazardAssessor internally.
#   This version turns hazard_model_id into an actual model registry.
"""

from __future__ import annotations

from curses import meta
import json
from dataclasses import dataclass, field # for future extensibility, e.g. adding more cost model parameters or pathfinder settings
from pathlib import Path # for clean path handling
from pickle import BUILD
import re
from typing import Any, Dict, List, Required, Tuple, Optional # for type hints, e.g. case dict structure, benchmark results, etc.

import numpy as np

# Refactored imports after repository structure cleanup

from src.data.data_handler import DataHandler, ROI
from src.models import hazard_assessor
from src.models.hazard_assessor import HazardAssessor
from src.planning.pathfinder import Pathfinder, build_cost_from_hazard
from src.evaluation.evaluator import Evaluator

RC = Tuple[int, int]

# ==================
# Dataclasses
# ==================

@dataclass
class BenchmarkConfig:
    """Fixed settings to keep comparisons fair across hazard model versions."""
    tile_id: str
    roi: ROI
    benchmark_id: str = "benchmark_v2"  # allows multiple benchmark suites to coexist without overwriting each other
    pixel_size_m: float = 20.0

    # Cost model (keep fixed for fairness)
    alpha: float = 10.0
    hazard_block: float = 0.95
    block_cost: float = 1e6

    # Pathfinder (keep fixed for fairness)
    connectivity: int = 8
    heuristic_weight: float = 2.0
    corridor_radii: Tuple[int, ...] = (25, 50, 80, 120, 180, 260, 400)

@dataclass
class BenchmarkCase:
    """Represents one start/goal pair and its metadata."""
    case_id: str
    start_rc: RC
    goal_rc: RC
    tier : str = "default"  # optional tier for categorizing cases by difficulty or other criteria
    notes: Optional[str] = None  # optional free-form notes about the case

    @classmethod
    def from_dict(cls, d: Dict[str, Any], tier: str = "default") -> BenchmarkCase:
        """Helper to create a BenchmarkCase from a dict, e.g. loaded from JSON."""
        return cls(
            case_id=d["id"],
            start_rc=tuple(d["start_rc"]),
            goal_rc=tuple(d["goal_rc"]),
            tier=tier,
            notes=d.get("notes"),
        )

@dataclass
class HazardModelSpec:
    """Specifies a hazard model variant to test.
    kind:
    - terrain_only
    -terrain_plus_crater
    -precomputed_hazard

    params:
    FREE - form dict for model specific settrings

    """
    model_id: str  # e.g. "terrain_only_v1", "terrain_plus_crater_v1", "precomputed_hazard_v1"
    kind: str  # e.g. "terrain_only", "terrain_plus_crater", "precomputed_hazard"
    params: Dict[str, Any] = field(default_factory=dict)  # model-specific parameters, e.g. crater_weight for terrain_plus_crater


# ==================
# BenchmarkRunner
# ==================


class BenchmarkRunner:
    """
    Multi-case, multi-model benchmark runner for Endymion.

    Strategy
    --------
    1) Load derived rasters once for the tile+ROI.
    2) Build each hazard model once.
    3) Run all benchmark cases against each model.
    4) Save one run directory per (case, model).
    5) Aggregate into summary.json + summary.csv.
    """

    def __init__(self, dh: DataHandler, cfg: BenchmarkConfig):
        self.dh = dh
        self.cfg = cfg

        self.pathfinder = Pathfinder(
            connectivity=cfg.connectivity,
            block_cost=cfg.block_cost,
            heuristic_weight=cfg.heuristic_weight,
        )

    # directory helpers to keep outputs organized by benchmark_id, case_id, and hazard_model_id
    def _benchmark_root(self) -> Path:
        """Root directory where all benchmark results for this ROI live."""
        return self.dh.derived_dir(self.cfg.tile_id, self.cfg.roi) / "benchmarks" / self.cfg.benchmark_id

    def _case_dir(self, case_id: str, hazard_model_id: str) -> Path:
        """Directory for one case and one hazard model variant."""
        return self._benchmark_root() / case_id / hazard_model_id
    
    # input loading

    def _load_base_inputs(self) -> Dict[str, np.ndarray]:
        """Loads the base rasters needed for hazard assessment and pathfinding."""
        derived = self.dh.load_derived(self.cfg.tile_id, self.cfg.roi)
        if derived is None:
            raise RuntimeError(
                "Derived rasters not found. Run the pipeline once to generate "
                "dem_m, slope_deg, roughness_rms in the derived cache."
            )
        required = ["dem_m", "slope_deg", "roughness_rms"]
        missing = [k for k in required if k not in derived]
        if missing:
            raise RuntimeError(f"Missing required rasters: {missing}")
        return derived
    
    def _build_hazard_for_model(
        self,
        model: HazardModelSpec,
        base: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build hazard map and metadata for one model variant.
        """
        dem_m = base["dem_m"]
        slope_deg = base["slope_deg"]
        roughness_rms = base["roughness_rms"]

        if model.kind == "terrain_only":
            assesor = HazardAssessor(
                slope_deg_max = model.params.get("slope_deg_max", 34.55),  # default to current value
                roughness_rms_max = model.params.get("roughness_rms_max", 18.56),  # default to current value
                w_slope= model.params.get("w_slope", 0.5),  # default to equal weighting
                w_roughness= model.params.get("w_roughness", 0.5),  # default to equal weighting
                use_impassable_mask= model.params.get("use_impassable_mask", True),  # default to using the mask
                impassable_slope_deg= model.params.get("impassable_slope_deg", 40.0),  # default to current value

            )

            out = assesor.assess(slope_deg=slope_deg, roughness_rms=roughness_rms, dem_m=dem_m)

            meta = dict(out["meta"])
            meta["model"] = model.model_id
            meta["benchmark_model_kind"] = model.kind

            return {
                "hazard": out["hazard"],
                "components": out.get("components", {}),  # in case the assessor provides component maps for analysis
                "hazard_meta": meta,
            }
        elif model.kind == "terrain_plus_crater":
            """
            Expected crater products already aligned to ROI.
            Minimal prototype combination:
                crater_term = w_crater_density * density_norm
                            + w_crater_distance * (1 - distance_norm)
            then combine with terrain hazard.
            """
            crater_density = np.asarray(
                model.params["crater_density"], dtype=np.float32
            )
            crater_distance_m = np.asarray(
                model.params["crater_distance_m"], dtype=np.float32
            )

            if crater_density.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: crater_density shape {crater_density.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )
            if crater_distance_m.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: crater_distance_m shape {crater_distance_m.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )
            
            # Assess terrain hazard first since crater-based hazard will be fused on top of it and may need to be masked by impassable terrain

            terrain_assessor = HazardAssessor(
                slope_deg_max=model.params.get("slope_deg_max", 34.55),
                roughness_rms_max=model.params.get("roughness_rms_max", 18.56),
                w_slope=model.params.get("w_slope", 0.7),
                w_roughness=model.params.get("w_roughness", 0.3),
                use_impassable_mask=model.params.get("use_impassable_mask", True),
                impassable_slope_deg=model.params.get("impassable_slope_deg", 40.0),
            )

            # Assess terrain hazard first since crater-based hazard will be fused on top of it and may need to be masked by impassable terrain
            terrain_out = terrain_assessor.assess(
                slope_deg=slope_deg,
                roughness_rms=roughness_rms,
                dem_m=dem_m,
            )
            terrain_hazard = terrain_out["hazard"].astype(np.float32, copy=False) # ensure float32 for fusion and output

            # crater_density is already ~[0,1] in the current crater_raster design
            density_n = np.clip(crater_density, 0.0, 1.0)

            max_distance_m = float(model.params.get("max_distance_m", 1000.0)) # max distance for normalization, default to 1km which is a common scale for crater influence, but should be set based on the actual crater_distance_m values provided to avoid excessive clipping
            if max_distance_m <= 0:
                raise ValueError("max_distance_m must be > 0 for terrain_plus_crater.")
            distance_n = np.clip(crater_distance_m / max_distance_m, 0.0, 1.0) # normalize distance to [0,1] where 0 is at the crater and 1 is at or beyond max_distance_m
            proximity_n = 1.0 - distance_n

            w_density = float(model.params.get("w_crater_density", 0.5))
            w_proximity = float(model.params.get("w_crater_proximity", 0.5))
            crater_term = (w_density * density_n + w_proximity * proximity_n).astype(
                np.float32
            )

            w_terrain = float(model.params.get("w_terrain", 0.8)) # terrain weight in final fusion, default to 0.8 to keep terrain as the primary driver of hazard while allowing crater information to modulate it
            w_crater = float(model.params.get("w_crater", 0.2)) # crater weight in final fusion, default to 0.2 to allow crater information to modulate terrain hazard without overwhelming it, but should be set based on the expected reliability and importance of the crater products relative to the terrain products

            hazard = np.clip(
                w_terrain * terrain_hazard + w_crater * crater_term,
                0.0,
                1.0,
            ).astype(np.float32)

            # use impassable mask based on slope_deg to set hazard=1 for any terrain above the impassable_slope_deg, since those areas should be avoided at all costs regardless of crater information. this also ensures that if the crater products have any issues (e.g. missing data, misalignment) the model will still produce a safe output by relying on the terrain-based hazard and the impassable mask.
            if model.params.get("use_impassable_mask", True):
                impassable_slope_deg = float(model.params.get("impassable_slope_deg", 40.0))
                hazard[slope_deg > impassable_slope_deg] = 1.0

            meta = {
                "model": model.model_id,
                "benchmark_model_kind": model.kind,
                "terrain_component": terrain_out["meta"],
                "crater_component": {
                    "w_crater_density": w_density,
                    "w_crater_proximity": w_proximity,
                    "max_distance_m": max_distance_m,
                    "density_stats": {
                        "min": float(np.nanmin(density_n)),
                        "max": float(np.nanmax(density_n)),
                        "mean": float(np.nanmean(density_n)),
                    },
                    "proximity_stats": {
                        "min": float(np.nanmin(proximity_n)),
                        "max": float(np.nanmax(proximity_n)),
                        "mean": float(np.nanmean(proximity_n)),
                    },
                },
                "fusion": {
                    "w_terrain": w_terrain,
                    "w_crater": w_crater,
                },
                "stats": {
                    "hazard_min": float(np.nanmin(hazard)),
                    "hazard_max": float(np.nanmax(hazard)),
                    "hazard_mean": float(np.nanmean(hazard)),
                    "nan_ratio": float(np.isnan(hazard).mean()),
                },
            }

            return {
                "hazard": hazard,
                "components": {
                    "terrain_hazard": terrain_hazard,
                    "crater_density_norm": density_n,
                    "crater_proximity_norm": proximity_n,
                    "crater_term": crater_term,
                },
                "hazard_meta": meta,
            }

        elif model.kind == "precomputed_hazard":
            hazard = np.asarray(model.params["hazard"], dtype=np.float32)
            if hazard.shape != slope_deg.shape:
                raise ValueError(
                    f"{model.model_id}: precomputed hazard shape {hazard.shape} "
                    f"does not match terrain shape {slope_deg.shape}"
                )

            # meta data about the hazard map itself. i don't want to bake the model-specific parameters into the meta since they can be arbitrary and may not be relevant for all model kinds, but we still want to capture some basic stats about the hazard map for analysis and debugging
            meta = {
                "model": model.model_id,
                "benchmark_model_kind": model.kind,
                "stats": {
                    "hazard_min": float(np.nanmin(hazard)),
                    "hazard_max": float(np.nanmax(hazard)),
                    "hazard_mean": float(np.nanmean(hazard)),
                    "nan_ratio": float(np.isnan(hazard).mean()),
                },
            }

            
            # return hazard with minimal metadata since it's precomputed and i don
            return {
                "hazard": hazard,
                "components": {},
                "hazard_meta": meta,
            }
        # potentially add more model kinds in the future, e.g. "ml_based_hazard" where the hazard is produced by a machine learning model that takes the base rasters as input

        else:
            raise ValueError(f"Unknown hazard model kind: {model.kind}")

    # ===============
    # Running cases
    # ===============

    def run_case_model(
            self,
            case: BenchmarkCase,
            model: HazardModelSpec,
            model_output: Dict[str, Any],
        ) -> Dict[str, Any]:

        ''' run one benchmark case against one that has been built by _build_hazard_for_model, to separate the hazard construction from the pathfinding and evaluation, and allow for more flexible experimentation and debugging. '''

        hazard = model_output["hazard"]
        hazard_meta = model_output["hazard_meta"]

        cost = build_cost_from_hazard(
            hazard,
            alpha=self.cfg.alpha,
            hazard_block=self.cfg.hazard_block,
            block_cost=self.cfg.block_cost,
        )

        # corridor search using the settings from the becnahmark config, which should be fixed across all models for a given benchmark suite to ensure fair comparisons. the corridor search will still adaptively find the best radius for each case, but the set of candidate radii and other pathfinder settings will be the same for all models.
        corr = self.pathfinder.find_path_corridor(
            cost=cost,
            start=case.start_rc,
            goal=case.goal_rc,
            corridor_radii=self.cfg.corridor_radii,
            hazard=hazard,
        )

        best = corr.get("best") # best path found across all corridors, or None if no path found
        best_radius = corr.get("best_radius_px") # the radius of the corridor that produced the best path, or None if no path found
        experiment = corr.get("experiment", []) # detailed info about the pathfinding experiment, e.g. which corridors were tried, how many nodes expanded in each, etc.

        success = bool(best and best.get("success", False)) # whether a successful path was found in any corridor
        path_rc = (
            np.asarray(best["path_rc"], dtype=np.int32) 
            if success 
            else np.zeros((0, 2), dtype=np.int32)
        )

        

    

























    def run_case(self, case: Dict[str, Any], hazard_model_id: str = "terrain_weighted_v1") -> Dict[str, Any]:
        """Runs one start/goal pair and saves artifacts."""
        case_id = case["id"]
        start: RC = tuple(case["start_rc"])
        goal: RC = tuple(case["goal_rc"])

        derived = self.dh.load_derived(self.cfg.tile_id, self.cfg.roi)
        if derived is None:
            raise RuntimeError(
                "Derived rasters not found. Run the pipeline once to generate "
                "dem_m, slope_deg, roughness_rms in the derived cache."
            )

        dem_m = derived["dem_m"]
        slope_deg = derived["slope_deg"]
        roughness_rms = derived["roughness_rms"]

        haz_out = self.hazard_assessor.assess(slope_deg=slope_deg, roughness_rms=roughness_rms, dem_m=dem_m)
        hazard = haz_out["hazard"]
        hazard_meta = haz_out["hazard_meta"]

        cost = build_cost_from_hazard(
            hazard,
            alpha=self.cfg.alpha,
            hazard_block=self.cfg.hazard_block,
            block_cost=self.cfg.block_cost,
        )

        corr = self.pathfinder.find_path_corridor(
            cost=cost,
            start=start,
            goal=goal,
            corridor_radii=self.cfg.corridor_radii,
            hazard=hazard,
        )

        best = corr.get("best", None)
        best_radius = corr.get("best_radius_px", None)  # robust if missing
        experiment = corr.get("experiment", [])

        success = bool(best and best.get("success", False))
        path_rc = np.asarray(best["path_rc"], dtype=np.int32) if success else np.zeros((0, 2), dtype=np.int32)

        out_dir = self._case_dir(case_id, hazard_model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "hazard.npy", hazard.astype(np.float32))
        np.save(out_dir / "cost.npy", cost.astype(np.float32))
        np.save(out_dir / "path_rc.npy", path_rc)

        nav_meta = {
            "tile_id": self.cfg.tile_id,
            "roi": list(self.cfg.roi),
            "shape": list(hazard.shape),
            "run_name": f"{self.cfg.benchmark_id}/{case_id}/{hazard_model_id}",
            "products": ["hazard", "cost", "path_rc"],
            "nav_meta": {
                "start_rc": list(start),
                "goal_rc": list(goal),
                "hazard_assessor": hazard_meta,
                "cost_model": {
                    "alpha": self.cfg.alpha,
                    "hazard_block": self.cfg.hazard_block,
                    "block_cost": self.cfg.block_cost,
                },
                "pathfinder": {
                    "connectivity": self.cfg.connectivity,
                    "heuristic_weight": self.cfg.heuristic_weight,
                    "corridor_radii": list(self.cfg.corridor_radii),
                    "corridor_radius_px_used": best_radius,
                },
                "result": {
                    "success": success,
                    "total_cost": float(best["total_cost"]) if success else None,
                    "path_len": int(path_rc.shape[0]),
                    "expansions": int((best.get("meta") or {}).get("expansions", -1)) if success else int((best.get("meta") or {}).get("expansions", -1)),
                    "failure_reason": None if success else "no_path_found_in_any_corridor",
                },
                "benchmark": {
                    "benchmark_id": self.cfg.benchmark_id,
                    "case_id": case_id,
                    "tier": case.get("tier"),
                    "hazard_model_id": hazard_model_id,
                },
                "corridor_experiment": experiment,
            },
        }

        (out_dir / "nav_meta.json").write_text(json.dumps(nav_meta, indent=2, allow_nan=False))

        evaluator = Evaluator(out_dir, pixel_size_m=self.cfg.pixel_size_m)
        evaluator.save("metrics.json")

        return {
            "case_id": case_id,
            "hazard_model_id": hazard_model_id,
            "success": success,
            "out_dir": str(out_dir),
        }

    def run_benchmark_file(self, benchmark_json_path: str | Path, hazard_model_id: str = "terrain_weighted_v1") -> List[Dict[str, Any]]:
        """Loads the benchmark spec file and runs all cases (tiers included)."""
        spec = json.loads(Path(benchmark_json_path).read_text())

        # Optional safety check if spec provides these fields
        if "tile_id" in spec and spec["tile_id"] != self.cfg.tile_id:
            raise ValueError(f"Benchmark spec tile_id={spec['tile_id']} does not match cfg.tile_id={self.cfg.tile_id}")
        if "roi" in spec and tuple(spec["roi"]) != tuple(self.cfg.roi):
            raise ValueError(f"Benchmark spec roi={spec['roi']} does not match cfg.roi={list(self.cfg.roi)}")

        results: List[Dict[str, Any]] = []

        for tier_name, cases in spec["tiers"].items():
            for case in cases:
                case = dict(case)
                case["tier"] = tier_name
                res = self.run_case(case, hazard_model_id=hazard_model_id)
                res["tier"] = tier_name
                results.append(res)

        root = self._benchmark_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / f"run_log_{hazard_model_id}.json").write_text(json.dumps(results, indent=2, allow_nan=False))

        return results
    
