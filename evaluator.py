import json
from pathlib import Path
import numpy as np

def _load_json_allow_infinity(path: Path) -> dict:
    """
    Loads JSON that may contain Infinity/-Infinity/NaN (non-standard JSON).
    Your nav_meta has "Infinity" in failed corridor trials.
    """
    s = path.read_text()

    s = s.replace(": Infinity", ": 1e309")   # becomes float('inf')
    s = s.replace(": -Infinity", ": -1e309")
    s = s.replace(": NaN", ": null")

    return json.loads(s)


def _path_step_lengths_px(path_rc: np.ndarray) -> np.ndarray:
    """Step lengths between consecutive path nodes in pixel units."""
    d = np.diff(path_rc, axis=0)
    return np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2))


def _count_turns(path_rc: np.ndarray) -> int:
    """Counts turns as changes in step direction (simple smoothness proxy)."""
    d = np.diff(path_rc, axis=0)
    d = np.clip(d, -1, 1)
    changes = np.any(d[1:] != d[:-1], axis=1)
    return int(np.sum(changes))


def evaluate_run(run_dir: str | Path, pixel_size_m: float = 20.0) -> dict:
    """
    Evaluates a navigation run directory containing:
      hazard.npy, cost.npy, path_rc.npy, nav_meta.json

    Uses your meta schema: settings live under meta["nav_meta"].
    """
    run_dir = Path(run_dir)

    hazard = np.load(run_dir / "hazard.npy")
    cost = np.load(run_dir / "cost.npy")
    path_rc = np.load(run_dir / "path_rc.npy")
    meta = _load_json_allow_infinity(run_dir / "nav_meta.json")

    nav = meta["nav_meta"]

    start_rc = nav["start_rc"]
    goal_rc = nav["goal_rc"]

    connectivity = int(nav["pathfinder"].get("connectivity", 8))

    # Cost-model threshold that defines "blocked" hazard region (great safety metric threshold)
    hazard_block = float(nav["cost_model"].get("hazard_block", 0.95))
    alpha = float(nav["cost_model"].get("alpha", 10.0))
    block_cost = float(nav["cost_model"].get("block_cost", 1_000_000.0))

    # Global hazard stats (for normalisation)
    haz_stats = nav["hazard_assessor"]["stats"]
    global_haz_mean = float(haz_stats.get("hazard_mean", np.nan))
    global_haz_min = float(haz_stats.get("hazard_min", np.nan))
    global_haz_max = float(haz_stats.get("hazard_max", np.nan))

    # sample along path 
    rr = path_rc[:, 0].astype(int)
    cc = path_rc[:, 1].astype(int)

    H_path = hazard[rr, cc].astype(float)
    C_path = cost[rr, cc].astype(float)

    # If any NaNs (there shouldnt be any), drop from statistics
    Hf = H_path[np.isfinite(H_path)]
    Cf = C_path[np.isfinite(C_path)]

    # geometry 
    step_len_px = _path_step_lengths_px(path_rc)
    path_length_px = float(np.sum(step_len_px))
    path_length_m = path_length_px * pixel_size_m

    sr, sc = float(start_rc[0]), float(start_rc[1])
    gr, gc = float(goal_rc[0]), float(goal_rc[1])
    d_straight_px = float(np.sqrt((gr - sr) ** 2 + (gc - sc) ** 2))
    d_straight_m = d_straight_px * pixel_size_m
    detour_ratio = (path_length_m / d_straight_m) if d_straight_m > 0 else np.nan

    turn_count = _count_turns(path_rc)
    turns_per_m = (turn_count / path_length_m) if path_length_m > 0 else np.nan

    # safety 
    path_haz_mean = float(np.mean(Hf)) if Hf.size else np.nan
    path_haz_max = float(np.max(Hf)) if Hf.size else np.nan
    path_haz_p95 = float(np.percentile(Hf, 95)) if Hf.size else np.nan
    path_haz_sum = float(np.sum(Hf)) if Hf.size else np.nan
    haz_per_m = (path_haz_sum / path_length_m) if (np.isfinite(path_haz_sum) and path_length_m > 0) else np.nan

    frac_block = float(np.mean(Hf >= hazard_block)) if Hf.size else np.nan

    # Compare to global mean hazard (the lower ratio the better)
    hazard_mean_ratio = (path_haz_mean / global_haz_mean) if (np.isfinite(path_haz_mean) and global_haz_mean > 0) else np.nan

    # Bounded safety score (higher better)
    safety_score = float(1.0 / (1.0 + haz_per_m)) if np.isfinite(haz_per_m) else np.nan

    # efficiency 
    cost_sum_along_path = float(np.sum(Cf)) if Cf.size else np.nan
    cost_per_m = (cost_sum_along_path / path_length_m) if (np.isfinite(cost_sum_along_path) and path_length_m > 0) else np.nan

    # Meta-reported planner results (for sanity checks and or reporting)
    res = nav.get("result", {})
    planner_total_cost = float(res.get("total_cost", np.nan)) if np.isfinite(res.get("total_cost", np.nan)) else np.inf
    expansions = int(res.get("expansions", -1))
    path_len_reported = int(res.get("path_len", -1))

    # Corridor experiment summary (baseline capability curve)
    corridor_trials = nav.get("corridor_experiment", [])
    corridor_summary = []
    for t in corridor_trials:
        corridor_summary.append({
            "radius_px": int(t.get("corridor_radius_px", -1)),
            "success": bool(t.get("success", False)),
            "expansions": int(t.get("expansions", -1)),
            "path_len": int(t.get("path_len", 0)),
            "total_cost": t.get("total_cost", None),
            "path_hazard_mean": t.get("path_hazard_mean", None),
            "path_hazard_max": t.get("path_hazard_max", None),
        })

    return {
        "run_id": {
            "run_name": meta.get("run_name"),
            "tile_id": meta.get("tile_id"),
            "roi": meta.get("roi"),
            "shape": meta.get("shape"),
        },
        "settings": {
            "connectivity": connectivity,
            "cost_model": {
                "alpha": alpha,
                "hazard_block": hazard_block,
                "block_cost": block_cost,
            },
            "hazard_assessor": nav["hazard_assessor"],
            "pathfinder": nav["pathfinder"],
        },
        "geometry": {
            "path_nodes": int(path_rc.shape[0]),
            "path_length_px": path_length_px,
            "path_length_m": path_length_m,
            "straight_line_m": d_straight_m,
            "detour_ratio": detour_ratio,
            "turn_count": turn_count,
            "turns_per_m": turns_per_m,
        },
        "safety": {
            "path_hazard_mean": path_haz_mean,
            "path_hazard_max": path_haz_max,
            "path_hazard_p95": path_haz_p95,
            "haz_per_m": haz_per_m,
            "frac_hazard_ge_block": frac_block,
            "global_hazard_mean": global_haz_mean,
            "hazard_mean_ratio": hazard_mean_ratio,
            "safety_score": safety_score,
        },
        "efficiency": {
            "cost_sum_along_path": cost_sum_along_path,
            "cost_per_m": cost_per_m,
            "planner_total_cost": planner_total_cost,
            "expansions": expansions,
            "path_len_reported": path_len_reported,
        },
        "corridor_experiment": corridor_summary,
        "notes": {
            "sanity_checks": {
                "global_hazard_min": global_haz_min,
                "global_hazard_max": global_haz_max,
                "meta_success": bool(res.get("success", False)),
            }
        }
    }

# store metrics
def save_metrics(run_dir: str | Path, out_name: str = "metrics.json") -> Path:
    """Writes evaluation metrics to run_dir/out_name and returns the path."""
    run_dir = Path(run_dir)
    metrics = evaluate_run(run_dir)
    out_path = run_dir / out_name
    out_path.write_text(json.dumps(metrics, indent=2))
    return out_path
