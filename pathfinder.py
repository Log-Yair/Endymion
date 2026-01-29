# -*- coding: utf-8 -*-
"""
Endymion Prototype : Safety Masks + Weighted A* (Pathfinder)

This patch does TWO things:
1) Adds a second "safety" mask based on hazard (e.g., hazard >= 0.95 is blocked).
2) Adds Weighted A* to drastically reduce expansions:
   - Standard A*:    f = g + h
   - Weighted A*:    f = g + w*h   (w > 1 speeds up search, may be slightly suboptimal)

Important safety note:
- Weighted A* does NOT make paths unsafe by itself.
- Safety comes from the masks (blocked cells). Weighted A* only changes search efficiency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Any
import heapq
import numpy as np


RC = Tuple[int, int]  # (row, col)


def make_corridor_mask(shape: Tuple[int, int], start: RC, goal: RC, radius_px: int) -> np.ndarray:
    """
    Create a corridor (tube) mask around the straight-line segment start->goal.

    True = allowed search region
    False = ignored by corridor-restricted A*
    """
    H, W = shape
    (r0, c0) = start
    (r1, c1) = goal

    rr, cc = np.indices((H, W), dtype=np.float32)

    x0, y0 = float(c0), float(r0)  # x=col, y=row
    x1, y1 = float(c1), float(r1)

    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy

    if denom == 0.0:
        dist2 = (cc - x0) ** 2 + (rr - y0) ** 2
        return dist2 <= float(radius_px * radius_px)

    t = ((cc - x0) * dx + (rr - y0) * dy) / denom
    t = np.clip(t, 0.0, 1.0)

    x_closest = x0 + t * dx
    y_closest = y0 + t * dy

    dist2 = (cc - x_closest) ** 2 + (rr - y_closest) ** 2
    return dist2 <= float(radius_px * radius_px)

# ============================================================
# 1) SECOND SAFETY MASK (hazard ceiling -> blocked cells)
# ============================================================

def build_cost_from_hazard(
    hazard: np.ndarray,
    alpha: float = 10.0,
    hazard_block: float = 0.95,
    block_cost: float = 1e6,
) -> np.ndarray:
    """
    Convert hazard [0..1] into a traversal cost grid, with a conservative safety block.

    Parameters: 
    
    hazard : np.ndarray
        Hazard raster in [0,1], float32, same shape as DEM patch.
    alpha : float
        Weight of hazard relative to distance (higher => avoid hazard more).
    hazard_block : float
        SECOND SAFETY MASK:
        Cells with hazard >= hazard_block are treated as non-traversable.
    block_cost : float
        Cost used to mark blocked cells.

    Returns:
    
    cost : np.ndarray
        Positive traversal cost grid, float32. Blocked cells set to block_cost.
    """
    if hazard.ndim != 2:
        raise ValueError("hazard must be a 2D raster.")
    if np.isnan(hazard).any():
        raise ValueError("hazard contains NaNs; fix/replace before building cost.")

    # Base cost: always positive (distance still matters via the +1.0)
    cost = (1.0 + float(alpha) * hazard).astype(np.float32)

    # Second safety mask: block very hazardous regions
    cost[hazard >= float(hazard_block)] = float(block_cost)

    return cost


# 2) PATHFINDER WITH WEIGHTED A*

@dataclass
class Pathfinder:
    """
    A* pathfinder over a 2D cost raster, with optional Weighted A*.

    Key additions:
    - heuristic_weight (w):
        w = 1.0 => standard A* (optimal, slower)
        w > 1.0 => weighted A* (faster, may be slightly suboptimal)
    - block_cost:
        any cell with cost >= block_cost is treated as blocked
    """
    connectivity: int = 8
    block_cost: float = 1e6
    heuristic_weight: float = 1.2  # set to 1.2 or 1.5 for speed

    def find_path(self, cost: np.ndarray, start: RC, goal: RC, allowed_mask: Optional[np.ndarray] = None) -> Dict[str, object]:

        if allowed_mask is not None:
            if allowed_mask.shape != cost.shape:
                raise ValueError("allowed_mask must match cost shape.")
            if allowed_mask.dtype != bool:
                allowed_mask = allowed_mask.astype(bool)
            if not allowed_mask[start] or not allowed_mask[goal]:
                return {"success": False, "path_rc": [], "total_cost": float("inf"),
                        "meta": {"reason": "start_or_goal_outside_corridor"}}
            
        """Compute least-cost path from start to goal over cost grid."""
        self._validate_inputs(cost, start, goal)
        if self.heuristic_weight < 1.0:
            raise ValueError("heuristic_weight must be >= 1.0 (1.0 = standard A*).")

        H, W = cost.shape
        sr, sc = start
        gr, gc = goal

        # If start/goal are blocked, fail early
        if cost[sr, sc] >= self.block_cost or cost[gr, gc] >= self.block_cost:
            return {
                "success": False,
                "path_rc": [],
                "total_cost": float("inf"),
                "meta": {"reason": "start_or_goal_blocked"},
            }

        # Best-known cost-to-come
        g = np.full((H, W), np.inf, dtype=np.float32)
        g[sr, sc] = 0.0

        # Parent pointers for reconstruction
        parent_r = np.full((H, W), -1, dtype=np.int32)
        parent_c = np.full((H, W), -1, dtype=np.int32)

        # Closed set: cells we have finalized
        closed = np.zeros((H, W), dtype=bool)

        # Priority queue entries: (f, g, r, c)
        pq: List[Tuple[float, float, int, int]] = []
        h0 = self._heuristic(start, goal)
        f0 = 0.0 + self.heuristic_weight * h0  # <-- Weighted A* here
        heapq.heappush(pq, (float(f0), 0.0, sr, sc))

        expansions = 0

        while pq:
            f_cur, g_cur, r, c = heapq.heappop(pq)

            # Skip stale entries
            if closed[r, c]:
                continue
            closed[r, c] = True
            expansions += 1

            # Goal reached
            if (r, c) == (gr, gc):
                path = self._reconstruct_path(parent_r, parent_c, start, goal)
                return {
                    "success": True,
                    "path_rc": path,
                    "total_cost": float(g[gr, gc]),
                    "meta": {
                        "expansions": expansions,
                        "connectivity": self.connectivity,
                        "heuristic_weight": self.heuristic_weight,
                    },
                }

            # Explore neighbours
            for nr, nc, step_dist in self._neighbours(r, c, H, W):
                if closed[nr, nc]:
                    continue
                if cost[nr, nc] >= self.block_cost:
                    continue  # blocked

                if allowed_mask is not None and not allowed_mask[nr, nc]:
                    continue  # outside allowed corridor

                # Move cost uses average of cell costs * step distance
                move_cost = 0.5 * (cost[r, c] + cost[nr, nc]) * step_dist
                tentative = g[r, c] + move_cost

                if tentative < g[nr, nc]:
                    g[nr, nc] = tentative
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c

                    h = self._heuristic((nr, nc), goal)
                    f = tentative + self.heuristic_weight * h  # <-- Weighted A* here
                    heapq.heappush(pq, (float(f), float(tentative), nr, nc))

                

        return {
            "success": False,
            "path_rc": [],
            "total_cost": float("inf"),
            "meta": {"reason": "no_path", "expansions": expansions},
        }

    # -------------------------
    # Helpers
    # -------------------------

    def _heuristic(self, a: RC, b: RC) -> float:
        """
        Distance-only heuristic.
        NOTE: It ignores hazard; safety is enforced via blocking masks.

        For 8-connectivity, octile distance; for 4-connectivity, Manhattan.
        """
        (r0, c0), (r1, c1) = a, b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        if self.connectivity == 4:
            return float(dr + dc)

        dmin = min(dr, dc)
        dmax = max(dr, dc)
        return float((dmax - dmin) + np.sqrt(2.0) * dmin)

    def _neighbours(self, r: int, c: int, H: int, W: int):
        """Yield neighbour coords + step distance (1 or sqrt(2))."""
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.connectivity == 8:
            rt2 = float(np.sqrt(2.0))
            moves += [(-1, -1, rt2), (-1, 1, rt2), (1, -1, rt2), (1, 1, rt2)]

        for dr, dc, dist in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield nr, nc, dist

    @staticmethod
    def _reconstruct_path(parent_r: np.ndarray, parent_c: np.ndarray, start: RC, goal: RC):
        """Reconstruct path by walking parents from goal back to start."""
        path: List[RC] = []
        r, c = goal
        sr, sc = start

        while not (r == sr and c == sc):
            path.append((r, c))
            pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
            if pr < 0 or pc < 0:
                return []
            r, c = pr, pc

        path.append((sr, sc))
        path.reverse()
        return path

    @staticmethod
    def _validate_inputs(cost: np.ndarray, start: RC, goal: RC) -> None:
        """Basic sanity checks for raster pathfinding."""
        if not isinstance(cost, np.ndarray) or cost.ndim != 2:
            raise ValueError("cost must be a 2D numpy array.")
        if np.isnan(cost).any():
            raise ValueError("cost contains NaNs; fill or mask them before pathfinding.")

        H, W = cost.shape
        for name, (r, c) in [("start", start), ("goal", goal)]:
            if not (0 <= r < H and 0 <= c < W):
                raise ValueError(f"{name} {r,c} out of bounds for cost shape {cost.shape}.")
            
    def find_path_corridor(
        self,
        cost: np.ndarray,
        start: RC,
        goal: RC,
        corridor_radii: Sequence[int],
        hazard: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Try A* inside progressively wider corridor masks until a path is found.

        Returns a dict with:
        - best: the best path result (or None)
        - best_radius_px: the corridor radius that worked (or None)
        - experiment: list of per-radius logs
        """
        H, W = cost.shape
        experiment: list[dict[str, Any]] = []

        best = None
        best_rad = None

        for rad in corridor_radii:
            allowed = make_corridor_mask((H, W), start, goal, radius_px=int(rad))

            # Skip radii where start/goal aren't in corridor
            if not allowed[start] or not allowed[goal]:
                experiment.append({
                    "corridor_radius_px": int(rad),
                    "success": False,
                    "reason": "start_or_goal_outside_corridor",
                })
                continue

            res = self.find_path(cost, start, goal, allowed_mask=allowed)

            row: dict[str, Any] = {
                "corridor_radius_px": int(rad),
                "success": bool(res.get("success", False)),
                "expansions": res.get("meta", {}).get("expansions", None),
                "path_len": len(res.get("path_rc", [])),
                "total_cost": float(res.get("total_cost", float("inf"))),
            }

            # Optional hazard diagnostics (nice for logs / report)
            if hazard is not None and row["success"] and row["path_len"] > 0:
                path_rc = np.asarray(res["path_rc"], dtype=np.int32)
                vals = hazard[path_rc[:, 0], path_rc[:, 1]]
                row["path_hazard_mean"] = float(np.mean(vals))
                row["path_hazard_max"] = float(np.max(vals))

            experiment.append(row)

            if row["success"] and row["path_len"] > 0:
                best = res
                best_rad = int(rad)
                break

        return {"best": best, "best_radius_px": best_rad, "experiment": experiment}    




