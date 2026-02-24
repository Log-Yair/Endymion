# -*- coding: utf-8 -*-
"""
Endymion Prototype : Safety Masks + Weighted A* (Pathfinder)

Adds:
1) Hazard blocking mask (hazard >= hazard_block -> blocked).
2) Weighted A* (f = g + w*h) to reduce expansions.

Safety note:
- Weighted A* affects efficiency / optimality, not safety.
- Safety is enforced via masks (blocked cells / corridor restrictions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Any
import heapq
import numpy as np

RC = Tuple[int, int]  # (row, col)


def make_corridor_mask(shape: Tuple[int, int], start: RC, goal: RC, radius_px: int) -> np.ndarray:
    """
    Create a corridor/tube mask around the straight-line segment start->goal.

    Returns
    -------
    mask : np.ndarray (H,W) bool
        True  = allowed search region
        False = ignored by corridor-restricted A*
    """
    H, W = shape
    (r0, c0) = start
    (r1, c1) = goal

    rr, cc = np.indices((H, W), dtype=np.float32)

    # Use x=col, y=row for standard geometry formulas
    x0, y0 = float(c0), float(r0)
    x1, y1 = float(c1), float(r1)

    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy

    if denom == 0.0:
        # start == goal: corridor becomes a disk around the start point
        dist2 = (cc - x0) ** 2 + (rr - y0) ** 2
        return dist2 <= float(radius_px * radius_px)

    # Project each cell onto the segment using parameter t in [0,1]
    t = ((cc - x0) * dx + (rr - y0) * dy) / denom
    t = np.clip(t, 0.0, 1.0)

    x_closest = x0 + t * dx
    y_closest = y0 + t * dy

    dist2 = (cc - x_closest) ** 2 + (rr - y_closest) ** 2
    return dist2 <= float(radius_px * radius_px)


def build_cost_from_hazard(
    hazard: np.ndarray,
    alpha: float = 10.0,
    hazard_block: float = 0.95,
    block_cost: float = 1e6,
) -> np.ndarray:
    """
    Convert hazard [0..1] into a traversal cost grid.

    cost = 1 + alpha * hazard

    Cells with hazard >= hazard_block become blocked (cost = block_cost).
    """
    if hazard.ndim != 2:
        raise ValueError("hazard must be a 2D raster.")
    if np.isnan(hazard).any():
        raise ValueError("hazard contains NaNs; fix/replace before building cost.")

    cost = (1.0 + float(alpha) * hazard).astype(np.float32)
    cost[hazard >= float(hazard_block)] = float(block_cost)
    return cost


@dataclass
class Pathfinder:
    """
    A* / Weighted A* over a 2D cost raster.

    Parameters
    ----------
    connectivity : int
        4 or 8 neighbours
    block_cost : float
        Any cell with cost >= block_cost is treated as blocked
    heuristic_weight : float
        1.0 = standard A* (optimal)
        >1.0 = weighted A* (faster, may be slightly suboptimal)
    """
    connectivity: int = 8
    block_cost: float = 1e6
    heuristic_weight: float = 1.2

    def find_path(
        self,
        cost: np.ndarray,
        start: RC,
        goal: RC,
        allowed_mask: Optional[np.ndarray] = None
    ) -> Dict[str, object]:
        """Compute a least-cost path from start to goal on a cost grid."""
        self._validate_inputs(cost, start, goal)

        if self.heuristic_weight < 1.0:
            raise ValueError("heuristic_weight must be >= 1.0 (1.0 = standard A*).")

        # Corridor mask sanity checks (optional feature)
        if allowed_mask is not None:
            if allowed_mask.shape != cost.shape:
                raise ValueError("allowed_mask must match cost shape.")
            if allowed_mask.dtype != bool:
                allowed_mask = allowed_mask.astype(bool)

            # If start/goal are not inside corridor, fail immediately
            if (not allowed_mask[start]) or (not allowed_mask[goal]):
                return {
                    "success": False,
                    "path_rc": [],
                    "total_cost": float("inf"),
                    "meta": {"reason": "start_or_goal_outside_corridor"},
                }

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

        # Parent pointers
        parent_r = np.full((H, W), -1, dtype=np.int32)
        parent_c = np.full((H, W), -1, dtype=np.int32)

        # Closed set
        closed = np.zeros((H, W), dtype=bool)

        # Priority queue entries: (f, g, r, c)
        pq: List[Tuple[float, float, int, int]] = []
        f0 = self.heuristic_weight * self._heuristic(start, goal)
        heapq.heappush(pq, (float(f0), 0.0, sr, sc))

        expansions = 0

        while pq:
            f_cur, g_cur, r, c = heapq.heappop(pq)

            if closed[r, c]:
                continue
            closed[r, c] = True
            expansions += 1

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

            for nr, nc, step_dist in self._neighbours(r, c, H, W):

                # Corridor restriction MUST be checked before exploring
                if allowed_mask is not None and not allowed_mask[nr, nc]:
                    continue

                if closed[nr, nc]:
                    continue
                if cost[nr, nc] >= self.block_cost:
                    continue

                move_cost = 0.5 * (cost[r, c] + cost[nr, nc]) * step_dist
                tentative = g[r, c] + move_cost

                if tentative < g[nr, nc]:
                    g[nr, nc] = tentative
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c

                    h = self._heuristic((nr, nc), goal)
                    f = tentative + self.heuristic_weight * h
                    heapq.heappush(pq, (float(f), float(tentative), nr, nc))

        return {
            "success": False,
            "path_rc": [],
            "total_cost": float("inf"),
            "meta": {"reason": "no_path", "expansions": expansions},
        }

    def find_path_corridor(
        self,
        cost: np.ndarray,
        start: RC,
        goal: RC,
        corridor_radii: Sequence[int],
        hazard: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Try progressively wider corridor masks until a path is found.

        Returns:
        - best: best path result (or None)
        - best_radius_px: radius that worked (or None)
        - experiment: list of per-radius logs
        """
        H, W = cost.shape
        experiment: List[Dict[str, Any]] = []

        best = None
        best_rad = None

        for rad in corridor_radii:
            allowed = make_corridor_mask((H, W), start, goal, radius_px=int(rad))

            # Fail fast if corridor doesn't contain endpoints
            if not allowed[start] or not allowed[goal]:
                experiment.append({
                    "corridor_radius_px": int(rad),
                    "success": False,
                    "expansions": None,
                    "path_len": 0,
                    "total_cost": float("inf"),
                    "reason": "start_or_goal_outside_corridor",
                })
                continue

            res = self.find_path(cost, start, goal, allowed_mask=allowed)

            row: Dict[str, Any] = {
                "corridor_radius_px": int(rad),
                "success": bool(res.get("success", False)),
                "expansions": res.get("meta", {}).get("expansions", None),
                "path_len": len(res.get("path_rc", [])),
                "total_cost": float(res.get("total_cost", float("inf"))),
            }

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

    # -------------------------
    # Helpers
    # -------------------------

    def _heuristic(self, a: RC, b: RC) -> float:
        """Distance-only heuristic (octile for 8-connectivity, Manhattan for 4)."""
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