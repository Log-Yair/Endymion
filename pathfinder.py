
# -*- coding: utf-8 -*-
"""
Pathfinder for Endymion (Prototype)

Computes a least-cost path over a raster "cost" grid using A*.
- No NetworkX graph construction (memory-friendly)
- Supports 4- or 8-neighbour moves
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np

RC = Tuple[int, int]  # (row, col)


@dataclass
class Pathfinder:
    """
    A* pathfinder over a 2D cost raster.
    """
    connectivity: int = 8          # 4 or 8
    block_cost: float = 1e6        # cells >= this are treated as blocked

    def find_path(
        self,
        cost: np.ndarray,
        start: RC,
        goal: RC,
    ) -> Dict[str, object]:
        """
        Returns:
          - path_rc: list of (r,c) from start->goal (inclusive)
          - success: bool
          - total_cost: float
          - meta: diagnostics
        """
        self._validate_inputs(cost, start, goal)

        H, W = cost.shape # dimensions
        sr, sc = start # start row,col
        gr, gc = goal # goal row,col

        # Early exit if start/goal blocked
        if cost[sr, sc] >= self.block_cost or cost[gr, gc] >= self.block_cost:
            return {
                "success": False,
                "path_rc": [],
                "total_cost": float("inf"),
                "meta": {"reason": "start_or_goal_blocked"},
            }

        # A* bookkeeping arrays
        g = np.full((H, W), np.inf, dtype=np.float32)     # best-known cost-to-come
        parent_r = np.full((H, W), -1, dtype=np.int32)    # parent pointers
        parent_c = np.full((H, W), -1, dtype=np.int32)    
        closed = np.zeros((H, W), dtype=bool)

        g[sr, sc] = 0.0

        # Priority queue entries: (f, g, r, c)
        pq: List[Tuple[float, float, int, int]] = []    # min-heap
        heapq.heappush(pq, (self._heuristic(start, goal), 0.0, sr, sc))

        expansions = 0 # number of expanded nodes

        while pq:      
            f_cur, g_cur, r, c = heapq.heappop(pq) # current cell

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
                    },
                }

            # Explore neighbours
            for nr, nc, step_dist in self._neighbours(r, c, H, W):
                if closed[nr, nc]:
                    continue
                if cost[nr, nc] >= self.block_cost:
                    continue  # blocked cell

                # Move cost: average cell cost * step distance (keeps diagonal sensible)
                move_cost = 0.5 * (cost[r, c] + cost[nr, nc]) * step_dist
                tentative = g[r, c] + move_cost

                if tentative < g[nr, nc]:
                    g[nr, nc] = tentative
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c

                    h = self._heuristic((nr, nc), goal)
                    heapq.heappush(pq, (float(tentative + h), float(tentative), nr, nc))

        # No route found
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
        Admissible heuristic for grid movement.
        For 8-connectivity, octile distance is a good choice.
        For 4-connectivity, Manhattan is fine.
        """
        (r0, c0), (r1, c1) = a, b # unpack
        dr = abs(r1 - r0) # row difference
        dc = abs(c1 - c0) # col difference

        if self.connectivity == 4:  # Manhattan distance
            return float(dr + dc)

        # 8-connected (octile distance)
        dmin = min(dr, dc)
        dmax = max(dr, dc)
        return float((dmax - dmin) + np.sqrt(2.0) * dmin)
    
    def _neighbours(self, r: int, c: int, H: int, W: int):
        """
        Yield neighbour cells plus step distance (1 for cardinal, sqrt(2) for diagonal).
        """
        # Cardinal moves
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

        # Diagonals (optional)
        if self.connectivity == 8:
            moves += [(-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
                      (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0))]

        for dr, dc, dist in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                yield nr, nc, float(dist)

    @staticmethod
    def _reconstruct_path(parent_r: np.ndarray, parent_c: np.ndarray, start: RC, goal: RC) -> List[RC]:
        """
        Reconstruct path from parent pointers (goal -> start).
        """
        path: List[RC] = []
        r, c = goal
        sr, sc = start

        # Walk backwards from goal to start
        while not (r == sr and c == sc):
            path.append((r, c))
            pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
            if pr < 0 or pc < 0:
                # parent missing => reconstruction failed
                return []
            r, c = pr, pc

        path.append((sr, sc))
        path.reverse()
        return path

    @staticmethod 
    def _validate_inputs(cost: np.ndarray, start: RC, goal: RC) -> None:
        """Basic sanity checks."""
        if not isinstance(cost, np.ndarray) or cost.ndim != 2:
            raise ValueError("cost must be a 2D numpy array.")
        H, W = cost.shape
        for name, (r, c) in [("start", start), ("goal", goal)]:
            if not (0 <= r < H and 0 <= c < W):
                raise ValueError(f"{name} {r,c} out of bounds for cost shape {cost.shape}.")
        if np.isnan(cost).any():
            raise ValueError("cost contains NaNs; fill or mask them before pathfinding.")