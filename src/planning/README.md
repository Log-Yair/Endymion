
# Endymion Planning Module

## Overview

The `src/planning/` module is responsible for generating hazard-aware navigation paths across the ROI terrain grid.

It converts the hazard map into a traversal-cost grid and then uses pathfinding to compute a route between a defined start and goal location.

---

## Files

```text
src/planning/
│
└── pathfinder.py
````

---

## Role in the Pipeline

```text
Hazard map
    ↓
Cost grid
    ↓
Pathfinder
    ↓
Path coordinates + planner metadata
```

---

## Purpose

The planning module does not simply look for the shortest geometric route.

Instead, it searches for a route that balances:

* distance
* terrain hazard
* blocked / unsafe cells
* search efficiency

This allows Endymion to support hazard-aware navigation rather than basic shortest-path planning.

---

## `pathfinder.py`

## Main Components

### `build_cost_from_hazard`

Converts a hazard map into a traversal-cost grid.

```text
cost = 1 + alpha * hazard
```

In LaTeX:

```markdown
$$
C = 1 + \alpha H
$$
```

Where:

* `C` = traversal cost
* `H` = hazard value
* `alpha` = hazard influence factor

---

## Blocking Rule

Cells above the hazard blocking threshold are treated as blocked.

```text
hazard >= hazard_block → blocked
```

In LaTeX:

```markdown
$$
C =
\begin{cases}
C_{block}, & H \geq H_{block} \\
1 + \alpha H, & H < H_{block}
\end{cases}
$$
```

This prevents the planner from routing through terrain that is considered too hazardous.

---

## Pathfinder

### Algorithm

The module implements **Weighted A*** over a 2D raster grid.

Weighted A* modifies the normal A* priority function:

```markdown
$$
f(n) = g(n) + w \cdot h(n)
$$
```

Where:

* `g(n)` = cost from start to current node
* `h(n)` = heuristic estimate from current node to goal
* `w` = heuristic weight
* `f(n)` = priority score

A higher heuristic weight can reduce the number of explored nodes, but may produce a slightly less optimal path.

---

## Connectivity

The planner supports grid-based movement.

Typical final setting:

```text
connectivity = 8
```

This allows movement in:

* horizontal directions
* vertical directions
* diagonal directions

Diagonal movement helps produce more natural routes across raster terrain.

---

## Movement Cost

The movement cost between neighbouring cells is based on the average cost of the current and next cell.

```markdown
$$
move\_cost = \frac{C_{current} + C_{next}}{2} \cdot d
$$
```

Where:

* `C_current` = cost of current cell
* `C_next` = cost of neighbouring cell
* `d` = movement distance

  * `1` for horizontal/vertical moves
  * `sqrt(2)` for diagonal moves

---

## Corridor-Aware Search

The module also supports optional corridor-restricted planning.

A corridor mask limits the search area to a tube around the straight-line path between start and goal.

```text
full grid search → corridor-limited search
```

This helps reduce unnecessary expansions while keeping the planner focused on a plausible route region.

---

## Outputs

A successful pathfinding run returns:

```text
success
path_rc
total_cost
meta
```

### `path_rc`

List of route coordinates:

```text
(row, col)
```

### `total_cost`

Planner cost accumulated along the route.

### `meta`

Metadata such as:

```text
expansions
connectivity
heuristic_weight
used_corridor_radius
```

---

## Common Failure Cases

### `start_or_goal_blocked`

Occurs when the start or goal cell has a cost greater than or equal to the blocking cost.

### `start_or_goal_outside_corridor`

Occurs when corridor search is enabled but the start or goal lies outside the allowed corridor mask.

### `no_path`

Occurs when no valid route exists between start and goal under the current cost and blocking rules.

---

## Relationship to Hazard Modelling

The pathfinder does not decide what is dangerous.

That responsibility belongs to the hazard model.

The planning module only interprets the hazard map through a traversal-cost grid.

```text
HazardAssessor defines risk
Pathfinder searches through that risk
Evaluator measures the result
```

---

## Relationship to Evaluation

The planner output is later evaluated by `Evaluator`.

The evaluator measures:

* path length
* mean path hazard
* maximum path hazard
* safety score
* cost per metre
* total planner cost
* expansions

This makes it possible to analyse whether a route is not only successful, but also safe and efficient.

---

## Design Notes

### Why Weighted A*?

Weighted A* was selected because it provides a practical balance between route quality and computational efficiency.

Compared with standard A*, it can reduce search effort by increasing the influence of the heuristic.

### Why Use Hazard Cost?

A shortest path across unsafe terrain is not useful for navigation.

By converting hazard into traversal cost, the planner is encouraged to avoid dangerous terrain even if the resulting path is slightly longer.

### Why Keep Blocking Separate?

Blocking prevents extremely hazardous terrain from being treated as merely expensive.

This is useful for enforcing hard safety constraints.

---

## Limitations

* The planner operates on a 2D raster grid.
* It does not model rover dynamics.
* It does not include energy consumption or wheel-soil interaction.
* Safety depends on the quality of the hazard map.
* Weighted A* may trade optimality for faster search.

---

## Summary

The `src/planning/` module converts Endymion’s hazard model into an actionable navigation route.

It provides:

* hazard-to-cost conversion
* blocked-cell handling
* Weighted A* pathfinding
* optional corridor search
* planner metadata for evaluation

This module is the bridge between hazard assessment and route generation.

```
```
