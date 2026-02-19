from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Tuple, List

# 8-neighborhood offsets (D8)
NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def resolve_flats_barnes_2014(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    equal_tol: float = 0.0,
    lower_tol: float = 0.0,
    apply_to_dem: bool = False,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Flat resolution for flow-routing based on Barnes et al. (2014).

    The procedure:
      1) Identify NOFLOW cells (cells without a strictly lower neighbor).
      2) Detect HighEdges: NOFLOW cells adjacent to higher terrain.
      3) Detect LowEdges: flow-defined cells adjacent to equal-elevation NOFLOW cells;
         seed the corresponding NOFLOW neighbor cells as flat-entry points.
      4) Label drainable flats by flood-filling from seeds over equal-elevation NOFLOW cells.
      5) Compute two within-flat distance fields:
           - away-from-higher (from HighEdges)
           - towards-lower (from low-edge seeds)
      6) Combine into an integer FlatMask that imposes a monotone drainage gradient.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM array.
    nodata : float
        NoData marker (use np.nan if NoData is NaN).
    equal_tol : float
        Equality tolerance for flat membership (0.0 for strict equality).
    lower_tol : float
        Strictness for the "lower" test (0.0 for strict).
    apply_to_dem : bool
        If True, add epsilon * FlatMask to DEM on flat cells (optional).
    epsilon : float
        Increment magnitude used only when apply_to_dem=True.

    Returns
    -------
    dem_out : np.ndarray
        DEM (optionally modified on flat cells), float64.
    flat_mask : np.ndarray
        Integer FlatMask increments (int32), 0 outside flats.
    labels : np.ndarray
        Flat labels (int32), 0 outside flats.
    stats : dict
        Summary counters.
    """
    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    n_rows, n_cols = dem_values.shape

    # Valid-data mask
    if np.isnan(nodata):
        valid_mask = np.isfinite(dem_values)
    else:
        valid_mask = (dem_values != nodata) & np.isfinite(dem_values)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < n_rows and 0 <= c < n_cols

    def is_equal(a: float, b: float) -> bool:
        return abs(a - b) <= equal_tol

    def is_lower(a: float, b: float) -> bool:
        return (b - a) < -lower_tol

    # Step 0: NOFLOW / HASFLOW classification
    has_lower = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            if not valid_mask[r, c]:
                continue
            z0 = dem_values[r, c]
            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                    continue
                if is_lower(z0, dem_values[nr, nc]):
                    has_lower[r, c] = True
                    break

    NOFLOW = valid_mask & (~has_lower)
    HASFLOW = valid_mask & has_lower

    # Step 1: Edge detection
    high_edges: List[Tuple[int, int]] = []
    low_seeds: List[Tuple[int, int]] = []

    for r in range(n_rows):
        for c in range(n_cols):
            if not valid_mask[r, c]:
                continue
            z0 = dem_values[r, c]

            if NOFLOW[r, c]:
                # HighEdges: NOFLOW cells adjacent to higher terrain
                for dr, dc in NEIGHBOR_OFFSETS_8:
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                        continue
                    if (dem_values[nr, nc] - z0) > equal_tol:
                        high_edges.append((r, c))
                        break

            elif HASFLOW[r, c]:
                # LowEdges: flow-defined cells adjacent to equal-elevation NOFLOW cells
                # Seeds are the adjacent NOFLOW cells (flat entry points)
                for dr, dc in NEIGHBOR_OFFSETS_8:
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                        continue
                    if NOFLOW[nr, nc] and is_equal(z0, dem_values[nr, nc]):
                        low_seeds.append((nr, nc))

    if len(low_seeds) == 0:
        dem_out = dem_values.copy()
        dem_out[~valid_mask] = (np.nan if np.isnan(nodata) else nodata)
        flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
        labels = np.zeros((n_rows, n_cols), dtype=np.int32)
        stats = {
            "n_flats": 0,
            "n_flat_cells": 0,
            "n_low_seeds": 0,
            "n_high_edges": int(len(high_edges)),
        }
        return dem_out, flat_mask, labels, stats

    # Deduplicate seeds
    low_seeds = list(dict.fromkeys(low_seeds))

    # Step 1b: Flat labeling (flood-fill from seeds over equal-elevation NOFLOW cells)
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    label_id = 0

    for sr, sc in low_seeds:
        if labels[sr, sc] != 0:
            continue
        if not NOFLOW[sr, sc]:
            continue

        label_id += 1
        flat_elev = dem_values[sr, sc]
        q = deque([(sr, sc)])
        labels[sr, sc] = label_id

        while q:
            r, c = q.popleft()
            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                    continue
                if labels[nr, nc] != 0:
                    continue
                if not NOFLOW[nr, nc]:
                    continue
                if not is_equal(dem_values[nr, nc], flat_elev):
                    continue

                labels[nr, nc] = label_id
                q.append((nr, nc))

    if label_id == 0:
        dem_out = dem_values.copy()
        dem_out[~valid_mask] = (np.nan if np.isnan(nodata) else nodata)
        flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
        stats = {
            "n_flats": 0,
            "n_flat_cells": 0,
            "n_low_seeds": int(len(low_seeds)),
            "n_high_edges": int(len(high_edges)),
        }
        return dem_out, flat_mask, labels, stats

    # Keep only drainable HighEdges (belonging to labeled flats)
    high_edges = [(r, c) for (r, c) in high_edges if labels[r, c] != 0]

    # Step 2: Away-from-higher distance (within each flat)
    away_dist = np.full((n_rows, n_cols), -1, dtype=np.int32)
    flat_height = np.zeros(label_id + 1, dtype=np.int32)

    q = deque()
    for r, c in high_edges:
        away_dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        lbl = labels[r, c]
        if lbl == 0:
            continue

        away_level = away_dist[r, c] + 1
        if away_level > flat_height[lbl]:
            flat_height[lbl] = away_level

        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                continue
            if labels[nr, nc] != lbl:
                continue
            if not NOFLOW[nr, nc]:
                continue
            if away_dist[nr, nc] != -1:
                continue

            away_dist[nr, nc] = away_dist[r, c] + 1
            q.append((nr, nc))

    # Step 3: Towards-lower distance and FlatMask combination
    towards_dist = np.full((n_rows, n_cols), -1, dtype=np.int32)
    flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)

    q = deque()
    for r, c in low_seeds:
        lbl = labels[r, c]
        if lbl == 0:
            continue
        towards_dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        lbl = labels[r, c]
        if lbl == 0:
            continue

        towards_level = towards_dist[r, c] + 1

        if away_dist[r, c] != -1:
            away_level = away_dist[r, c] + 1
            flat_mask[r, c] = (flat_height[lbl] - away_level) + (2 * towards_level)
        else:
            flat_mask[r, c] = (2 * towards_level)

        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not valid_mask[nr, nc]:
                continue
            if labels[nr, nc] != lbl:
                continue
            if not NOFLOW[nr, nc]:
                continue
            if towards_dist[nr, nc] != -1:
                continue

            towards_dist[nr, nc] = towards_dist[r, c] + 1
            q.append((nr, nc))

    # Optional elevation perturbation
    dem_out = dem_values.copy()
    if apply_to_dem:
        apply = (labels > 0) & valid_mask & (flat_mask > 0)
        dem_out[apply] = dem_out[apply] + epsilon * flat_mask[apply]

    # Preserve NoData
    if np.isnan(nodata):
        dem_out[~valid_mask] = np.nan
    else:
        dem_out[~valid_mask] = nodata

    stats: Dict[str, int] = {
        "n_flats": int(label_id),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_low_seeds": int(len(low_seeds)),
        "n_high_edges": int(len(high_edges)),
        "apply_to_dem": int(apply_to_dem),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
    }

    return dem_out, flat_mask.astype(np.int32), labels, stats
