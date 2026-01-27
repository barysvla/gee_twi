from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Tuple

# Neighbor offsets for a full 3x3 neighborhood (including diagonals)
NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def resolve_flats_barnes_tie(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    epsilon: float = 2e-5,
    equal_tol: float = 3e-3,
    lower_tol: float = 0.0,
    treat_oob_as_lower: bool = True,
    require_low_edge_only: bool = True,
    force_all_flats: bool = False,
    include_equal_ties: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Resolve flat areas (plateaus) by imposing a tiny monotone gradient, following
    the Barnes / Garbrecht & Martz approach with two BFS passes.

    Key idea:
    - Identify plateau regions (flat candidates) and label them.
    - Build "high edges" (adjacent to higher terrain) and "low edges" (adjacent to lower terrain).
    - Compute two distance transforms within each plateau:
        A) away-from-higher (push flow away from higher rim)
        B) towards-lower (pull flow towards outlets; dominant)
    - Combine both fields into FlatMask and add epsilon * FlatMask to DEM on plateau cells.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM array.
    nodata : float
        NoData marker; use np.nan if NoData is represented as NaN.
    epsilon : float
        Small elevation increment applied per FlatMask unit (creates a consistent gradient).
    equal_tol : float
        Tolerance for treating elevations as "equal" when labeling plateaus and detecting edges.
    lower_tol : float
        Strict tolerance for the "lower" test; dz < -lower_tol means strictly lower.
    treat_oob_as_lower : bool
        If True, treat out-of-bounds or invalid neighbors as lower (acts as an outlet at boundaries).
    require_low_edge_only : bool
        If True, only resolve plateaus that have at least one low edge (i.e., are drainable).
    force_all_flats : bool
        If True, attempt to resolve even closed plateaus by seeding perimeter as low edges.
    include_equal_ties : bool
        If True, plateau labeling may absorb equal-elevation neighbors even if they are not "true flats".

    Returns
    -------
    dem_out : np.ndarray
        DEM with epsilon*FlatMask applied on plateau cells (float64).
    flat_mask : np.ndarray
        Integer field that encodes combined "away" and "towards" distances (int32).
    labels : np.ndarray
        Plateau labels (int32), 0 = non-plateau.
    stats : dict
        Summary statistics and used parameters.
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

    # 1) Precompute: does a cell have a strictly-lower neighbor?
    has_lower_neighbor = np.zeros_like(valid_mask, dtype=bool)
    for dr, dc in NEIGHBOR_OFFSETS_8:
        r0, r1 = max(0, -dr), min(n_rows, n_rows - dr)
        c0, c1 = max(0, -dc), min(n_cols, n_cols - dc)
        if r0 >= r1 or c0 >= c1:
            continue

        a = dem_values[r0:r1, c0:c1]
        b = dem_values[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        v = valid_mask[r0:r1, c0:c1] & valid_mask[r0 + dr : r1 + dr, c0 + dc : c1 + dc]

        # b is strictly lower than a -> current cell has a lower neighbor
        has_lower_neighbor[r0:r1, c0:c1] |= v & ((b - a) < -lower_tol)

    # Candidate plateau cells: valid and no strictly-lower neighbor
    plateau_candidates = valid_mask & (~has_lower_neighbor)

    # 2) Plateau labeling (single pass BFS)
    labels = np.zeros_like(valid_mask, dtype=np.int32)
    label_id = 0

    for r in range(n_rows):
        for c in range(n_cols):
            if not plateau_candidates[r, c] or labels[r, c] != 0:
                continue

            label_id += 1
            q = deque([(r, c)])
            labels[r, c] = label_id

            while q:
                cr, cc = q.popleft()
                z_cur = dem_values[cr, cc]

                for dr, dc in NEIGHBOR_OFFSETS_8:
                    nr, nc = cr + dr, cc + dc
                    if (not in_bounds(nr, nc)) or (not valid_mask[nr, nc]) or (labels[nr, nc] != 0):
                        continue

                    if abs(dem_values[nr, nc] - z_cur) > equal_tol:
                        continue

                    if (not include_equal_ties) and (not plateau_candidates[nr, nc]):
                        continue

                    labels[nr, nc] = label_id
                    q.append((nr, nc))

    if label_id == 0:
        dem_out = dem_values.copy()
        dem_out[~valid_mask] = (np.nan if np.isnan(nodata) else nodata)
        stats = {
            "n_flats": 0,
            "n_flats_active": 0,
            "n_flats_drainable": 0,
            "n_flat_cells": 0,
            "n_changed_cells": 0,
        }
        return dem_out, np.zeros_like(labels, dtype=np.int32), labels, stats

    # 3) Build plateau edges
    high_edges = [deque() for _ in range(label_id + 1)]
    low_edges = [deque() for _ in range(label_id + 1)]
    has_low_edge = np.zeros(label_id + 1, dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            lbl = labels[r, c]
            if lbl == 0:
                continue

            z0 = dem_values[r, c]
            adjacent_higher = False
            adjacent_lower = False
            touches_boundary = False

            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc

                # Boundary / invalid neighbor handling
                if (not in_bounds(nr, nc)) or (not valid_mask[nr, nc]):
                    touches_boundary = True
                    if treat_oob_as_lower:
                        adjacent_lower = True
                    continue

                if labels[nr, nc] == lbl:
                    continue

                dz = dem_values[nr, nc] - z0

                if dz > equal_tol:
                    adjacent_higher = True

                if dz < -lower_tol:
                    adjacent_lower = True
                elif abs(dz) <= equal_tol and has_lower_neighbor[nr, nc]:
                    # Cascade: an equal-elevation neighbor that itself has a lower neighbor
                    adjacent_lower = True

            if adjacent_lower:
                low_edges[lbl].append((r, c))
                has_low_edge[lbl] = True

            if adjacent_higher:
                high_edges[lbl].append((r, c))

            # Optional fallback for closed plateaus: treat perimeter cells as low edges
            if force_all_flats and (not adjacent_lower) and touches_boundary:
                low_edges[lbl].append((r, c))
                has_low_edge[lbl] = True

    # If forcing closed plateaus: seed at least some perimeter cells once
    if force_all_flats:
        for lbl in range(1, label_id + 1):
            if has_low_edge[lbl]:
                continue

            seeded = False
            for r in range(n_rows):
                for c in range(n_cols):
                    if labels[r, c] != lbl:
                        continue
                    for dr, dc in NEIGHBOR_OFFSETS_8:
                        nr, nc = r + dr, c + dc
                        if (not in_bounds(nr, nc)) or (not valid_mask[nr, nc]) or (labels[nr, nc] != lbl):
                            low_edges[lbl].append((r, c))
                            seeded = True
                            break
            if seeded:
                has_low_edge[lbl] = True

    def plateau_is_active(lbl: int) -> bool:
        if require_low_edge_only:
            return bool(has_low_edge[lbl])
        return bool(has_low_edge[lbl]) or force_all_flats

    # 4) Two BFS passes and combination into flat_mask
    away = np.full(labels.shape, -1, dtype=np.int32)
    towards = np.full(labels.shape, -1, dtype=np.int32)
    flat_mask = np.zeros_like(labels, dtype=np.int32)
    max_away_per_label = np.zeros(label_id + 1, dtype=np.int32)

    # A) Away-from-higher
    for lbl in range(1, label_id + 1):
        if not plateau_is_active(lbl):
            continue

        q = high_edges[lbl]
        if not q:
            continue

        for sr, sc in q:
            away[sr, sc] = 1

        while q:
            cr, cc = q.popleft()
            dist = away[cr, cc]
            if dist > max_away_per_label[lbl]:
                max_away_per_label[lbl] = dist

            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < n_rows
                    and 0 <= nc < n_cols
                    and labels[nr, nc] == lbl
                    and away[nr, nc] == -1
                ):
                    away[nr, nc] = dist + 1
                    q.append((nr, nc))

    # B) Towards-lower (dominant) + combine
    n_drainable = 0
    n_active = 0

    for lbl in range(1, label_id + 1):
        if not plateau_is_active(lbl):
            continue

        q = low_edges[lbl]
        if not q:
            continue

        n_active += 1
        if has_low_edge[lbl]:
            n_drainable += 1

        for sr, sc in q:
            towards[sr, sc] = 1

        while q:
            cr, cc = q.popleft()
            dist = towards[cr, cc]

            if away[cr, cc] != -1:
                # Combine both fields; towards dominates
                flat_mask[cr, cc] = (max_away_per_label[lbl] - away[cr, cc]) + 2 * dist
            else:
                flat_mask[cr, cc] = 2 * dist

            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < n_rows
                    and 0 <= nc < n_cols
                    and labels[nr, nc] == lbl
                    and towards[nr, nc] == -1
                ):
                    towards[nr, nc] = dist + 1
                    q.append((nr, nc))

    # 5) Apply epsilon * flat_mask on plateau cells
    dem_out = dem_values.copy()
    apply_mask = (labels > 0) & valid_mask & (flat_mask != 0)
    dem_out[apply_mask] = dem_out[apply_mask] + epsilon * flat_mask[apply_mask]

    if np.isnan(nodata):
        dem_out[~valid_mask] = np.nan
    else:
        dem_out[~valid_mask] = nodata

    stats: Dict[str, int] = {
        "n_flats": int(label_id),
        "n_flats_active": int(n_active),
        "n_flats_drainable": int(n_drainable),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_changed_cells": int(np.count_nonzero(apply_mask)),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "require_low_edge_only": bool(require_low_edge_only),
        "force_all_flats": bool(force_all_flats),
        "include_equal_ties": bool(include_equal_ties),
        "epsilon": float(epsilon),
    }

    return dem_out, flat_mask.astype(np.int32), labels, stats
