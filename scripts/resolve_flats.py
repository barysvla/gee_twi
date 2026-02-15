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

# 8-neighborhood offsets (D8)
OFFS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def resolve_flats_garbrecht_martz_1997(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    vertical_resolution: float = 1.0,
    equal_tol: float = 0.0,
    lower_tol: float = 0.0,
    treat_oob_as_lower: bool = True,
    max_exception_iters: int = 50,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Garbrecht & Martz (1997) flat-resolution algorithm.

    Implements:
      Step 1: Gradient towards lower terrain (backward growth from outlets).
      Step 2: Gradient away from higher terrain (growth from higher edges; previously
              incremented cells are incremented again each pass -> produces a reversed distance ramp).
      Step 3: Linear addition of both gradients, apply to DEM using an infinitesimal increment.
      Exceptional situation: If increments cancel, apply half-increment following Step 1 repeatedly
                             until no flat cell remains.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM.
    nodata : float
        NoData marker (np.nan supported).
    vertical_resolution : float
        Vertical DEM resolution vr (meters etc.). Paper uses increment = 2/100000 * vr.
    equal_tol : float
        Tolerance for treating elevations as equal when defining a flat surface.
        Set 0.0 for strict (closest to the paper).
    lower_tol : float
        Tolerance for strictly lower comparisons; dz < -lower_tol means strictly lower.
        Set 0.0 for strict.
    treat_oob_as_lower : bool
        Paper assumes boundary cells can drain outward. This models out-of-bounds as lower.
    max_exception_iters : int
        Safety cap for exceptional-case iterations.

    Returns
    -------
    dem_out : np.ndarray
        DEM modified by tiny increments (float64).
    fields : dict[str, np.ndarray]
        Diagnostic fields:
          - labels: int32 flat-surface labels (0 = not in any flat surface)
          - inc_towards: int32 increments from Step 1
          - inc_away: int32 increments from Step 2
          - inc_total: int32 total increments (Step 3, without half-fixes)
          - inc_half_added: int32 extra "half increments" applied in exceptional fix
    stats : dict[str, float]
        Summary stats and used increments.
    """
    Z = np.asarray(dem, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    nrows, ncols = Z.shape

    # Valid-data mask
    if np.isnan(nodata):
        valid = np.isfinite(Z)
    else:
        valid = (Z != nodata) & np.isfinite(Z)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < nrows and 0 <= c < ncols

    # --- Helper: strict-equality within tolerance ---
    def is_equal(a: float, b: float) -> bool:
        return abs(a - b) <= equal_tol

    # --- Helper: determine if a cell has a strictly lower neighbor (in a given surface) ---
    def has_strict_lower_neighbor(surface: np.ndarray, r: int, c: int, base_z: float) -> bool:
        for dr, dc in OFFS8:
            nr, nc = r + dr, c + dc
            if not inb(nr, nc) or (not valid[nr, nc]):
                if treat_oob_as_lower:
                    return True
                continue
            if surface[nr, nc] < base_z - lower_tol:
                return True
        return False

    # --- 1) Identify flat surfaces as connected components of equal elevation
    # A "flat surface" here means: a connected region of (approximately) equal elevation
    # that contains at least one cell without a strictly-lower neighbor AND has at least one outlet
    # (a cell in the region adjacent to strictly-lower terrain).
    labels = np.zeros((nrows, ncols), dtype=np.int32)
    label_id = 0

    # Precompute candidate cells: any valid cell; we will build plateaus by equality components.
    visited = np.zeros((nrows, ncols), dtype=bool)

    # These arrays will store per-label masks: outlet seeds and high-edge seeds
    # (we keep lists of coordinates to avoid huge per-label boolean stacks).
    outlet_seeds: Dict[int, list[tuple[int, int]]] = {}
    highedge_seeds: Dict[int, list[tuple[int, int]]] = {}
    flat_cells_count = 0
    drainable_flats_count = 0

    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c] or visited[r, c]:
                continue

            # Flood-fill connected component of equal elevation (within equal_tol).
            base_z = Z[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc) or (not valid[nr, nc]) or visited[nr, nc]:
                        continue
                    if is_equal(Z[nr, nc], base_z):
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))

            # Decide if this component is a "flat surface" requiring treatment:
            # - contains at least one cell with no strictly-lower neighbor
            # - contains at least one outlet cell (adjacent to strictly-lower terrain)
            any_noflow = False
            any_outlet = False

            # For Step 1, seeds are cells in the component that ARE adjacent to lower terrain.
            comp_outlets: list[tuple[int, int]] = []

            # For Step 2, initial high edges are cells adjacent to higher terrain AND not adjacent to lower terrain.
            comp_highedges: list[tuple[int, int]] = []

            for (cr, cc) in comp:
                # Outlet check: adjacent to strictly lower than base_z
                is_outlet = False
                is_adj_higher = False

                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc) or (not valid[nr, nc]):
                        if treat_oob_as_lower:
                            is_outlet = True
                        continue
                    dz = Z[nr, nc] - base_z
                    if dz < -lower_tol:
                        is_outlet = True
                    elif dz > equal_tol:
                        is_adj_higher = True

                if is_outlet:
                    any_outlet = True
                    comp_outlets.append((cr, cc))
                else:
                    # If no strictly-lower neighbor, the cell has no local downslope exit within the DEM
                    any_noflow = True

                # High-edge (Step 2) seed definition from paper:
                # adjacent to higher terrain AND no adjacent lower terrain.
                if (not is_outlet) and is_adj_higher:
                    comp_highedges.append((cr, cc))

            # If it is not "flat" (no noflow cells), skip
            # If it has no outlet, it cannot drain (G&M requires at least one lower neighbor at edge).
            if (not any_noflow) or (not any_outlet):
                continue

            # Label the component as a flat surface
            label_id += 1
            for (cr, cc) in comp:
                labels[cr, cc] = label_id
            outlet_seeds[label_id] = comp_outlets
            highedge_seeds[label_id] = comp_highedges
            flat_cells_count += len(comp)
            drainable_flats_count += 1

    # If no flats, return early
    if label_id == 0:
        out = Z.copy()
        out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        fields = {
            "labels": labels,
            "inc_towards": np.zeros_like(labels, dtype=np.int32),
            "inc_away": np.zeros_like(labels, dtype=np.int32),
            "inc_total": np.zeros_like(labels, dtype=np.int32),
            "inc_half_added": np.zeros_like(labels, dtype=np.int32),
        }
        stats = {
            "n_flats": 0.0,
            "n_flat_cells": 0.0,
            "increment_unit": 2.0 * vertical_resolution / 100000.0,
            "half_increment_unit": vertical_resolution / 100000.0,
        }
        return out, fields, stats

    # Paper's increment units
    inc_unit = 2.0 * vertical_resolution / 100000.0
    half_unit = 1.0 * vertical_resolution / 100000.0

    # --- Step 1: Gradient towards lower terrain (distance from outlets) ---
    inc_towards = np.zeros((nrows, ncols), dtype=np.int32)

    for lbl in range(1, label_id + 1):
        # BFS within this label from outlet seeds; outlets get 0 increments.
        seeds = outlet_seeds.get(lbl, [])
        if not seeds:
            # Not drainable; should not occur due to filtering.
            continue

        dist = np.full((nrows, ncols), -1, dtype=np.int32)
        q = deque()

        for (sr, sc) in seeds:
            dist[sr, sc] = 0
            q.append((sr, sc))

        while q:
            cr, cc = q.popleft()
            d0 = dist[cr, cc]
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc):
                    continue
                if labels[nr, nc] != lbl:
                    continue
                if dist[nr, nc] != -1:
                    continue
                dist[nr, nc] = d0 + 1
                q.append((nr, nc))

        # Cells at distance d>0 are incremented by d (Step 1 "passes")
        sel = (labels == lbl) & (dist > 0)
        inc_towards[sel] = dist[sel]

    # --- Step 2: Gradient away from higher terrain ---
    # This step produces a REVERSED distance ramp:
    # cells closest to higher terrain get the largest increment counts.
    inc_away = np.zeros((nrows, ncols), dtype=np.int32)

    for lbl in range(1, label_id + 1):
        # Eligible cells: flat cells NOT adjacent to lower terrain (paper excludes outlet-adjacent cells here).
        # We reconstruct "adjacent to lower" using the Step 1 seeds list.
        outlet_set = set(outlet_seeds.get(lbl, []))

        eligible = np.zeros((nrows, ncols), dtype=bool)
        region_idx = np.argwhere(labels == lbl)
        for (cr, cc) in region_idx:
            eligible[cr, cc] = (cr, cc) not in outlet_set

        # Initial high-edge seeds for Step 2 (may be empty in some geometries).
        seeds = highedge_seeds.get(lbl, [])

        if not seeds:
            # If no high edges exist, Step 2 contributes nothing for this flat surface.
            continue

        dist = np.full((nrows, ncols), -1, dtype=np.int32)
        q = deque()

        for (sr, sc) in seeds:
            if not eligible[sr, sc]:
                continue
            dist[sr, sc] = 0
            q.append((sr, sc))

        while q:
            cr, cc = q.popleft()
            d0 = dist[cr, cc]
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc):
                    continue
                if not eligible[nr, nc]:
                    continue
                if dist[nr, nc] != -1:
                    continue
                dist[nr, nc] = d0 + 1
                q.append((nr, nc))

        # Reverse the ramp to match "previously incremented cells are incremented again each pass"
        # Desired final increments:
        #   high-edge cells (dist=0) get max+1 increments,
        #   next ring gets max, ... farthest gets 1.
        max_d = dist[eligible].max() if np.any(dist[eligible] >= 0) else -1
        if max_d < 0:
            continue

        sel = eligible & (dist >= 0)
        inc_away[sel] = (max_d + 1) - dist[sel]

    # --- Step 3: Combine increments and apply to DEM ---
    inc_total = inc_towards + inc_away
    dem_out = Z.copy()
    apply = (labels > 0) & valid & (inc_total > 0)
    dem_out[apply] = dem_out[apply] + inc_unit * inc_total[apply]

    # --- Exceptional situation: cancellation creates new noflow cells ---
    # We detect remaining noflow cells inside each labeled flat, then apply Step 1 again
    # using HALF increment unit until resolved.
    inc_half_added = np.zeros((nrows, ncols), dtype=np.int32)

    def compute_noflow_mask(surface: np.ndarray, lbl: int, base_z: float) -> np.ndarray:
        """Cells in label lbl that have no strictly-lower neighbor in 'surface'."""
        m = np.zeros((nrows, ncols), dtype=bool)
        coords = np.argwhere(labels == lbl)
        for (cr, cc) in coords:
            z0 = surface[cr, cc]
            # In a perfect G&M scenario, these should be very close to base_z + increments.
            # We use z0 for correctness after increments.
            has_lower = False
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc) or (not valid[nr, nc]):
                    if treat_oob_as_lower:
                        has_lower = True
                    continue
                if surface[nr, nc] < z0 - lower_tol:
                    has_lower = True
            if not has_lower:
                m[cr, cc] = True
        return m

    exception_iters_used = 0

    for lbl in range(1, label_id + 1):
        # Safety: skip if somehow not drainable
        if not outlet_seeds.get(lbl):
            continue

        # Iterate half-increment fix until no noflow remains or cap reached.
        for _ in range(max_exception_iters):
            # Detect unresolved cells (no downslope neighbor) within this flat after Step 3.
            # If none, done.
            noflow = compute_noflow_mask(dem_out, lbl, 0.0)
            if not np.any(noflow):
                break

            # Build BFS distance from cells that DO have a downslope (i.e., NOT in noflow)
            # This matches "repeat Step 1" behavior: increment cells lacking drainage until they drain.
            region = (labels == lbl)
            seeds = np.argwhere(region & (~noflow))

            if seeds.size == 0:
                # No seed with drainage exists => cannot fix by Step 1 (should not happen for drainable flats)
                break

            dist = np.full((nrows, ncols), -1, dtype=np.int32)
            q = deque()

            for (sr, sc) in seeds:
                dist[sr, sc] = 0
                q.append((sr, sc))

            while q:
                cr, cc = q.popleft()
                d0 = dist[cr, cc]
                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc):
                        continue
                    if labels[nr, nc] != lbl:
                        continue
                    if dist[nr, nc] != -1:
                        continue
                    dist[nr, nc] = d0 + 1
                    q.append((nr, nc))

            # Apply half increments only where needed (noflow cells), proportionally to distance.
            # One "pass" of Step 1 corresponds to +1 increment, hence +half_unit here.
            sel = noflow & (dist > 0)
            if not np.any(sel):
                # If only seeds are noflow, fallback: bump all noflow by 1 half increment.
                sel = noflow

            dem_out[sel] = dem_out[sel] + half_unit * 1.0
            inc_half_added[sel] += 1

            exception_iters_used += 1

        # end per-label fix loop

    # Restore NoData
    if np.isnan(nodata):
        dem_out[~valid] = np.nan
    else:
        dem_out[~valid] = nodata

    fields = {
        "labels": labels.astype(np.int32),
        "inc_towards": inc_towards.astype(np.int32),
        "inc_away": inc_away.astype(np.int32),
        "inc_total": inc_total.astype(np.int32),
        "inc_half_added": inc_half_added.astype(np.int32),
    }

    stats = {
        "n_flats": float(label_id),
        "n_flat_cells": float(flat_cells_count),
        "n_flats_drainable": float(drainable_flats_count),
        "increment_unit": float(inc_unit),
        "half_increment_unit": float(half_unit),
        "exception_iters_used": float(exception_iters_used),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "vertical_resolution": float(vertical_resolution),
    }

    return dem_out, fields, stats
