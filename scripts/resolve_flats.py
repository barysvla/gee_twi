from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional

# 8-neighborhood offsets (D8): order is arbitrary but must be consistent everywhere
# Index k in this list is used as the encoded flow direction (0..7).
NEIGHBOR_OFFSETS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# FlowDirs special values (int16):
# - NODATA_DIR: cell is outside DEM / invalid
# - NOFLOW_DIR: cell has no strictly lower neighbor (flat/pit candidate in the initial flowdir pass)
NODATA_DIR = -1
NOFLOW_DIR = -2


def resolve_flats_barnes_2014(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    equal_tol: float = 0.0,
    lower_tol: float = 0.0,
    treat_oob_as_lower: bool = True,
    apply_to_dem: str = "none",  # "none" | "epsilon"
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Flat resolution following Barnes, Lehman & Mulla (2014), implemented as a direct
    translation of Algorithms 1–7.

    The algorithm resolves drainable flat areas (cells without local downslope gradient)
    by constructing an auxiliary integer mask (`FlatMask`) that enforces a unique drainage
    pattern inside each flat. The mask is obtained as a superposition of two within-flat
    gradients: (1) away from higher terrain and (2) towards lower terrain, with the latter
    given double weight to guarantee drainage without iterative correction.

    In brief:
    - initial D8 flow directions are computed and cells without a strictly lower neighbor
      are marked as NOFLOW,
    - flat boundary cells adjacent to higher terrain (HighEdges) and to outlets (LowEdges)
      are identified,
    - each drainable flat is labeled by flood-filling equal-elevation cells,
    - a breadth-first expansion from HighEdges builds the “away-from-higher” gradient,
    - a breadth-first expansion from LowEdges builds the “towards-lower” gradient and both
      components are combined into `FlatMask`,
    - flow directions for previously NOFLOW cells are reassigned using masked D8 within
      each labeled flat.

    Optionally, the DEM itself can be modified by adding epsilon * FlatMask, but the standard
    workflow uses only `FlatMask` and `Labels`.

    Returns:
      dem_out   : float64 DEM, optionally modified on drainable flats
      flat_mask : integer drainage mask (0 outside flats)
      labels    : flat labels (0 outside flats)
      flowdirs  : D8 flow directions (0..7), or NOFLOW/NODATA
      stats     : basic diagnostic counters
    """
    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    n_rows, n_cols = dem_values.shape

    # Build a boolean validity mask once: True for cells participating in DEM computations.
    # - If nodata is NaN: only finite values are valid.
    # - If nodata is numeric: valid means finite and not equal to nodata.
    if np.isnan(nodata):
        valid = np.isfinite(dem_values)
    else:
        valid = np.isfinite(dem_values) & (dem_values != nodata)

    def in_bounds(r: int, c: int) -> bool:
        # True if (r,c) lies inside the raster bounds
        return 0 <= r < n_rows and 0 <= c < n_cols

    def is_equal(a: float, b: float) -> bool:
        # Equality predicate used to define flat membership and low-edge adjacency
        return abs(a - b) <= equal_tol

    def is_strictly_lower(z_here: float, z_nbr: float) -> bool:
        # True if neighbor is lower than current by more than lower_tol
        return (z_nbr - z_here) < -lower_tol

    def is_edge_cell(r: int, c: int) -> bool:
        # Edge cells are treated specially when treat_oob_as_lower=True
        return r == 0 or c == 0 or r == n_rows - 1 or c == n_cols - 1

    # -------------------------------------------------------------------------
    # STEP 0: Initial flow directions (Algorithm 2 example: D8FLOWDIRS)
    # -------------------------------------------------------------------------
    flowdirs = np.full((n_rows, n_cols), NODATA_DIR, dtype=np.int16)

    for r in range(n_rows):
        for c in range(n_cols):
            if not valid[r, c]:
                continue

            z0 = dem_values[r, c]

            if treat_oob_as_lower and is_edge_cell(r, c):
                # Prefer an actual in-bounds downslope neighbor if one exists.
                # Otherwise, assign an arbitrary defined direction (the cell is considered draining outward).
                best_dir: Optional[int] = None
                best_z = z0
                for k, (dr, dc) in enumerate(NEIGHBOR_OFFSETS_8):
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc) or not valid[nr, nc]:
                        continue
                    zn = dem_values[nr, nc]
                    if is_strictly_lower(z0, zn) and (best_dir is None or zn < best_z):
                        best_dir = k
                        best_z = zn
                flowdirs[r, c] = 0 if best_dir is None else best_dir
                continue

            # Interior cells: standard D8 selection of a strictly lower neighbor with minimum elevation.
            best_dir = None
            best_z = z0
            for k, (dr, dc) in enumerate(NEIGHBOR_OFFSETS_8):
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid[nr, nc]:
                    continue
                zn = dem_values[nr, nc]
                if is_strictly_lower(z0, zn) and (best_dir is None or zn < best_z):
                    best_dir = k
                    best_z = zn

            flowdirs[r, c] = NOFLOW_DIR if best_dir is None else best_dir

    # -------------------------------------------------------------------------
    # STEP 1: Locate flat boundary cells (Algorithm 3: FLATEDGES)
    # -------------------------------------------------------------------------
    high_edges: Deque[Tuple[int, int]] = deque()
    low_edges: Deque[Tuple[int, int]] = deque()

    for r in range(n_rows):
        for c in range(n_cols):
            if not valid[r, c]:
                continue

            z0 = dem_values[r, c]

            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid[nr, nc]:
                    continue

                # Low edge criterion (Algorithm 3, lines 7-9)
                if (
                    flowdirs[r, c] != NOFLOW_DIR
                    and flowdirs[nr, nc] == NOFLOW_DIR
                    and is_equal(z0, dem_values[nr, nc])
                ):
                    low_edges.append((r, c))
                    break

                # High edge criterion (Algorithm 3, lines 10-12)
                if flowdirs[r, c] == NOFLOW_DIR and (dem_values[nr, nc] - z0) > equal_tol:
                    high_edges.append((r, c))
                    break

    if len(low_edges) == 0:
        dem_out = dem_values.copy()
        dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)

        flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
        labels = np.zeros((n_rows, n_cols), dtype=np.int32)

        stats = {
            "n_flats": 0,
            "n_flat_cells": 0,
            "n_low_edges": 0,
            "n_high_edges": int(len(high_edges)),
            "note_undrainable_flats": int(len(high_edges) > 0),
            "apply_to_dem_mode": 0,
        }
        return dem_out, flat_mask, labels, flowdirs, stats

    def dedup_queue(q: Deque[Tuple[int, int]]) -> Deque[Tuple[int, int]]:
        seen = set()
        out = deque()
        for item in q:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    low_edges = dedup_queue(low_edges)
    high_edges = dedup_queue(high_edges)

    # -------------------------------------------------------------------------
    # STEP 1b: Label each drainable flat (Algorithm 4: LABELFLATS)
    # -------------------------------------------------------------------------
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    current_label = 0

    for (sr, sc) in list(low_edges):
        if labels[sr, sc] != 0:
            continue

        E = dem_values[sr, sc]
        current_label += 1

        to_fill: Deque[Tuple[int, int]] = deque()
        to_fill.append((sr, sc))

        while to_fill:
            r, c = to_fill.popleft()

            if not in_bounds(r, c) or not valid[r, c]:
                continue
            if not is_equal(dem_values[r, c], E):
                continue
            if labels[r, c] != 0:
                continue

            labels[r, c] = current_label

            for dr, dc in NEIGHBOR_OFFSETS_8:
                to_fill.append((r + dr, c + dc))

    # Remove HighEdges that are not part of any labeled flat.
    # Enforce: if a cell is also in LowEdges, it must not be used as a HighEdge.
    low_edge_set = set(low_edges)

    high_edges_filtered = deque()
    removed_unlabeled = 0
    removed_low_dominates = 0

    for rc in high_edges:
        r, c = rc
        if labels[r, c] == 0:
            removed_unlabeled += 1
            continue
        if rc in low_edge_set:
            removed_low_dominates += 1
            continue
        high_edges_filtered.append(rc)

    high_edges = high_edges_filtered

    # -------------------------------------------------------------------------
    # STEP 2: Gradient away from higher terrain (Algorithm 5: AWAYFROMHIGHER)
    # -------------------------------------------------------------------------
    flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
    flat_height = np.zeros(current_label + 1, dtype=np.int32)

    MARKER = (-999999, -999999)
    loops = 1

    q: Deque[Tuple[int, int]] = deque(high_edges)
    q.append(MARKER)

    while len(q) > 1:
        r, c = q.popleft()

        if (r, c) == MARKER:
            loops += 1
            q.append(MARKER)
            continue

        if flat_mask[r, c] > 0:
            continue

        flat_mask[r, c] = loops
        lbl = labels[r, c]

        if loops > flat_height[lbl]:
            flat_height[lbl] = loops

        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not valid[nr, nc]:
                continue
            if labels[nr, nc] != lbl:
                continue
            if flowdirs[nr, nc] != NOFLOW_DIR:
                continue
            q.append((nr, nc))

    # -------------------------------------------------------------------------
    # STEP 3: Gradient towards lower terrain + superposition (Algorithm 6: TOWARDSLOWER)
    # -------------------------------------------------------------------------
    flat_mask = -flat_mask

    loops = 1
    q = deque(low_edges)
    q.append(MARKER)

    while len(q) > 1:
        r, c = q.popleft()

        if (r, c) == MARKER:
            loops += 1
            q.append(MARKER)
            continue

        if flat_mask[r, c] > 0:
            continue

        lbl = labels[r, c]

        if flat_mask[r, c] < 0:
            flat_mask[r, c] = flat_height[lbl] + flat_mask[r, c] + 2 * loops
        else:
            flat_mask[r, c] = 2 * loops

        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not valid[nr, nc]:
                continue
            if labels[nr, nc] != lbl:
                continue
            if flowdirs[nr, nc] != NOFLOW_DIR:
                continue
            q.append((nr, nc))

    # -------------------------------------------------------------------------
    # STEP 4: Determine flow directions over flats using FlatMask (Algorithm 7: D8MASKEDFLOWDIRS)
    # -------------------------------------------------------------------------
    for r in range(n_rows):
        for c in range(n_cols):
            if flowdirs[r, c] == NODATA_DIR:
                continue
            if flowdirs[r, c] != NOFLOW_DIR:
                continue

            lbl = labels[r, c]
            if lbl == 0:
                continue

            emin = flat_mask[r, c]
            best_dir = None

            for k, (dr, dc) in enumerate(NEIGHBOR_OFFSETS_8):
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                if flowdirs[nr, nc] == NODATA_DIR:
                    continue
                if labels[nr, nc] != lbl:
                    continue
                if flat_mask[nr, nc] < emin:
                    emin = flat_mask[nr, nc]
                    best_dir = k

            if best_dir is not None:
                flowdirs[r, c] = best_dir

    # -------------------------------------------------------------------------
    # Optional: alter DEM elevations instead of using the mask (epsilon only)
    # -------------------------------------------------------------------------
    dem_out = dem_values.copy()

    if apply_to_dem not in ("none", "epsilon"):
        raise ValueError('apply_to_dem must be "none" or "epsilon".')

    if apply_to_dem == "epsilon":
        m = (labels > 0) & valid & (flat_mask > 0)
        dem_out[m] = dem_out[m] + epsilon * flat_mask[m]

    dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)

    stats: Dict[str, int] = {
        "n_flats": int(current_label),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_low_edges": int(len(low_edges)),
        "n_high_edges": int(len(high_edges)),
        "removed_highedges_unlabeled": int(removed_unlabeled),
        "removed_highedges_low_dominates": int(removed_low_dominates),
        "apply_to_dem_mode": {"none": 0, "epsilon": 1}[apply_to_dem],
    }

    return dem_out, flat_mask.astype(np.int32), labels.astype(np.int32), flowdirs, stats
    
