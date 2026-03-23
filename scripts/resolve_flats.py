from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# D8 neighbourhood offsets.
# The order is fixed because the index is used as the encoded
# flow-direction value (0..7).
NEIGHBOR_OFFSETS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Special flow-direction codes.
NODATA_DIR: int = -1
NOFLOW_DIR: int = -2

# Queue marker separating breadth-first expansion levels.
QUEUE_MARKER: Tuple[int, int] = (-1, -1)


def _deduplicate_queue(q: Deque[Tuple[int, int]]) -> Deque[Tuple[int, int]]:
    """Return a queue with duplicate coordinates removed while preserving order."""
    seen = set()
    out: Deque[Tuple[int, int]] = deque()

    for item in q:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def resolve_flats_barnes_2014(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    equal_tol: float = 0.0,
    lower_tol: float = 0.0,
    treat_oob_as_lower: bool = True,
    apply_to_dem: str = "none",
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Resolve drainable flat areas in a raster DEM following Barnes et al. (2014).

    This implementation follows the main logic of the flat-resolution
    procedure described by Barnes, Lehman, and Mulla (2014). It constructs
    an auxiliary integer mask (`flat_mask`) that imposes a unique drainage
    pattern over drainable flats without iterative DEM correction.

    The procedure consists of the following steps:

    Step 0
        Compute initial D8 flow directions. Cells without a strictly lower
        neighbour are marked as `NOFLOW_DIR`.

    Step 1
        Identify flat boundary cells adjacent to higher terrain
        (`high_edges`) and to lower terrain (`low_edges`).

    Step 1b
        Label each drainable flat by flood-filling connected cells of equal
        elevation.

    Step 2
        Build a gradient away from higher terrain by breadth-first expansion
        from `high_edges`.

    Step 3
        Build a gradient towards lower terrain by breadth-first expansion
        from `low_edges` and combine both gradients, assigning double weight
        to the towards-lower component.

    Step 4
        Reassign flow directions for labeled flat cells previously marked as
        `NOFLOW_DIR`.

    Relative to the reference C++/RichDEM implementation, this version adds
    support for configurable elevation tolerances, explicit handling of
    raster-edge drainage, and optional epsilon-based DEM modification.

    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    nodata : float, default=np.nan
        NoData marker. If set to NaN, all non-finite values are treated as
        invalid.
    equal_tol : float, default=0.0
        Absolute tolerance used when comparing elevations for flat membership
        and equal-elevation edge detection.
    lower_tol : float, default=0.0
        Minimum required elevation drop for a neighbour to be considered
        strictly lower.
    treat_oob_as_lower : bool, default=True
        If True, raster-edge cells are treated as draining outward when no
        strictly lower in-bounds neighbour exists.
    apply_to_dem : {"none", "epsilon"}, default="none"
        Optional DEM modification mode. The standard workflow keeps the DEM
        unchanged and uses only `flat_mask` and `flowdirs`. If set to
        `"epsilon"`, drainable flat cells are incremented by
        `epsilon * flat_mask`.
    epsilon : float, default=1e-5
        Increment used when `apply_to_dem="epsilon"`.

    Returns
    -------
    dem_out : np.ndarray
        Output DEM, optionally modified on drainable flats.
    flat_mask : np.ndarray
        Integer drainage mask. Cells outside labeled flats are zero.
    labels : np.ndarray
        Flat labels. Cells outside flats are zero.
    flowdirs : np.ndarray
        D8 flow directions encoded as 0..7, or special values `NODATA_DIR`
        and `NOFLOW_DIR`.
    stats : dict
        Diagnostic counters describing identified and resolved flats.

    Reference
    ---------
    Barnes, R., Lehman, C., Mulla, D. (2014).
    An efficient assignment of drainage direction over flat surfaces in raster
    digital elevation models.
    Computers & Geosciences, 62, 128–135.
    """
    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    n_rows, n_cols = dem_values.shape

    if np.isnan(nodata):
        valid = np.isfinite(dem_values)
    else:
        valid = np.isfinite(dem_values) & (dem_values != nodata)

    def in_bounds(r: int, c: int) -> bool:
        """Return True if the cell lies inside the raster extent."""
        return 0 <= r < n_rows and 0 <= c < n_cols

    def is_equal(a: float, b: float) -> bool:
        """Return True if two elevations are equal within `equal_tol`."""
        return abs(a - b) <= equal_tol

    def is_strictly_lower(z_here: float, z_nbr: float) -> bool:
        """Return True if the neighbour is lower by more than `lower_tol`."""
        return (z_nbr - z_here) < -lower_tol

    def is_edge_cell(r: int, c: int) -> bool:
        """Return True if the cell lies on the raster boundary."""
        return r == 0 or c == 0 or r == n_rows - 1 or c == n_cols - 1

    # ---------------------------------------------------------------------
    # Step 0: Compute initial D8 flow directions
    # ---------------------------------------------------------------------
    flowdirs = np.full((n_rows, n_cols), NODATA_DIR, dtype=np.int16)

    for r in range(n_rows):
        for c in range(n_cols):
            if not valid[r, c]:
                continue

            z0 = dem_values[r, c]

            if treat_oob_as_lower and is_edge_cell(r, c):
                # Assign a real downslope neighbour when present.
                # Otherwise represent outward drainage from the raster edge.
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

    # ---------------------------------------------------------------------
    # Step 1: Identify flat boundary cells
    # ---------------------------------------------------------------------
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

                # Low edge: a flow-defined cell adjacent to an equal-elevation
                # cell with no local downslope.
                if (
                    flowdirs[r, c] != NOFLOW_DIR
                    and flowdirs[nr, nc] == NOFLOW_DIR
                    and is_equal(z0, dem_values[nr, nc])
                ):
                    low_edges.append((r, c))
                    break

                # High edge: a NOFLOW cell adjacent to higher terrain.
                if flowdirs[r, c] == NOFLOW_DIR and (dem_values[nr, nc] - z0) > equal_tol:
                    high_edges.append((r, c))
                    break

    if len(low_edges) == 0:
        dem_out = dem_values.copy()
        dem_out[~valid] = np.nan if np.isnan(nodata) else nodata

        flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
        labels = np.zeros((n_rows, n_cols), dtype=np.int32)

        stats: Dict[str, int] = {
            "n_flats": 0,
            "n_flat_cells": 0,
            "n_low_edges": 0,
            "n_high_edges": int(len(high_edges)),
            "note_undrainable_flats": int(len(high_edges) > 0),
            "apply_to_dem_mode": 0,
        }
        return dem_out, flat_mask, labels, flowdirs, stats

    low_edges = _deduplicate_queue(low_edges)
    high_edges = _deduplicate_queue(high_edges)

    # ---------------------------------------------------------------------
    # Step 1b: Label each drainable flat
    # ---------------------------------------------------------------------
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    current_label = 0

    for sr, sc in list(low_edges):
        if labels[sr, sc] != 0:
            continue

        flat_elevation = dem_values[sr, sc]
        current_label += 1

        to_fill: Deque[Tuple[int, int]] = deque()
        to_fill.append((sr, sc))

        while to_fill:
            r, c = to_fill.popleft()

            if not in_bounds(r, c) or not valid[r, c]:
                continue
            if not is_equal(dem_values[r, c], flat_elevation):
                continue
            if labels[r, c] != 0:
                continue

            labels[r, c] = current_label

            for dr, dc in NEIGHBOR_OFFSETS_8:
                to_fill.append((r + dr, c + dc))

    # Retain only high-edge cells belonging to labeled drainable flats.
    # If a cell appears in both edge sets, low-edge status takes precedence.
    low_edge_set = set(low_edges)

    high_edges_filtered: Deque[Tuple[int, int]] = deque()
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

    # ---------------------------------------------------------------------
    # Step 2: Build gradient away from higher terrain
    # ---------------------------------------------------------------------
    flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
    flat_height = np.zeros(current_label + 1, dtype=np.int32)

    loops = 1
    q: Deque[Tuple[int, int]] = deque(high_edges)
    q.append(QUEUE_MARKER)

    while len(q) > 1:
        r, c = q.popleft()

        if (r, c) == QUEUE_MARKER:
            loops += 1
            q.append(QUEUE_MARKER)
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

    # ---------------------------------------------------------------------
    # Step 3: Build gradient towards lower terrain and combine gradients
    # ---------------------------------------------------------------------
    flat_mask = -flat_mask

    loops = 1
    q = deque(low_edges)
    q.append(QUEUE_MARKER)

    while len(q) > 1:
        r, c = q.popleft()

        if (r, c) == QUEUE_MARKER:
            loops += 1
            q.append(QUEUE_MARKER)
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

    # ---------------------------------------------------------------------
    # Step 4: Reassign flow directions within labeled flats
    # ---------------------------------------------------------------------
    for r in range(n_rows):
        for c in range(n_cols):
            if flowdirs[r, c] == NODATA_DIR:
                continue
            if flowdirs[r, c] != NOFLOW_DIR:
                continue

            lbl = labels[r, c]
            if lbl == 0:
                continue

            min_mask = flat_mask[r, c]
            best_dir: Optional[int] = None

            for k, (dr, dc) in enumerate(NEIGHBOR_OFFSETS_8):
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                if flowdirs[nr, nc] == NODATA_DIR:
                    continue
                if labels[nr, nc] != lbl:
                    continue

                nbr_mask = flat_mask[nr, nc]

                if nbr_mask < min_mask:
                    min_mask = nbr_mask
                    best_dir = k
                elif (
                    nbr_mask == min_mask
                    and best_dir is not None
                    and (best_dir % 2 == 1)
                    and (k % 2 == 0)
                ):
                    # In tie cases, prefer a diagonal direction to maintain
                    # behaviour consistent with the reference implementation.
                    best_dir = k

            if best_dir is not None:
                flowdirs[r, c] = best_dir

    # ---------------------------------------------------------------------
    # Optional DEM modification
    # ---------------------------------------------------------------------
    dem_out = dem_values.copy()

    if apply_to_dem not in ("none", "epsilon"):
        raise ValueError('apply_to_dem must be "none" or "epsilon".')

    if apply_to_dem == "epsilon":
        mask = (labels > 0) & valid & (flat_mask > 0)
        dem_out[mask] = dem_out[mask] + epsilon * flat_mask[mask]

    dem_out[~valid] = np.nan if np.isnan(nodata) else nodata

    stats = {
        "n_flats": int(current_label),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_low_edges": int(len(low_edges)),
        "n_high_edges": int(len(high_edges)),
        "removed_highedges_unlabeled": int(removed_unlabeled),
        "removed_highedges_low_dominates": int(removed_low_dominates),
        "apply_to_dem_mode": {"none": 0, "epsilon": 1}[apply_to_dem],
    }

    return (
        dem_out,
        flat_mask.astype(np.int32),
        labels.astype(np.int32),
        flowdirs,
        stats,
    )
