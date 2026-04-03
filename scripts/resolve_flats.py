from __future__ import annotations

"""
Flat resolution as the second step of DEM hydrological conditioning.

The function `resolve_flats_barnes_2014` resolves drainable flats in a
DEM after depression filling using the flat-resolution procedure of
Barnes et al. (2014). The output is a DEM in which the resolved flat
ordering is encoded directly into the surface using epsilon-scaled
increments for subsequent flow-direction computation.
"""

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

# D8 neighbourhood offsets.
# The offsets define the 8-cell neighbourhood used for flat detection
# and breadth-first propagation:
# [NE, E, SE, S, SW, W, NW, N]
NEIGHBOR_OFFSETS_8: List[Tuple[int, int]] = [
    (-1, 1), (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1), (-1, -1), (-1, 0),
]


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
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Resolve drainable flat areas in a raster DEM following Barnes et al. (2014).
    
    This function implements the flat-resolution procedure of Barnes,
    Lehman, and Mulla (2014) as the second step of DEM hydrological
    conditioning. It constructs an auxiliary integer mask (`flat_mask`)
    based on a superposition of two breadth-first gradients, which imposes
    a consistent and convergent drainage pattern over drainable flats.
    
    Relative to the reference C++/RichDEM implementation, this version encodes the
    resolved flat ordering directly into the DEM surface using epsilon-scaled increments
    instead of assigning flow directions within flats. The returned DEM therefore
    contains a small artificial gradient over formerly flat areas and can be used
    directly for subsequent flow-direction computation (e.g., D8 or MFD). It also
    introduces configurable elevation tolerances and explicit handling of
    raster-edge drainage.
    
    The procedure consists of the following steps:
    
    Step 0
        Detect cells without a strictly lower neighbour. These cells are treated
        as candidates for flat areas.
    
    Step 1
        Identify flat-edge cells adjacent to higher terrain (`high_edges`)
        and cells adjacent to lower terrain (`low_edges`) that define
        potential outlets.
    
    Step 1b
        Label each drainable flat by flood-filling connected cells of equal
        elevation starting from `low_edges`.
    
    Step 2
        Build a gradient away from higher terrain by breadth-first
        expansion from `high_edges`.
    
    Step 3
        Build a gradient towards lower terrain by breadth-first expansion
        from `low_edges` and combine both gradients, assigning double
        weight to the towards-lower component.
    
    Step 4
        Apply the combined flat ordering to the DEM using epsilon-scaled
        increments.

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
    epsilon : float, default=1e-5
        Increment used to encode flat ordering into the DEM surface.
    
    Returns
    -------
    dem_out : np.ndarray
        Output DEM with NoData preserved and flat areas modified to include
        a small artificial gradient.
    flat_mask : np.ndarray
        Integer flat-resolution mask. Cells outside labeled flats are zero.
    labels : np.ndarray
        Flat labels. Cells outside flats are zero.
    stats : dict
        Diagnostic counters describing identified and resolved flats.
    
    References
    ----------
    Barnes, R., Lehman, C., Mulla, D. (2014).
    An efficient assignment of drainage direction over flat surfaces in raster
    digital elevation models.
    Computers & Geosciences, 62, 128–135.
    """
    # Convert input DEM to a floating-point array and validate shape.
    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")
    
    n_rows, n_cols = dem_values.shape
    
    # Determine the valid computational domain of the DEM.
    # Cells marked as NoData (or non-finite) are excluded from all
    # subsequent flat detection and propagation steps.
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
    # Step 0: Detect cells without local downslope
    # ---------------------------------------------------------------------
    # Identify valid cells that do not have any strictly lower neighbour.
    # These cells are treated as flat candidates in subsequent steps,
    # whereas cells with at least one lower neighbour are not part of the
    # flat-resolution domain.
    flat_candidates = np.zeros((n_rows, n_cols), dtype=bool)
    
    for r in range(n_rows):
        for c in range(n_cols):
            if not valid[r, c]:
                continue
    
            z0 = dem_values[r, c]
            has_downslope = False
    
            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid[nr, nc]:
                    continue
    
                if is_strictly_lower(z0, dem_values[nr, nc]):
                    has_downslope = True
                    break
    
            flat_candidates[r, c] = not has_downslope
    
    
    # ---------------------------------------------------------------------
    # Step 1: Identify flat boundary cells
    # ---------------------------------------------------------------------
    # Identify the boundary cells needed to resolve drainable flats.
    # `high_edges` mark flat cells adjacent to higher terrain, whereas
    # `low_edges` mark cells adjacent to drainable flats on their outlet
    # side and therefore define the seeds used to label such flats.
    high_edges: Deque[Tuple[int, int]] = deque()
    low_edges: Deque[Tuple[int, int]] = deque()
    
    for r in range(n_rows):
        for c in range(n_cols):
            if not valid[r, c]:
                continue
    
            z0 = dem_values[r, c]
    
            # Treat raster-edge cells without a downslope neighbour as drainable
            # flat outlets when outward drainage across the DEM boundary is allowed.
            if (
                treat_oob_as_lower
                and is_edge_cell(r, c)
                and flat_candidates[r, c]
            ):
                low_edges.append((r, c))
    
            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or not valid[nr, nc]:
                    continue
    
                # Low edge: a non-flat cell adjacent to an equal-elevation
                # flat candidate, indicating the outlet side of a drainable flat.
                if (
                    (not flat_candidates[r, c])
                    and flat_candidates[nr, nc]
                    and is_equal(z0, dem_values[nr, nc])
                ):
                    low_edges.append((r, c))
                    break
    
                # High edge: a flat candidate adjacent to higher terrain.
                # These cells seed the gradient directed away from higher ground.
                if flat_candidates[r, c] and (dem_values[nr, nc] - z0) > equal_tol:
                    high_edges.append((r, c))
                    break
    
    # If no low-edge cells exist, no drainable flats were identified.
    # The DEM may contain only undrainable flats or no flats at all.
    # Return the original DEM together with empty flat-resolution outputs.
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
        }
        return dem_out, flat_mask, labels, stats
    
    # Remove duplicate edge coordinates while preserving discovery order.
    low_edges = _deduplicate_queue(low_edges)
    high_edges = _deduplicate_queue(high_edges)

    # ---------------------------------------------------------------------
    # Step 1b: Label each drainable flat
    # ---------------------------------------------------------------------
    # Starting from low-edge locations, flood-fill connected equal-elevation
    # cells so that each drainable flat receives a unique label.
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
    # Propagate breadth-first from high-edge cells to assign increasing
    # mask values with distance from higher terrain. At the same time,
    # record the maximum away-from-higher level reached within each labeled
    # flat for use in the subsequent combination step.
    flat_mask = np.zeros((n_rows, n_cols), dtype=np.int32)
    flat_height = np.zeros(current_label + 1, dtype=np.int32)
    
    # Use a queue marker to separate BFS levels so that `loops` corresponds
    # to the current breadth-first expansion distance from high edges.
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
    
        # Store the maximum away-from-higher level reached in this flat.
        if loops > flat_height[lbl]:
            flat_height[lbl] = loops
    
        for dr, dc in NEIGHBOR_OFFSETS_8:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not valid[nr, nc]:
                continue
            if labels[nr, nc] != lbl:
                continue
            if not flat_candidates[nr, nc]:
                continue
    
            q.append((nr, nc))

    # ---------------------------------------------------------------------
    # Step 3: Build gradient towards lower terrain and combine gradients
    # ---------------------------------------------------------------------
    # Negate the away-from-higher component so it can be distinguished from
    # cells not yet processed in the second pass. Then propagate breadth-
    # first from low-edge cells to build the towards-lower component and
    # combine both gradients into the final flat mask.
    flat_mask = -flat_mask
    
    # As in Step 2, use a queue marker to separate BFS levels so that
    # `loops` represents the current expansion distance from low edges.
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
    
        # If the cell already has an away-from-higher value, combine it with
        # the towards-lower component. Following Barnes et al. (2014), the
        # towards-lower gradient is weighted twice to ensure drainage toward
        # the outlet dominates the final ordering.
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
            if not flat_candidates[nr, nc]:
                continue
    
            q.append((nr, nc))

    # ---------------------------------------------------------------------
    # Step 4: Apply flat ordering to the DEM
    # ---------------------------------------------------------------------
    # Encode the resolved flat ordering directly into the DEM surface by
    # adding epsilon-scaled increments to labeled drainable flat cells.
    # This produces a DEM with a small artificial gradient that can be used
    # for subsequent flow-direction computation.
    dem_out = dem_values.copy()
    
    # Modify only valid cells belonging to labeled drainable flats.
    mask = (labels > 0) & valid & (flat_mask > 0)
    dem_out[mask] = dem_out[mask] + epsilon * flat_mask[mask]
    
    # Restore the original NoData mask in the output DEM.
    dem_out[~valid] = np.nan if np.isnan(nodata) else nodata
    
    # Collect diagnostic counters describing the identified and resolved flats.
    stats = {
        "n_flats": int(current_label),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_low_edges": int(len(low_edges)),
        "n_high_edges": int(len(high_edges)),
        "removed_highedges_unlabeled": int(removed_unlabeled),
        "removed_highedges_low_dominates": int(removed_low_dominates),
    }
    
    return (
        dem_out,
        flat_mask.astype(np.int32),
        labels.astype(np.int32),
        stats,
    )
