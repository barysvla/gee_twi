from __future__ import annotations

"""
Flow-accumulation computation for DEM-based flow routing.

The function `flow_acc` computes flow accumulation from precomputed
flow-direction inputs, supporting both D8 (single-flow) and MFD
(multiple-flow) routing schemes. It is used after flow-direction
computation to derive upstream contributing area for each cell.

The accumulation is computed using a topological traversal of the flow
graph, ensuring that each cell is evaluated only after all upstream
contributions have been resolved. This requires the routing network to
be acyclic, which is ensured by prior hydrological conditioning and
flat resolution.
"""

from collections import deque
from typing import Literal

import numpy as np

# D8 neighbourhood offsets.
# The order is fixed because the index is used as the encoded
# flow-direction value:
# 0..7 = [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[tuple[int, int]] = [
    (-1, 1), (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1), (-1, -1), (-1, 0),
]


def flow_acc(
    *,
    dir_idx: np.ndarray | None = None,
    flow_weights: np.ndarray | None = None,
    nodata_mask: np.ndarray | None = None,
    pixel_area_m2: np.ndarray | None = None,
    out: Literal["cells", "m2", "km2"] = "km2",
    cycle_check: bool = True,
) -> np.ndarray:
    """
    Compute flow accumulation for D8 and FD8/MFD routing schemes.

    This function computes flow accumulation using a topological,
    dependency-based traversal of the flow graph. Cells are processed
    from locations without upstream inflow toward downstream cells, so
    that each cell is evaluated only after all contributing neighbours
    have been resolved.

    This approach is consistent with the concept of flow-network
    extraction and downstream traversal introduced by O'Callaghan and
    Mark (1984), and with subsequent implementation-oriented
    formulations used in hydrological modelling, including Barták
    (2008). In these formulations, the drainage network is treated as
    a directed acyclic graph, and accumulation is computed by resolving
    upstream dependencies in a topological order.

    In the multi-flow case, accumulated contributions are distributed
    among multiple downstream neighbours according to routing weights.

    Exactly one of `dir_idx` or `flow_weights` must be provided.

    The procedure consists of the following steps:

    Step 0
        Validate the input configuration and determine whether D8 or
        FD8/MFD routing is used.

    Step 1
        Prepare the flow-routing representation and define the valid
        computational domain.

    Step 2
        Initialize accumulation values in the requested output units.

    Step 3
        Count upstream inflow dependencies for each valid cell.

    Step 4
        Initialize the processing queue with cells that have no upstream
        inflow.

    Step 5
        Propagate accumulation downstream in topological order.

    Step 6
        Optionally verify that all valid cells were processed.

    Step 7
        Convert accumulation values to the requested output units.

    Parameters
    ----------
    dir_idx : np.ndarray of shape (H, W), optional
        D8 flow-direction indices encoded as:
            0..7 = [NE, E, SE, S, SW, W, NW, N]
            -1   = no defined outflow
        If `nodata_mask` is None, values below 0 are conservatively
        treated as NoData.
    flow_weights : np.ndarray of shape (H, W, 8), optional
        Multi-flow routing weights to D8 neighbours in the order
        [NE, E, SE, S, SW, W, NW, N]. All weights must be non-negative.
        Cells with outflow are typically expected to have weights
        summing to 1.
    nodata_mask : np.ndarray of shape (H, W), optional
        Boolean mask indicating invalid cells (True = NoData). NoData
        cells neither contribute nor receive flow.
        If None:
            - for D8, inferred as `(dir_idx < 0)`
            - for MFD, all cells are assumed valid
    pixel_area_m2 : np.ndarray of shape (H, W), optional
        Per-cell pixel area in square metres. Required when `out` is
        `"m2"` or `"km2"`.
    out : {"cells", "m2", "km2"}, default="km2"
        Output units:
            "cells" = contributing cell count
            "m2"    = contributing area in square metres
            "km2"   = contributing area in square kilometres
    cycle_check : bool, default=True
        If True, verify that all valid cells were processed. A mismatch
        usually indicates a cycle in the flow graph caused by unresolved
        flats, depressions, or incorrect hydrological conditioning.

    Returns
    -------
    acc : np.ndarray of shape (H, W), dtype float32
        Flow-accumulation raster in the requested units. NoData cells
        contain value 0.

    References
    ----------
    O'Callaghan, J. F., & Mark, D. M. (1984).
    The extraction of drainage networks from digital elevation data.
    Computer Vision, Graphics, and Image Processing, 28(3), 323–344.

    Barták, V. (2008).
    Algoritmy pro zpracování digitálních modelů terénu s aplikacemi
    v hydrologickém modelování. Diplomová práce, Česká zemědělská
    univerzita v Praze.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate the input configuration
    # ---------------------------------------------------------------------
    if (dir_idx is None) == (flow_weights is None):
        raise ValueError("Provide exactly one of dir_idx or flow_weights.")

    is_d8 = dir_idx is not None

    # ---------------------------------------------------------------------
    # Step 1: Prepare the flow-routing representation
    # ---------------------------------------------------------------------
    if is_d8:
        d = np.asarray(dir_idx)
        if d.ndim != 2:
            raise ValueError("dir_idx must have shape (H, W).")

        h, w = d.shape

        if nodata_mask is None:
            nodata = d < 0
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)
            if nodata.shape != (h, w):
                raise ValueError("nodata_mask must have shape (H, W).")

        # Retain only valid D8 direction indices.
        # All other values are treated as cells without downstream outflow.
        d_sane = np.full((h, w), -1, dtype=np.int16)
        valid_dir = (~nodata) & (d >= 0) & (d < 8)
        d_sane[valid_dir] = d[valid_dir]

    else:
        wgt = np.asarray(flow_weights, dtype=np.float32)
        if wgt.ndim != 3 or wgt.shape[2] != 8:
            raise ValueError("flow_weights must have shape (H, W, 8).")

        h, w, _ = wgt.shape

        if nodata_mask is None:
            nodata = np.zeros((h, w), dtype=bool)
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)
            if nodata.shape != (h, w):
                raise ValueError("nodata_mask must have shape (H, W).")

        # Replace NaN values with 0 and clamp negative weights to 0.
        wgt = np.nan_to_num(wgt, nan=0.0)
        np.maximum(wgt, 0.0, out=wgt)

        # NoData cells must not route flow.
        wgt[nodata, :] = 0.0

    # ---------------------------------------------------------------------
    # Step 2: Initialize accumulation values
    # ---------------------------------------------------------------------
    if out == "cells":
        acc = np.ones((h, w), dtype=np.float64)
    elif out in ("m2", "km2"):
        if pixel_area_m2 is None:
            raise ValueError('pixel_area_m2 is required when out is "m2" or "km2".')

        pa = np.asarray(pixel_area_m2, dtype=np.float64)
        if pa.shape != (h, w):
            raise ValueError("pixel_area_m2 must have shape (H, W).")

        acc = pa.copy()
    else:
        raise ValueError('out must be one of "cells", "m2", or "km2".')

    # NoData cells neither contribute nor accumulate flow.
    acc[nodata] = 0.0

    # ---------------------------------------------------------------------
    # Step 3: Count upstream inflow dependencies
    # ---------------------------------------------------------------------
    # `indeg` stores the number of upstream dependencies for each cell,
    # i.e. how many contributing neighbours must be processed before the
    # current cell can be safely evaluated in topological order.
    indeg = np.zeros((h, w), dtype=np.int32)

    if is_d8:
        for i in range(h):
            for j in range(w):
                if nodata[i, j]:
                    continue

                k = int(d_sane[i, j])
                if k < 0:
                    continue

                di, dj = D8_OFFSETS[k]
                ni, nj = i + di, j + dj

                if 0 <= ni < h and 0 <= nj < w and (not nodata[ni, nj]):
                    indeg[ni, nj] += 1

    else:
        for i in range(h):
            for j in range(w):
                if nodata[i, j]:
                    continue

                for k, (di, dj) in enumerate(D8_OFFSETS):
                    if wgt[i, j, k] <= 0.0:
                        continue

                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and (not nodata[ni, nj]):
                        indeg[ni, nj] += 1

    # ---------------------------------------------------------------------
    # Step 4: Initialize the processing queue
    # ---------------------------------------------------------------------
    # Start with cells that have no upstream inflow dependencies.
    # These correspond to source cells of the flow graph and can be
    # processed immediately because their initial accumulation value
    # is already complete.
    q: deque[tuple[int, int]] = deque()
    for i, j in np.argwhere((indeg == 0) & (~nodata)):
        q.append((int(i), int(j)))

    # ---------------------------------------------------------------------
    # Step 5: Propagate accumulation downstream
    # ---------------------------------------------------------------------
    visited = 0

    # Process cells in dependency order. A cell enters the queue only after
    # all of its upstream contributors have already been propagated to it,
    # so `acc[i, j]` is complete at the time of processing.
    while q:
        i, j = q.popleft()
        visited += 1

        if is_d8:
            # Transfer the full accumulated value to the single downstream cell.
            k = int(d_sane[i, j])
            if k < 0:
                continue

            di, dj = D8_OFFSETS[k]
            ni, nj = i + di, j + dj

            if 0 <= ni < h and 0 <= nj < w and (not nodata[ni, nj]):
                acc[ni, nj] += acc[i, j]

                # One upstream dependency of the downstream cell has now been resolved.
                indeg[ni, nj] -= 1

                # Once all upstream dependencies have been resolved, the downstream
                # cell is ready for processing.
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

        else:
            # Distribute the accumulated value among all downstream cells
            # with positive routing weights.
            a = acc[i, j]

            for k, (di, dj) in enumerate(D8_OFFSETS):
                weight = float(wgt[i, j, k])
                if weight <= 0.0:
                    continue

                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and (not nodata[ni, nj]):
                    acc[ni, nj] += a * weight

                    # One upstream dependency of the downstream cell has now been resolved.
                    indeg[ni, nj] -= 1

                    # Once all upstream dependencies have been resolved, the downstream
                    # cell is ready for processing.
                    if indeg[ni, nj] == 0:
                        q.append((ni, nj))

    # ---------------------------------------------------------------------
    # Step 6: Optionally verify complete processing
    # ---------------------------------------------------------------------
    # In a valid flow graph, all valid cells must be reachable in a
    # topological traversal. If not, the routing graph likely contains
    # a cycle, which usually indicates unresolved flats, depressions,
    # or invalid flow-direction input.
    if cycle_check:
        if visited != int((~nodata).sum()):
            raise RuntimeError(
                "Cycle detected in flow graph. "
                "Check hydrological conditioning and flat resolution."
            )

    # ---------------------------------------------------------------------
    # Step 7: Convert output units if required
    # ---------------------------------------------------------------------
    if out == "km2":
        acc *= 1e-6

    return acc.astype(np.float32)
