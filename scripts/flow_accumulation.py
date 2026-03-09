from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np


# D8 neighborhood offsets in the following order:
# [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[tuple[int, int]] = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def compute_flow_accumulation(
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

    The function propagates per-cell contributions through a directed,
    acyclic flow network derived either from a single-flow (D8) direction
    raster or from a multi-flow (FD8/MFD) weight tensor. Cells are processed
    progressively from locations without upstream inflow toward downstream
    cells, ensuring that each cell is evaluated only after all of its
    contributing neighbors have been resolved.

    Exactly one of `dir_idx` or `flow_weights` must be provided.

    Parameters
    ----------
    dir_idx : ndarray of shape (H, W), optional
        D8 flow direction indices. Encoding:
            0..7  = [NE, E, SE, S, SW, W, NW, N]
            -1    = no outflow (sink/outlet) or NoData (depending on nodata_mask).
        If `nodata_mask` is None, values < 0 are conservatively treated as NoData.

    flow_weights : ndarray of shape (H, W, 8), optional
        Multi-flow (FD8/MFD) routing weights to D8 neighbors in the order
        [NE, E, SE, S, SW, W, NW, N].
        All weights must be non-negative. For cells with outflow,
        weights are expected to sum to 1.

    nodata_mask : ndarray of shape (H, W), optional
        Boolean mask indicating invalid cells (True = NoData).
        NoData cells neither contribute nor receive flow.
        If None:
            - For D8, inferred as (dir_idx < 0).
            - For MFD, all cells are assumed valid.

    pixel_area_m2 : ndarray of shape (H, W), optional
        Per-cell pixel area in square meters.
        Required when `out` is "m2" or "km2".
        In WGS84 workflows, this must account for latitude-dependent
        variation in pixel size.

    out : {"cells", "m2", "km2"}, default="km2"
        Output units:
            "cells"  – contributing cell count
            "m2"     – contributing area in square meters
            "km2"    – contributing area in square kilometers

    cycle_check : bool, default=True
        If True, verifies that the flow network is acyclic by comparing
        the number of processed cells with the number of valid cells.
        Raises RuntimeError if a cycle is detected, which usually indicates
        unresolved flats, depressions, or incorrect hydrological conditioning.

    Returns
    -------
    acc : ndarray of shape (H, W), dtype float32
        Flow accumulation raster in the requested units.
    """

    # -------------------------------------------------------------------------
    # 1. Validate input configuration
    # -------------------------------------------------------------------------

    if (dir_idx is None) == (flow_weights is None):
        raise ValueError("Provide exactly one of dir_idx or flow_weights.")

    is_d8 = dir_idx is not None

    # -------------------------------------------------------------------------
    # 2. Prepare the flow-routing representation
    # -------------------------------------------------------------------------

    if is_d8:
        # D8 mode: each cell has at most one downstream receiver.
        d = np.asarray(dir_idx)
        if d.ndim != 2:
            raise ValueError("dir_idx must have shape (H, W).")
        H, W = d.shape

        # Define the valid computational domain.
        if nodata_mask is None:
            # Conservative fallback: negative direction indices are treated as NoData.
            nodata = (d < 0)
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)

        # Keep only valid D8 indices (0..7); all other values are treated
        # as cells without downstream outflow.
        d_sane = np.full((H, W), -1, dtype=np.int16)
        valid_dir = (~nodata) & (d >= 0) & (d < 8)
        d_sane[valid_dir] = d[valid_dir]

    else:
        # MFD/FD8 mode: each cell may distribute flow to multiple neighbors.
        Wgt = np.asarray(flow_weights, dtype=np.float32)
        if Wgt.ndim != 3 or Wgt.shape[2] != 8:
            raise ValueError("flow_weights must have shape (H, W, 8).")
        H, W, _ = Wgt.shape

        if nodata_mask is None:
            nodata = np.zeros((H, W), dtype=bool)
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)

        # Improve numerical robustness:
        # - replace NaN values with 0
        # - clamp negative weights to 0
        Wgt = np.nan_to_num(Wgt, nan=0.0)
        np.maximum(Wgt, 0.0, out=Wgt)

        # NoData cells must not send any flow.
        Wgt[nodata, :] = 0.0

    # -------------------------------------------------------------------------
    # 3. Initialize accumulation values
    # -------------------------------------------------------------------------

    # Each valid cell contributes either:
    # - 1, when accumulation is expressed as contributing cell count
    # - its own pixel area, when accumulation is expressed in area units
    if out == "cells":
        acc = np.ones((H, W), dtype=np.float64)
    else:
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 array is required.")
        pa = np.asarray(pixel_area_m2, dtype=np.float64)
        if pa.shape != (H, W):
            raise ValueError("pixel_area_m2 must have shape (H, W).")

        # Each cell starts with its own contributing area.
        acc = pa.copy()

    # NoData cells neither contribute nor accumulate flow.
    acc[nodata] = 0.0

    # -------------------------------------------------------------------------
    # 4. Count upstream inflows for each cell
    # -------------------------------------------------------------------------

    # indeg[i, j] stores the number of neighboring cells routing flow into (i, j).
    indeg = np.zeros((H, W), dtype=np.int32)

    if is_d8:
        for i in range(H):
            for j in range(W):
                if nodata[i, j]:
                    continue
                k = int(d_sane[i, j])
                if k < 0:
                    continue

                di, dj = D8_OFFSETS[k]
                ni, nj = i + di, j + dj

                if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                    indeg[ni, nj] += 1
    else:
        for i in range(H):
            for j in range(W):
                if nodata[i, j]:
                    continue
                for k, (di, dj) in enumerate(D8_OFFSETS):
                    if Wgt[i, j, k] <= 0.0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                        indeg[ni, nj] += 1

    # -------------------------------------------------------------------------
    # 5. Initialize the processing queue
    # -------------------------------------------------------------------------

    # Start from cells without upstream inflow. These usually correspond
    # to source areas, ridges, or local topographic highs.
    q: deque[tuple[int, int]] = deque()
    for i, j in np.argwhere((indeg == 0) & (~nodata)):
        q.append((int(i), int(j)))

    # -------------------------------------------------------------------------
    # 6. Propagate accumulation downstream
    # -------------------------------------------------------------------------

    visited = 0

    while q:
        i, j = q.popleft()
        visited += 1

        if is_d8:
            # D8 mode: transfer the full accumulated value to one downstream cell.
            k = int(d_sane[i, j])
            if k < 0:
                continue

            di, dj = D8_OFFSETS[k]
            ni, nj = i + di, j + dj

            if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                acc[ni, nj] += acc[i, j]

                # One upstream dependency has now been resolved for the receiver.
                indeg[ni, nj] -= 1
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

        else:
            # MFD/FD8 mode: distribute the accumulated value proportionally
            # among all downstream receivers with positive routing weights.
            a = acc[i, j]

            for k, (di, dj) in enumerate(D8_OFFSETS):
                w = float(Wgt[i, j, k])
                if w <= 0.0:
                    continue

                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                    acc[ni, nj] += a * w

                    # One upstream dependency has now been resolved for the receiver.
                    indeg[ni, nj] -= 1
                    if indeg[ni, nj] == 0:
                        q.append((ni, nj))

    # -------------------------------------------------------------------------
    # 7. Optional cycle detection
    # -------------------------------------------------------------------------

    if cycle_check:
        # In a correctly conditioned DEM, all valid cells should be processed.
        if visited != int((~nodata).sum()):
            raise RuntimeError(
                "Cycle detected in flow graph. "
                "Check hydrological conditioning and flat resolution."
            )

    # -------------------------------------------------------------------------
    # 8. Convert output units if needed
    # -------------------------------------------------------------------------

    if out == "km2":
        acc *= 1e-6  # Convert m² to km²

    return acc.astype(np.float32)
    
