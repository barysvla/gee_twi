from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np


# D8 neighbors in the order: [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[tuple[int, int]] = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def compute_flow_accumulation_mfd_fd8(
    flow_weights: np.ndarray,
    *,
    nodata_mask: np.ndarray | None = None,
    pixel_area_m2: float | np.ndarray | None = None,
    out: Literal["cells", "m2", "km2"] = "km2",
    renormalize: bool = False,
    cycle_check: bool = True,
) -> np.ndarray:
    """
    Topological flow accumulation for FD8/MFD weights (works for Quinn 1991, Qin 2007, etc.).

    Parameters
    ----------
    flow_weights
        Array (H, W, 8) with non-negative outflow weights to D8 neighbors:
        [NE, E, SE, S, SW, W, NW, N].
        For cells with outflow, weights should sum to 1 (per cell).
    nodata_mask
        Boolean array (H, W), True where invalid/NoData. If None, all valid.
    pixel_area_m2
        Required for out='m2' or out='km2'. Can be a scalar or an (H, W) array.
        Use per-cell area if working in EPSG:4326 with varying pixel area by latitude.
    out
        Output units:
        - 'cells' : contributing cell count (starts with 1 per valid cell)
        - 'm2'    : contributing area in square meters
        - 'km2'   : contributing area in square kilometers
    renormalize
        If True, renormalize positive weights per cell to sum to 1.
        Use only as a safety guard (e.g., float drift, external inputs).
    cycle_check
        If True, raise if a cycle is detected (typically unresolved flats/sinks).

    Returns
    -------
    acc : (H, W) float32
        Flow accumulation in requested units.
    """
    Wgt = np.asarray(flow_weights, dtype=np.float32)
    if Wgt.ndim != 3 or Wgt.shape[2] != 8:
        raise ValueError("flow_weights must have shape (H, W, 8).")
    H, W, _ = Wgt.shape

    # Prepare nodata mask
    if nodata_mask is None:
        nodata = np.zeros((H, W), dtype=bool)
    else:
        nodata = np.asarray(nodata_mask, dtype=bool)
        if nodata.shape != (H, W):
            raise ValueError("nodata_mask must have shape (H, W).")

    # Sanitize weights: replace NaNs/Infs, drop negatives, zero-out nodata cells
    Wgt = np.nan_to_num(Wgt, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(Wgt, 0.0, out=Wgt)
    Wgt[nodata, :] = 0.0

    # Optional per-cell renormalization (safety guard)
    if renormalize:
        sums = Wgt.sum(axis=2, dtype=np.float32)
        pos = (sums > 0.0) & (~nodata)
        Wgt[pos, :] /= sums[pos, None]

    # Initialize accumulation
    if out == "cells":
        acc = np.ones((H, W), dtype=np.float64)
    else:
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required for out='m2' or out='km2'.")
        if np.isscalar(pixel_area_m2):
            acc = np.full((H, W), float(pixel_area_m2), dtype=np.float64)
        else:
            pa = np.asarray(pixel_area_m2, dtype=np.float64)
            if pa.shape != (H, W):
                raise ValueError("pixel_area_m2 must be a scalar or have shape (H, W).")
            acc = pa.copy()

    acc[nodata] = 0.0

    # Build in-degree array for Kahn topological ordering
    indeg = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if nodata[i, j]:
                continue
            # Count downstream edges from (i,j) into valid cells
            for k, (di, dj) in enumerate(D8_OFFSETS):
                if Wgt[i, j, k] <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                    indeg[ni, nj] += 1

    # Initialize queue with sources (in-degree == 0)
    q: deque[tuple[int, int]] = deque()
    src = np.argwhere((indeg == 0) & (~nodata))
    for i, j in src:
        q.append((int(i), int(j)))

    visited = 0
    while q:
        i, j = q.popleft()
        visited += 1

        a = acc[i, j]
        if a == 0.0:
            # Still remove edges to keep topological progress correct
            for k, (di, dj) in enumerate(D8_OFFSETS):
                if Wgt[i, j, k] <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                    indeg[ni, nj] -= 1
                    if indeg[ni, nj] == 0:
                        q.append((ni, nj))
            continue

        # Propagate accumulated contribution downstream
        for k, (di, dj) in enumerate(D8_OFFSETS):
            w = float(Wgt[i, j, k])
            if w <= 0.0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                acc[ni, nj] += a * w
                indeg[ni, nj] -= 1
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

    if cycle_check:
        total_valid = int((~nodata).sum())
        if visited != total_valid:
            raise RuntimeError(
                "Cycle detected (unresolved flats/sinks). "
                "Run hydrological conditioning / flat resolution before accumulation."
            )

    if out == "km2":
        acc *= 1e-6

    return acc.astype(np.float32)
