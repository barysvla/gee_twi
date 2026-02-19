from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np


# D8 neighbors in the order: [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[tuple[int, int]] = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def compute_flow_accumulation_d8(
    dir_idx: np.ndarray,
    *,
    nodata_mask: np.ndarray | None = None,
    pixel_area_m2: float | np.ndarray | None = None,
    out: Literal["cells", "m2", "km2"] = "km2",
    cycle_check: bool = True,
) -> np.ndarray:
    """
    Compute D8 flow accumulation by topological ordering (Kahn algorithm).

    Each valid cell contributes:
      - 1 (out='cells'), or
      - pixel_area_m2 (out='m2'/'km2'),
    and routes it to at most one D8 neighbor given by dir_idx.

    Direction encoding:
      0..7 = [NE, E, SE, S, SW, W, NW, N]
      -1   = No outflow (sink/outlet) OR NoData, depending on nodata_mask.
            If nodata_mask is provided, it is authoritative and -1 is treated
            as "no outflow" on valid cells.
    """
    d = np.asarray(dir_idx)
    if d.ndim != 2:
        raise ValueError("dir_idx must have shape (H, W).")
    H, W = d.shape

    # --- NoData mask ---------------------------------------------------------
    # If nodata_mask is provided, it defines invalid cells (recommended).
    # Otherwise we conservatively treat d == -1 as NoData because we cannot
    # distinguish "no outflow" from missing data without an external mask.
    if nodata_mask is None:
        nodata = (d < 0)
    else:
        nodata = np.asarray(nodata_mask, dtype=bool)
        if nodata.shape != (H, W):
            raise ValueError("nodata_mask must have shape (H, W).")

    # --- Sanitize directions -------------------------------------------------
    # Keep only indices in [0..7] as valid outflow directions.
    # Values outside [0..7] are treated as "no outflow" (sink/outlet) for accumulation.
    d_sane = np.full((H, W), -1, dtype=np.int16)
    valid_dir = (~nodata) & (d >= 0) & (d < 8)
    d_sane[valid_dir] = d[valid_dir].astype(np.int16, copy=False)

    # --- Initialize per-cell contribution -----------------------------------
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

    # NoData cells contribute nothing and must not receive anything.
    acc[nodata] = 0.0

    # --- Build in-degree array ----------------------------------------------
    # indeg[y,x] = number of upstream cells flowing into this cell.
    indeg = np.zeros((H, W), dtype=np.int32)

    for i in range(H):
        for j in range(W):
            if nodata[i, j]:
                continue

            k = int(d_sane[i, j])
            if k < 0:
                # No outflow from this cell (sink/outlet): no outgoing edge.
                continue

            di, dj = D8_OFFSETS[k]
            ni, nj = i + di, j + dj

            # Count only edges that stay inside raster and point to a valid cell.
            if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                indeg[ni, nj] += 1

    # --- Initialize queue with sources --------------------------------------
    q: deque[tuple[int, int]] = deque()
    src = np.argwhere((indeg == 0) & (~nodata))
    for i, j in src:
        q.append((int(i), int(j)))

    # --- Topological propagation --------------------------------------------
    visited = 0
    while q:
        i, j = q.popleft()
        visited += 1

        k = int(d_sane[i, j])
        if k < 0:
            # Sink/outlet/no-outflow: nothing to propagate further.
            continue

        di, dj = D8_OFFSETS[k]
        ni, nj = i + di, j + dj

        if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
            acc[ni, nj] += acc[i, j]

            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                q.append((ni, nj))

    # --- Cycle detection -----------------------------------------------------
    if cycle_check:
        total_valid = int((~nodata).sum())
        if visited != total_valid:
            raise RuntimeError(
                "Cycle detected (unresolved flats/sinks or inconsistent directions). "
                "Run flat resolution / hydrological conditioning before accumulation."
            )

    # --- Unit conversion -----------------------------------------------------
    if out == "km2":
        acc *= 1e-6

    return acc.astype(np.float32)
