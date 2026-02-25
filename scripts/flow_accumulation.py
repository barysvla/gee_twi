from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np


# D8 neighbors in the order: [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[tuple[int, int]] = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def compute_flow_accumulation(
    *,
    dir_idx: np.ndarray | None = None,
    flow_weights: np.ndarray | None = None,
    nodata_mask: np.ndarray | None = None,
    pixel_area_m2: float | np.ndarray | None = None,
    out: Literal["cells", "m2", "km2"] = "km2",
    cycle_check: bool = True,
) -> np.ndarray:
    """
    Unified topological flow accumulation for:
      - D8 (dir_idx: (H, W) int, 0..7)
      - MFD/FD8 (flow_weights: (H, W, 8) float, non-negative, per-cell sum ~ 1)

    Exactly one of dir_idx or flow_weights must be provided.

    Parameters
    ----------
    dir_idx
        D8 direction indices (H, W), encoding:
          0..7 = [NE, E, SE, S, SW, W, NW, N]
          -1   = no outflow (sink/outlet) OR NoData depending on nodata_mask
        If nodata_mask is None, values < 0 are treated as NoData (conservative).
    flow_weights
        FD8/MFD weights (H, W, 8) to neighbors [NE, E, SE, S, SW, W, NW, N].
        For outflow cells, weights should sum to 1.
    nodata_mask
        Boolean (H, W), True where invalid/NoData. If None:
          - D8: inferred as (dir_idx < 0)
          - MFD: assumed all valid
    pixel_area_m2
        Required for out='m2'/'km2'. Scalar or (H, W).
    out
        'cells' / 'm2' / 'km2'
    cycle_check
        If True, raise on cycles (typically unresolved flats/sinks/inconsistent directions).

    Returns
    -------
    acc : (H, W) float32
        Flow accumulation in requested units.
    """
    # --- Validate inputs -----------------------------------------------------
    if (dir_idx is None) == (flow_weights is None):
        raise ValueError("Provide exactly one of dir_idx or flow_weights.")

    is_d8 = dir_idx is not None

    if is_d8:
        d = np.asarray(dir_idx)
        if d.ndim != 2:
            raise ValueError("dir_idx must have shape (H, W).")
        H, W = d.shape

        if nodata_mask is None:
            # Conservative: cannot distinguish 'no outflow' vs 'nodata' without mask
            nodata = (d < 0)
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)
            if nodata.shape != (H, W):
                raise ValueError("nodata_mask must have shape (H, W).")

        # Sanitize directions: keep only [0..7], else treat as no outflow (-1)
        d_sane = np.full((H, W), -1, dtype=np.int16)
        valid_dir = (~nodata) & (d >= 0) & (d < 8)
        d_sane[valid_dir] = d[valid_dir].astype(np.int16, copy=False)

    else:
        Wgt = np.asarray(flow_weights, dtype=np.float32)
        if Wgt.ndim != 3 or Wgt.shape[2] != 8:
            raise ValueError("flow_weights must have shape (H, W, 8).")
        H, W, _ = Wgt.shape

        if nodata_mask is None:
            nodata = np.zeros((H, W), dtype=bool)
        else:
            nodata = np.asarray(nodata_mask, dtype=bool)
            if nodata.shape != (H, W):
                raise ValueError("nodata_mask must have shape (H, W).")

        # Sanitize weights
        Wgt = np.nan_to_num(Wgt, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(Wgt, 0.0, out=Wgt)
        Wgt[nodata, :] = 0.0

    # --- Initialize accumulation --------------------------------------------
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

    # --- Build in-degree array (number of upstream edges into each cell) ----
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

    # --- Initialize queue with sources (in-degree == 0) ----------------------
    q: deque[tuple[int, int]] = deque()
    for i, j in np.argwhere((indeg == 0) & (~nodata)):
        q.append((int(i), int(j)))

    # --- Topological propagation --------------------------------------------
    visited = 0
    while q:
        i, j = q.popleft()
        visited += 1

        if is_d8:
            k = int(d_sane[i, j])
            if k < 0:
                continue
            di, dj = D8_OFFSETS[k]
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                acc[ni, nj] += acc[i, j]
                indeg[ni, nj] -= 1
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

        else:
            a = acc[i, j]

            # Always remove edges to allow topological progress
            for k, (di, dj) in enumerate(D8_OFFSETS):
                w = float(Wgt[i, j, k])
                if w <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and (not nodata[ni, nj]):
                    if a != 0.0:
                        acc[ni, nj] += a * w
                    indeg[ni, nj] -= 1
                    if indeg[ni, nj] == 0:
                        q.append((ni, nj))

    # --- Cycle detection -----------------------------------------------------
    if cycle_check:
        total_valid = int((~nodata).sum())
        if visited != total_valid:
            raise RuntimeError(
                "Cycle detected (unresolved flats/sinks or inconsistent directions). "
                "Run hydrological conditioning / flat resolution before accumulation."
            )

    # --- Unit conversion -----------------------------------------------------
    if out == "km2":
        acc *= 1e-6

    return acc.astype(np.float32)
