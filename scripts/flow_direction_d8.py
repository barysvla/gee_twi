from __future__ import annotations

"""
D8 flow-direction computation for DEM-based flow routing.

This script computes flow directions in a raster DEM using the D8
single-flow-direction approach, where each cell is assigned to one
downslope neighbour. It is used after hydrological conditioning to
define the drainage network for subsequent flow accumulation.
"""

from typing import Tuple

import numpy as np

# D8 neighbourhood offsets.
# The order is fixed because the index is used as the encoded
# flow-direction value:
# 0..7 = [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[Tuple[int, int]] = [
    (-1, 1), (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1), (-1, -1), (-1, 0),
]

# Special flow-direction codes.
NODATA_DIR: int = -1
NOFLOW_DIR: int = -2


def flow_dir_d8(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
    nodata_value: float | None = None,
    out_dtype=np.int16,
) -> np.ndarray:
    """
    Compute D8 flow directions for a raster DEM.

    For each cell, the downslope neighbour maximizing

        tan(beta) = (z_center - z_neighbour) / d

    is selected within the 8-neighbourhood, where d is the distance
    between cell centres expressed in CRS units.

    The D8 concept originates from O'Callaghan and Mark (1984), where
    flow is routed from each cell to a single downslope neighbour,
    forming a directed drainage network over the raster.

    Directions are encoded as:
        0..7 = [NE, E, SE, S, SW, W, NW, N]
        -1   = NoData
        -2   = valid cell with no downslope neighbour

    Returns
    -------
    dir_idx : np.ndarray
        D8 flow-direction indices.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate input and normalize DEM
    # ---------------------------------------------------------------------
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    h, w = z.shape

    if getattr(transform, "b", 0.0) != 0.0 or getattr(transform, "d", 0.0) != 0.0:
        raise ValueError("Rotated or sheared transforms are not supported.")

    # ---------------------------------------------------------------------
    # Step 1: Derive NoData mask
    # ---------------------------------------------------------------------
    if nodata_mask is not None:
        nodata = np.asarray(nodata_mask, dtype=bool)
        if nodata.shape != (h, w):
            raise ValueError("nodata_mask must have shape (H, W).")
    else:
        if nodata_value is not None:
            nodata = (~np.isfinite(z)) | (z == float(nodata_value))
        else:
            nodata = ~np.isfinite(z)

    # ---------------------------------------------------------------------
    # Step 2: Compute neighbour distances
    # ---------------------------------------------------------------------
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    d_diag = float(np.hypot(dx, dy))

    step_len = np.array(
        [d_diag, dx, d_diag, dy, d_diag, dx, d_diag, dy],
        dtype=np.float64,
    )

    # ---------------------------------------------------------------------
    # Step 3: Evaluate downslope directions
    # ---------------------------------------------------------------------
    dir_idx = np.full((h, w), NOFLOW_DIR, dtype=out_dtype)

    for i in range(h):
        for j in range(w):
            if nodata[i, j]:
                continue

            zc = z[i, j]

            best_k = NOFLOW_DIR
            best_tan = -np.inf

            for k, (di, dj) in enumerate(D8_OFFSETS):
                ni, nj = i + di, j + dj

                if not (0 <= ni < h and 0 <= nj < w):
                    continue
                if nodata[ni, nj]:
                    continue

                dz = zc - z[ni, nj]
                if dz <= 0.0:
                    continue

                d = step_len[k]
                tan_beta = dz / d

                if tan_beta > best_tan:
                    best_tan = tan_beta
                    best_k = k

            dir_idx[i, j] = best_k

    # ---------------------------------------------------------------------
    # Step 4: Assign NoData code
    # ---------------------------------------------------------------------
    dir_idx[nodata] = NODATA_DIR

    return dir_idx
