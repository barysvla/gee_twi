from __future__ import annotations

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


def compute_flow_direction_d8(
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

    The procedure consists of the following steps:

    Step 0
        Validate inputs and normalize the DEM array.

    Step 1
        Derive the NoData mask and define the valid computational domain.

    Step 2
        Compute neighbour distances in CRS units based on the raster
        transform.

    Step 3
        For each valid cell, evaluate all valid neighbours and select
        the direction with the maximum downslope gradient.

    Step 4
        Assign direction indices and restore NoData values.

    Directions are encoded as:
        0..7 = [NE, E, SE, S, SW, W, NW, N]
        -1   = NoData or no valid downslope neighbour

    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    transform : affine.Affine
        Affine transform describing raster georeferencing. Only north-up
        grids (no rotation or shear) are supported.
    nodata_mask : np.ndarray of shape (H, W), optional
        Boolean mask indicating invalid cells (True = NoData).
    nodata_value : float, optional
        Explicit NoData value used if `nodata_mask` is not provided.
    out_dtype : data-type, default=np.int16
        Output data type for direction indices.

    Returns
    -------
    dir_idx : np.ndarray of shape (H, W)
        D8 flow-direction indices.

    References
    ----------
    O'Callaghan, J. F., & Mark, D. M. (1984).
    The extraction of drainage networks from digital elevation data.
    Computer Vision, Graphics, and Image Processing, 28(3), 323–344.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate input and normalize DEM
    # ---------------------------------------------------------------------
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    h, w = z.shape

    # This implementation assumes a north-up raster (no rotation or shear).
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
    # Step 2: Compute neighbour distances in CRS units
    # ---------------------------------------------------------------------
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    d_diag = float(np.hypot(dx, dy))

    # Step lengths aligned with D8_OFFSETS ordering
    step_len = np.array(
        [d_diag, dx, d_diag, dy, d_diag, dx, d_diag, dy],
        dtype=np.float64,
    )

    # ---------------------------------------------------------------------
    # Step 3: Evaluate downslope directions
    # ---------------------------------------------------------------------
    dir_idx = np.full((h, w), -1, dtype=out_dtype)

    for i in range(h):
        for j in range(w):
            if nodata[i, j]:
                continue

            zc = z[i, j]

            best_k = -1
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
                if d <= 0.0:
                    continue

                tan_beta = dz / d
                if tan_beta > best_tan:
                    best_tan = tan_beta
                    best_k = k

            dir_idx[i, j] = best_k

    # ---------------------------------------------------------------------
    # Step 4: Restore NoData values
    # ---------------------------------------------------------------------
    dir_idx[nodata] = -1

    return dir_idx
