from __future__ import annotations

import numpy as np


# D8 neighbor offsets in fixed order: NE, E, SE, S, SW, W, NW, N
D8_OFFSETS = np.array(
    [
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
        (-1,  0),  # N
    ],
    dtype=np.int32,
)


def compute_flow_direction_d8(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
    nodata_value: float | None = None,
    min_slope: float = 0.0,
    out_dtype=np.int16,
) -> np.ndarray:
    """
    Compute D8 flow directions on a DEM using grid-unit distances (CRS units).

    For each cell, the downslope neighbor maximizing:
        tan(beta) = (z_center - z_neighbor) / d
    is selected in an 8-neighborhood, where d is measured in CRS units
    (for EPSG:4326 this means degrees).

    Directions are encoded as integers:
        0..7 = [NE, E, SE, S, SW, W, NW, N]
        -1   = NoData (and also "no downslope neighbor" if it occurs).
    """
    # --- Input normalization -------------------------------------------------
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")
    h, w = z.shape

    # --- Validate transform assumptions -------------------------------------
    # This implementation assumes a north-up raster (no rotation/shear).
    if getattr(transform, "b", 0.0) != 0.0 or getattr(transform, "d", 0.0) != 0.0:
        raise ValueError("Rotated/sheared transforms are not supported (expected north-up grid).")

    # --- NoData mask ---------------------------------------------------------
    if nodata_mask is not None:
        nodata = np.asarray(nodata_mask, dtype=bool)
    else:
        if nodata_value is not None:
            nodata = ~np.isfinite(z) | (z == float(nodata_value))
        else:
            nodata = ~np.isfinite(z)

    # --- Grid-unit neighbor distances ---------------------------------------
    # Distances are measured in CRS units (EPSG:4326 -> degrees).
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    d_diag = float(np.hypot(dx, dy))

    # Step lengths aligned with D8_OFFSETS ordering:
    # [NE, E, SE, S, SW, W, NW, N]
    step_len = np.array([d_diag, dx, d_diag, dy, d_diag, dx, d_diag, dy], dtype=np.float64)

    # --- Output allocation ---------------------------------------------------
    dir_idx = np.full((h, w), -1, dtype=out_dtype)

    # --- Main scan over raster cells -----------------------------------------
    for i in range(h):
        for j in range(w):
            if nodata[i, j]:
                continue

            zc = z[i, j]

            best_k = -1
            best_tan = -np.inf

            for k in range(8):
                di, dj = int(D8_OFFSETS[k, 0]), int(D8_OFFSETS[k, 1])
                ni, nj = i + di, j + dj

                # Bounds check: neighbors outside raster are ignored.
                if not (0 <= ni < h and 0 <= nj < w):
                    continue

                # Ignore NoData neighbors (treat as absent).
                if nodata[ni, nj]:
                    continue

                # Downslope requirement: only strictly lower neighbors are eligible.
                dz = zc - z[ni, nj]
                if dz <= 0.0:
                    continue

                d = float(step_len[k])
                if d <= 0.0:
                    continue

                tan_beta = dz / d
                if tan_beta <= float(min_slope):
                    continue

                if tan_beta > best_tan:
                    best_tan = tan_beta
                    best_k = k

            dir_idx[i, j] = best_k if best_k != -1 else -1

    dir_idx[nodata] = -1
    return dir_idx
