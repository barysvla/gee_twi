from __future__ import annotations

import numpy as np


def meters_per_degree_lat_lon(lat_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute meters per 1 degree of latitude/longitude at given latitude(s) using WGS84 ellipsoid.

    Parameters
    ----------
    lat_deg : array-like or float
        Latitude in degrees.

    Returns
    -------
    m_per_deg_lat : np.ndarray
        Meters per degree of latitude.
    m_per_deg_lon : np.ndarray
        Meters per degree of longitude.
    """
    a = 6378137.0  # WGS84 semi-major axis [m]
    f = 1.0 / 298.257223563  # WGS84 flattening
    e2 = (2.0 - f) * f  # eccentricity squared

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat)
    c = np.cos(lat)

    one_minus_e2s2 = 1.0 - e2 * s * s
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)  # meridional radius of curvature
    N = a / np.sqrt(one_minus_e2s2)               # prime vertical radius of curvature

    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows(transform, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row step lengths between pixel centers for a north-up grid in EPSG:4326.

    Notes
    -----
    This is only meaningful when the grid is in geographic coordinates (degrees).
    For projected CRS (meters), you should instead use constant step lengths derived from transform.

    Parameters
    ----------
    transform : rasterio.Affine-like
        Affine transform (north-up assumed).
    height : int
        Raster height (number of rows).

    Returns
    -------
    dx : np.ndarray
        East/West step length (meters) per row.
    dy : np.ndarray
        North/South step length (meters) per row.
    d_diag : np.ndarray
        Diagonal step length (meters) per row.
    """
    # Pixel sizes in degrees (north-up assumed: a>0, e<0)
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    # Latitude of the top-left corner and per-row step (negative for north-up)
    lat_origin = float(transform.f)
    lat_step = float(transform.e)

    # Row center latitudes (degrees)
    lat_centers_deg = lat_origin + lat_step * (np.arange(height, dtype=np.float64) + 0.5)

    mlat, mlon = meters_per_degree_lat_lon(lat_centers_deg)
    dx = mlon * deg_x
    dy = mlat * deg_y
    d_diag = np.hypot(dx, dy)

    return dx.astype(np.float64), dy.astype(np.float64), d_diag.astype(np.float64)


# D8 neighbor offsets in order: NE, E, SE, S, SW, W, NW, N
D8_OFFSETS = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def compute_flow_direction_quinn_1991(
    dem: np.ndarray,
    transform,
    *,
    p: float = 1.0,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Multi-flow direction (MFD) routing after Quinn et al. (1991) in an FD8-style neighborhood.

    For each cell, positive slopes to all downslope neighbors are computed and converted to weights:
        w_k ∝ L_k * (tan(beta_k))^p
    where:
        - tan(beta_k) = (z_center - z_neighbor) / d_k
        - d_k is the planimetric distance between cell centers
        - L_k is an effective contour-length factor (Quinn 1991 FD8 constants)
        - p=1 corresponds to Quinn (1991); p>1 is a Holmgren-style generalization.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM array. NoData should be NaN (recommended).
    transform : rasterio.Affine-like
        Affine transform (north-up). Used to derive step lengths in meters for EPSG:4326 grids.
    p : float, default=1.0
        Exponent controlling flow dispersion across downslope neighbors.
    nodata_mask : np.ndarray | None
        Boolean mask (True = invalid). If None, derived as ~isfinite(dem).

    Returns
    -------
    flow_weights : np.ndarray
        Array of shape (H, W, 8) with float32 weights for directions [NE, E, SE, S, SW, W, NW, N].
        Weights sum to 1 for cells with at least one downslope neighbor; otherwise all zeros.
    """
    z = np.asarray(dem, dtype=np.float64)
    height, width = z.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Per-row step lengths (meters) for geographic CRS (EPSG:4326).
    # If you later support projected CRS, replace this with constant dx/dy from transform.
    dx_row, dy_row, d_diag_row = step_lengths_for_rows(transform, height)

    # Effective contour-length factors (Quinn 1991 / FD8 constants)
    L_cardinal = 0.5
    L_diagonal = np.sqrt(2.0) / 4.0
    # Map to D8 order: NE(diag), E(card), SE(diag), S(card), SW(diag), W(card), NW(diag), N(card)
    L = np.array(
        [L_diagonal, L_cardinal, L_diagonal, L_cardinal, L_diagonal, L_cardinal, L_diagonal, L_cardinal],
        dtype=np.float64,
    )

    flow_weights = np.zeros((height, width, 8), dtype=np.float32)

    for i in range(height):
        # Step lengths to D8 neighbors in this row (meters)
        step_len = np.array(
            [d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i],
             d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i]],
            dtype=np.float64,
        )

        for j in range(width):
            if nodata_mask[i, j]:
                continue

            zc = z[i, j]
            weights = []
            indices = []

            for k, (di, dj) in enumerate(D8_OFFSETS):
                ni, nj = i + di, j + dj

                # Bounds + NoData check
                if not (0 <= ni < height and 0 <= nj < width):
                    continue
                if nodata_mask[ni, nj]:
                    continue

                d = float(step_len[k])
                if d <= 0.0:
                    continue

                dz = zc - z[ni, nj]
                if dz <= 0.0:
                    continue  # only positive slopes (downslope neighbors)

                tan_beta = dz / d
                w = L[k] * (tan_beta ** float(p))
                if w > 0.0:
                    weights.append(w)
                    indices.append(k)

            if not indices:
                continue

            s = float(np.sum(weights))
            if s > 0.0:
                inds = np.asarray(indices, dtype=int)
                flow_weights[i, j, inds] = (np.asarray(weights, dtype=np.float64) / s).astype(np.float32)

    # Keep NoData cells as zeros explicitly
    flow_weights[nodata_mask, :] = 0.0
    return flow_weights
