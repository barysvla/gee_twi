from __future__ import annotations

from typing import Tuple

import numpy as np


def meters_per_degree_lat_lon(
    lat_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute metres per degree of latitude and longitude on the WGS84 ellipsoid.

    This function evaluates the meridional and prime-vertical curvature
    radii of the WGS84 ellipsoid and converts them to linear distances
    corresponding to one degree of latitude and longitude at the given
    latitude.

    Parameters
    ----------
    lat_deg : np.ndarray or float
        Latitude in degrees.

    Returns
    -------
    m_per_deg_lat : np.ndarray
        Metres per degree of latitude.
    m_per_deg_lon : np.ndarray
        Metres per degree of longitude.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat)
    c = np.cos(lat)

    one_minus_e2s2 = 1.0 - e2 * s * s
    m = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)
    n = a / np.sqrt(one_minus_e2s2)

    m_per_deg_lat = (np.pi / 180.0) * m
    m_per_deg_lon = (np.pi / 180.0) * n * np.clip(c, 0.0, None)

    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows_epsg4326(
    transform,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row D8 step lengths for a north-up raster in EPSG:4326.

    The step lengths are evaluated at row-centre latitudes and expressed
    in metres. East-west, north-south, and diagonal distances are
    returned separately.

    Parameters
    ----------
    transform : affine.Affine
        Affine transform describing raster georeferencing. Only north-up
        grids are assumed.
    height : int
        Raster height in rows.

    Returns
    -------
    dx : np.ndarray
        East-west step length in metres for each row.
    dy : np.ndarray
        North-south step length in metres for each row.
    d_diag : np.ndarray
        Diagonal step length in metres for each row.
    """
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    lat_origin = float(transform.f)
    lat_step = float(transform.e)

    lat_centres_deg = lat_origin + lat_step * (np.arange(height, dtype=np.float64) + 0.5)

    mlat, mlon = meters_per_degree_lat_lon(lat_centres_deg)
    dx = mlon * deg_x
    dy = mlat * deg_y
    d_diag = np.hypot(dx, dy)

    return dx.astype(np.float64), dy.astype(np.float64), d_diag.astype(np.float64)


# D8 neighbourhood offsets.
# The order is fixed because the index is used as the encoded
# flow-direction value:
# 0..7 = [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS: list[Tuple[int, int]] = [
    (-1, 1), (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1), (-1, -1), (-1, 0),
]


def compute_flow_direction_mfd_quinn_1991(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
    assume_epsg4326: bool = True,
    min_slope: float = 0.0,
) -> np.ndarray:
    """
    Compute multi-flow routing weights following Quinn et al. (1991).

    This function computes multi-flow direction (MFD) routing weights in
    an FD8-style neighbourhood. For each cell, positive slopes to all
    downslope neighbours are evaluated and converted to routing weights

        w_k ∝ L_k * tan(beta_k)

    where `tan(beta_k)` is the local downslope gradient toward neighbour
    `k`, `d_k` is the distance between cell centres, and `L_k` is the
    effective contour-length factor. This formulation corresponds to
    Quinn et al. (1991).

    The procedure consists of the following steps:

    Step 0
        Validate inputs, normalize the DEM array, and define the valid
        computational domain.

    Step 1
        Compute D8 step lengths in metres, either per row for EPSG:4326
        rasters or as constant values for projected rasters.

    Step 2
        Define the effective contour-length factors for the FD8-style
        neighbourhood.

    Step 3
        For each valid cell, evaluate all valid downslope neighbours and
        compute unnormalized routing weights.

    Step 4
        Normalize routing weights so that they sum to 1 for cells with
        at least one downslope neighbour.

    Step 5
        Assign zero weights to NoData cells and return the output array.

    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    transform : affine.Affine
        Affine transform describing raster georeferencing. If
        `assume_epsg4326=True`, transform units are assumed to be
        degrees. Otherwise, transform units are assumed to be projected
        linear units.
    nodata_mask : np.ndarray of shape (H, W), optional
        Boolean mask indicating invalid cells (True = NoData). If None,
        it is derived from non-finite DEM values.
    assume_epsg4326 : bool, default=True
        If True, compute row-dependent metric step lengths for a raster
        in geographic coordinates. If False, use constant projected step
        lengths.
    min_slope : float, default=0.0
        Minimum admissible value of `tan(beta)`. Slopes less than or
        equal to this threshold are ignored.

    Returns
    -------
    flow_weights : np.ndarray of shape (H, W, 8), dtype float32
        Routing weights for directions [NE, E, SE, S, SW, W, NW, N].
        Weights sum to 1 for cells with at least one downslope
        neighbour. Cells without downslope neighbours contain only
        zeros.

    References
    ----------
    Quinn, P., Beven, K., Chevallier, P., & Planchon, O. (1991).
    The prediction of hillslope flow paths for distributed hydrological
    modelling using digital terrain models.
    Hydrological Processes, 5(1), 59–79.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate inputs and define the valid domain
    # ---------------------------------------------------------------------
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    height, width = z.shape

    if nodata_mask is None:
        nodata = ~np.isfinite(z)
    else:
        nodata = np.asarray(nodata_mask, dtype=bool)
        if nodata.shape != (height, width):
            raise ValueError("nodata_mask must have shape (H, W).")

    if min_slope < 0.0:
        raise ValueError("min_slope must be non-negative.")

    # ---------------------------------------------------------------------
    # Step 1: Compute D8 step lengths in metres
    # ---------------------------------------------------------------------
    if assume_epsg4326:
        dx_row, dy_row, d_diag_row = step_lengths_for_rows_epsg4326(transform, height)
    else:
        dx = float(abs(transform.a))
        dy = float(abs(transform.e))
        dx_row = np.full(height, dx, dtype=np.float64)
        dy_row = np.full(height, dy, dtype=np.float64)
        d_diag_row = np.full(height, float(np.hypot(dx, dy)), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Step 2: Define effective contour-length factors
    # ---------------------------------------------------------------------
    l_cardinal = 0.5
    l_diagonal = np.sqrt(2.0) / 4.0
    contour_length = np.array(
        [
            l_diagonal, l_cardinal, l_diagonal, l_cardinal,
            l_diagonal, l_cardinal, l_diagonal, l_cardinal,
        ],
        dtype=np.float64,
    )

    # ---------------------------------------------------------------------
    # Step 3: Evaluate downslope neighbours and compute raw weights
    # ---------------------------------------------------------------------
    flow_weights = np.zeros((height, width, 8), dtype=np.float32)

    for i in range(height):
        step_len = np.array(
            [
                d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i],
                d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i],
            ],
            dtype=np.float64,
        )

        for j in range(width):
            if nodata[i, j]:
                continue

            zc = z[i, j]
            weights = np.zeros(8, dtype=np.float64)

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for k, (di, dj) in enumerate(D8_OFFSETS):
                    ni, nj = i + di, j + dj

                    if not (0 <= ni < height and 0 <= nj < width):
                        continue
                    if nodata[ni, nj]:
                        continue

                    distance = step_len[k]
                    if distance <= 0.0:
                        continue

                    dz = zc - z[ni, nj]
                    if dz <= 0.0:
                        continue

                    tan_beta = dz / distance
                    if not np.isfinite(tan_beta) or tan_beta <= min_slope:
                        continue

                    wk = contour_length[k] * tan_beta
                    if np.isfinite(wk) and wk > 0.0:
                        weights[k] = wk

            # -----------------------------------------------------------------
            # Step 4: Normalize routing weights
            # -----------------------------------------------------------------
            weight_sum = float(weights.sum())
            if weight_sum > 0.0:
                flow_weights[i, j, :] = (weights / weight_sum).astype(np.float32)

    # ---------------------------------------------------------------------
    # Step 5: Assign zero weights to NoData cells
    # ---------------------------------------------------------------------
    flow_weights[nodata, :] = 0.0

    return flow_weights
