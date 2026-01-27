from __future__ import annotations

import numpy as np


# --- Helpers: meters per degree & per-row step lengths (EPSG:4326) ---
def meters_per_degree_lat_lon(lat_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Meters per 1 degree of latitude/longitude at given latitude(s), WGS84 ellipsoid."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat)
    c = np.cos(lat)

    one_minus_e2s2 = 1.0 - e2 * s * s
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)  # meridional radius
    N = a / np.sqrt(one_minus_e2s2)               # prime vertical radius

    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows_epsg4326(transform, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row step lengths between pixel centers in meters for a north-up EPSG:4326 grid.
    Assumes rasterio-like affine with a>0, e<0, f=top-left latitude.
    """
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    lat_origin = float(transform.f)
    lat_step = float(transform.e)  # negative for north-up
    lat_centers = lat_origin + lat_step * (np.arange(height, dtype=np.float64) + 0.5)

    mlat, mlon = meters_per_degree_lat_lon(lat_centers)
    dx = mlon * deg_x
    dy = mlat * deg_y
    d_diag = np.hypot(dx, dy)
    return dx.astype(np.float64), dy.astype(np.float64), d_diag.astype(np.float64)


# D8 neighbors ordered: [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]


def qin2007_flow_partition_exponent(e: float) -> float:
    """
    Qin et al. (2007) Equation (4): f(e) = 8.9 * min(e, 1) + 1.1
    where e is the maximum downslope gradient in tan units (max tan(beta)).
    """
    return 8.9 * min(float(e), 1.0) + 1.1


def compute_flow_direction_qin_2007(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Qin et al. (2007) MFD-md (maximum downslope gradient) weights in an FD8 neighborhood.

    Weights follow Eq. (5) with Eq. (4):
        d_i = L_i * (tanβ_i)^(f(e)) / Σ_j [ L_j * (tanβ_j)^(f(e)) ]
    where:
        - tanβ_i = (z_center - z_neighbor) / distance_i   (downslope only)
        - e = max_i(tanβ_i) over downslope neighbors
        - f(e) = 8.9 * min(e, 1) + 1.1   (p in [1.1, 10])

    L_i follows Quinn et al. (1991) effective contour length as used by Qin et al. (2007):
        - cardinal directions: 0.5
        - diagonal directions: 0.354

    Returns
    -------
    flow : (H, W, 8) float32 weights to [NE, E, SE, S, SW, W, NW, N].
           Sum per cell is 1 where at least one downslope neighbor exists; else all zeros.
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Per-row metric step lengths (EPSG:4326 degrees -> meters).
    dx, dy, ddiag = step_lengths_for_rows_epsg4326(transform, H)

    # Quinn et al. (1991) effective contour length constants used by Qin et al. (2007).
    L_card = 0.5
    L_diag = 0.354
    L_vec = np.array([L_diag, L_card, L_diag, L_card, L_diag, L_card, L_diag, L_card], dtype=np.float64)

    flow = np.zeros((H, W, 8), dtype=np.float32)

    for i in range(H):
        # Planimetric distances to neighbors for this row (meters), order NE..N
        step_len = np.array(
            [ddiag[i], dx[i], ddiag[i], dy[i], ddiag[i], dx[i], ddiag[i], dy[i]],
            dtype=np.float64,
        )

        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]

            tanb = np.zeros(8, dtype=np.float64)
            for k, (di, dj) in enumerate(D8_OFFSETS):
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if nodata_mask[ni, nj]:
                    continue

                d = float(step_len[k])
                if d <= 0.0:
                    continue

                dz = zc - Z[ni, nj]
                if dz <= 0.0:
                    continue  # downslope only

                tb = dz / d  # tan(beta)
                if np.isfinite(tb) and tb > 0.0:
                    tanb[k] = tb

            e = float(np.max(tanb))
            if e <= 0.0:
                continue  # no downslope neighbor

            p = qin2007_flow_partition_exponent(e)

            with np.errstate(over="ignore", invalid="ignore"):
                w = L_vec * np.power(tanb, p)  # no dtype argument; tanb is float64

            w[~np.isfinite(w)] = 0.0
            w[w < 0.0] = 0.0

            s = float(w.sum())
            if s > 0.0:
                flow[i, j, :] = (w / s).astype(np.float32)

    flow[nodata_mask, :] = 0.0
    return flow
