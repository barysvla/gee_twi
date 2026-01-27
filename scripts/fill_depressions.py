import heapq
import numpy as np

# Neighbor offsets for a full 3x3 neighborhood (including diagonals)
NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def priority_flood_fill(
    dem: np.ndarray,
    nodata: float = np.nan,
    seed_internal_nodata_as_outlet: bool = True,
    return_fill_depth: bool = True,
):
    """
    Priority-Flood (Barnes et al.) depression filling.

    The algorithm:
    - floods from the raster boundary (and optionally from neighbors of internal NoData islands),
    - uses a priority queue keyed by the current "water level" (cell elevation),
    - guarantees that every valid cell has an outlet (no closed depressions remain).

    Parameters
    ----------
    dem : np.ndarray
        2D DEM array (float-like).
    nodata : float
        NoData marker; use np.nan if NoData is represented as NaN.
    seed_internal_nodata_as_outlet : bool
        If True, treat internal NoData regions as potential outlets by seeding their valid neighbors.
        This is robust for real DEMs with masks/gaps.
    return_fill_depth : bool
        If True, also return per-cell fill depth (filled - original).

    Returns
    -------
    dem_filled : np.ndarray
        Depression-filled DEM (float64), with NoData preserved.
    fill_depth : np.ndarray, optional
        Fill depth per cell (float64). Returned only if return_fill_depth=True.
    """
    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    n_rows, n_cols = dem_values.shape

    # Valid-data mask
    if np.isnan(nodata):
        valid_mask = np.isfinite(dem_values)
    else:
        valid_mask = (dem_values != nodata) & np.isfinite(dem_values)

    dem_filled = dem_values.copy()
    visited_mask = np.zeros_like(valid_mask, dtype=bool)
    priority_queue = []  # entries: (elevation, row, col)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < n_rows and 0 <= c < n_cols

    # 1) Seed: all valid border cells of the raster (natural outlets)
    for c in range(n_cols):
        if valid_mask[0, c] and not visited_mask[0, c]:
            heapq.heappush(priority_queue, (dem_filled[0, c], 0, c))
            visited_mask[0, c] = True
        if valid_mask[n_rows - 1, c] and not visited_mask[n_rows - 1, c]:
            heapq.heappush(priority_queue, (dem_filled[n_rows - 1, c], n_rows - 1, c))
            visited_mask[n_rows - 1, c] = True

    for r in range(1, n_rows - 1):
        if valid_mask[r, 0] and not visited_mask[r, 0]:
            heapq.heappush(priority_queue, (dem_filled[r, 0], r, 0))
            visited_mask[r, 0] = True
        if valid_mask[r, n_cols - 1] and not visited_mask[r, n_cols - 1]:
            heapq.heappush(priority_queue, (dem_filled[r, n_cols - 1], r, n_cols - 1))
            visited_mask[r, n_cols - 1] = True

    # 2) Optional seed: valid cells adjacent to NoData (treat internal NoData as outlets)
    if seed_internal_nodata_as_outlet:
        for r in range(n_rows):
            for c in range(n_cols):
                if not valid_mask[r, c] or visited_mask[r, c]:
                    continue

                for dr, dc in NEIGHBOR_OFFSETS_8:
                    rr, cc = r + dr, c + dc
                    # Neighbor outside raster OR neighbor is NoData -> seed this cell
                    if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]):
                        heapq.heappush(priority_queue, (dem_filled[r, c], r, c))
                        visited_mask[r, c] = True
                        break

    # 3) Main loop: always expand from the currently lowest water level
    while priority_queue:
        water_level, r, c = heapq.heappop(priority_queue)

        for dr, dc in NEIGHBOR_OFFSETS_8:
            rr, cc = r + dr, c + dc
            if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]) or visited_mask[rr, cc]:
                continue

            visited_mask[rr, cc] = True

            # Raise neighbor if it lies below the current water level
            if dem_filled[rr, cc] < water_level:
                dem_filled[rr, cc] = water_level

            heapq.heappush(priority_queue, (dem_filled[rr, cc], rr, cc))

    # Preserve NoData values
    if np.isnan(nodata):
        dem_filled[~valid_mask] = np.nan
    else:
        dem_filled[~valid_mask] = float(nodata)

    if not return_fill_depth:
        return dem_filled

    fill_depth = dem_filled - dem_values
    if np.isnan(nodata):
        fill_depth[~valid_mask] = np.nan
    else:
        fill_depth[~valid_mask] = 0.0

    return dem_filled, fill_depth
