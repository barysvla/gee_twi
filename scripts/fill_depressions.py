import heapq
import numpy as np

# D8 neighbourhood offsets used for raster traversal.
NEIGHBOR_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def priority_flood_fill(
    dem: np.ndarray,
    nodata: float = np.nan,
    seed_internal_nodata_as_outlet: bool = True,
    return_fill_depth: bool = False,
):
    """
    Priority-Flood depression filling algorithm.

    This function implements the basic Priority-Flood algorithm for
    removal of closed depressions in a digital elevation model (DEM),
    as described by Barnes, Lehman and Mulla (2014). The implementation
    follows the logic of the reference C++ implementation
    `original_priority_flood` provided by the authors and corresponds
    to the basic variant of the algorithm (Algorithm 1), which uses a
    single priority queue.

    The algorithm floods the DEM inward from the raster boundary.
    Cells are processed in order of increasing elevation using a
    priority queue. If a neighbouring cell lies below the current
    spill elevation, its value is raised so that every valid cell
    has an outlet and no closed depressions remain.

    This implementation additionally supports optional seeding of
    cells adjacent to internal NoData regions so that masked areas
    can act as potential outlets when working with real-world DEMs.

    Parameters
    ----------
    dem : np.ndarray
        2D array representing DEM elevations.
    nodata : float
        NoData marker. Use np.nan if missing data are stored as NaN.
    seed_internal_nodata_as_outlet : bool
        If True, cells adjacent to internal NoData regions are treated
        as potential outlets and are seeded into the priority queue.
    return_fill_depth : bool
        If True, also return the per-cell fill depth (filled minus original elevation).

    Returns
    -------
    dem_filled : np.ndarray
        Depression-filled DEM with NoData preserved.
    fill_depth : np.ndarray, optional
        Fill depth per cell. Returned only if return_fill_depth=True.

    Reference
    ---------
    Barnes, R., Lehman, C., Mulla, D. (2014).
    Priority-Flood: An optimal depression-filling and watershed-labeling
    algorithm for digital elevation models.
    Computers & Geosciences, 62, 117–127.
    """

    dem_values = np.asarray(dem, dtype=np.float64)
    if dem_values.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    n_rows, n_cols = dem_values.shape

    # Identify cells that participate in the hydrologic correction.
    if np.isnan(nodata):
        valid_mask = np.isfinite(dem_values)
    else:
        valid_mask = (dem_values != nodata) & np.isfinite(dem_values)

    dem_filled = dem_values.copy()
    visited_mask = np.zeros_like(valid_mask, dtype=bool)
    priority_queue = []  # Heap entries are stored as (elevation, row, col).

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < n_rows and 0 <= c < n_cols

    # Seed valid raster-edge cells, which act as natural outlets.
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

    # Optionally treat cells adjacent to internal NoData regions as additional outlets.
    if seed_internal_nodata_as_outlet:
        for r in range(n_rows):
            for c in range(n_cols):
                if not valid_mask[r, c] or visited_mask[r, c]:
                    continue

                for dr, dc in NEIGHBOR_OFFSETS_8:
                    rr, cc = r + dr, c + dc
                    if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]):
                        heapq.heappush(priority_queue, (dem_filled[r, c], r, c))
                        visited_mask[r, c] = True
                        break

    # Propagate inward from the lowest queued cell and fill depressions as needed.
    while priority_queue:
        water_level, r, c = heapq.heappop(priority_queue)

        for dr, dc in NEIGHBOR_OFFSETS_8:
            rr, cc = r + dr, c + dc
            if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]) or visited_mask[rr, cc]:
                continue

            visited_mask[rr, cc] = True

            # Enforce that newly reached cells are not lower than the current spill elevation.
            if dem_filled[rr, cc] < water_level:
                dem_filled[rr, cc] = water_level

            heapq.heappush(priority_queue, (dem_filled[rr, cc], rr, cc))

    # Restore the original NoData mask in the output raster.
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
    
