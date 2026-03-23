from __future__ import annotations

import heapq
from typing import List, Tuple, Union

import numpy as np

# D8 neighbourhood offsets.
NEIGHBOR_OFFSETS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def fill_depressions(
    dem: np.ndarray,
    nodata: float = np.nan,
    seed_internal_nodata_as_outlet: bool = True,
    return_fill_depth: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Fill closed depressions in a raster DEM using the Priority-Flood algorithm.

    This function implements the basic Priority-Flood algorithm for
    depression filling in digital elevation models, following Barnes,
    Lehman, and Mulla (2014). The implementation corresponds to the
    basic variant based on a single priority queue.

    The algorithm processes cells in order of increasing elevation,
    starting from outlet cells. If a newly reached cell lies below the
    current spill elevation, its value is raised to the spill level.
    This ensures that each valid cell drains to an outlet and that
    closed depressions are removed.

    The procedure consists of the following steps:

    Step 0
        Prepare the DEM, derive the valid-data mask, and initialize the
        priority queue and visited mask.

    Step 1
        Seed all valid raster-edge cells into the priority queue.

    Step 2
        Optionally seed valid cells adjacent to internal NoData regions,
        allowing masked regions to act as additional outlets.

    Step 3
        Propagate inward from the lowest queued cell and raise neighbour
        elevations where required to remove closed depressions.

    Step 4
        Restore the original NoData mask in the output raster and
        optionally compute per-cell fill depth.

    Relative to the reference implementation, this version adds optional
    handling of internal NoData-adjacent cells as outlet seeds for
    finite and masked raster domains.

    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    nodata : float, default=np.nan
        NoData marker. If set to NaN, all non-finite values are treated
        as invalid.
    seed_internal_nodata_as_outlet : bool, default=True
        If True, valid cells adjacent to internal NoData regions are
        treated as additional outlet seeds.
    return_fill_depth : bool, default=False
        If True, also return the per-cell fill depth, defined as the
        filled elevation minus the original elevation.

    Returns
    -------
    dem_filled : np.ndarray
        Depression-filled DEM with NoData preserved.
    fill_depth : np.ndarray, optional
        Per-cell fill depth. Returned only if `return_fill_depth=True`.

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

    def in_bounds(r: int, c: int) -> bool:
        """Return True if the cell lies inside the raster extent."""
        return 0 <= r < n_rows and 0 <= c < n_cols

    # ---------------------------------------------------------------------
    # Step 0: Prepare masks and initialize the priority queue
    # ---------------------------------------------------------------------
    if np.isnan(nodata):
        valid_mask = np.isfinite(dem_values)
    else:
        valid_mask = np.isfinite(dem_values) & (dem_values != nodata)

    dem_filled = dem_values.copy()
    visited_mask = np.zeros_like(valid_mask, dtype=bool)
    priority_queue: List[Tuple[float, int, int]] = []

    # ---------------------------------------------------------------------
    # Step 1: Seed valid raster-edge cells
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Step 2: Optionally seed cells adjacent to internal NoData regions
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Step 3: Propagate inward and fill depressions
    # ---------------------------------------------------------------------
    while priority_queue:
        water_level, r, c = heapq.heappop(priority_queue)

        for dr, dc in NEIGHBOR_OFFSETS_8:
            rr, cc = r + dr, c + dc
            if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]) or visited_mask[rr, cc]:
                continue

            visited_mask[rr, cc] = True

            # Raise the neighbour to the current spill elevation if needed.
            if dem_filled[rr, cc] < water_level:
                dem_filled[rr, cc] = water_level

            heapq.heappush(priority_queue, (dem_filled[rr, cc], rr, cc))

    # ---------------------------------------------------------------------
    # Step 4: Restore NoData and optionally compute fill depth
    # ---------------------------------------------------------------------
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
