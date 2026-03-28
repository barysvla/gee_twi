from __future__ import annotations

"""
Depression filling as the first step of DEM hydrological conditioning.

The function `fill_depressions` removes closed depressions in a DEM
using the original single-queue Priority-Flood algorithm (Algorithm 1
in Barnes et al., 2014) before subsequent flat resolution.
"""

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
    Fill closed depressions in a raster DEM using the original Priority-Flood algorithm.
    
    This function implements Algorithm 1 (original Priority-Flood) as
    described by Barnes, Lehman, and Mulla (2014), based on a single
    priority queue.
    
    The algorithm starts from raster-edge outlet cells and processes cells
    in priority order from the lowest currently known drainage level
    connected to an outlet. When a newly reached valid neighbour lies
    below the current spill level, its elevation is raised to that level.
    In this way, closed depressions are removed relative to the chosen
    outlet set.
    
    Relative to the reference implementation, this version optionally
    extends the outlet seed set by including valid cells adjacent to
    masked internal NoData regions, allowing such regions to act as
    additional drainage boundaries in finite raster domains.
    
    The resulting DEM is hydrologically conditioned with respect to
    closed depressions, but it may still contain flat areas and is
    therefore intended to be followed by a flat-resolution step before
    flow-direction assignment.
    
    The procedure consists of the following steps:
    
    Step 0
        Prepare the DEM, derive the valid-data mask, and initialize the
        priority queue and visited mask.
    
    Step 1
        Seed all valid raster-edge cells into the priority queue.
    
    Step 2
        Optionally seed additional valid cells adjacent to masked internal
        NoData regions, allowing such regions to act as supplementary
        drainage boundaries.
    
    Step 3
        Repeatedly pop the lowest-priority queued cell, visit its
        unprocessed neighbours, and raise neighbour elevations to the
        current spill level where necessary.
    
    Step 4
        Restore the original NoData mask in the output raster and
        optionally compute per-cell fill depth.
    
    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    nodata : float, default=np.nan
        NoData marker. If set to NaN, all non-finite values are treated
        as invalid.
    seed_internal_nodata_as_outlet : bool, default=True
        If True, valid cells adjacent to masked internal NoData regions
        are treated as additional outlet seeds.
    return_fill_depth : bool, default=False
        If True, also return the per-cell fill depth, defined as the
        filled elevation minus the original elevation.
    
    Returns
    -------
    dem_filled : np.ndarray
        Depression-filled DEM with NoData preserved.
    fill_depth : np.ndarray, optional
        Per-cell fill depth. Returned only if `return_fill_depth=True`.
    
    References
    ----------
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

    # `valid_mask` defines the computational domain of the DEM.
    # `visited_mask` tracks cells that have already been inserted into the
    # flood structure so that each valid cell is processed only once.
    # `dem_filled` is updated in place as depressions are filled.
    dem_filled = dem_values.copy()
    visited_mask = np.zeros_like(valid_mask, dtype=bool)
    priority_queue: List[Tuple[float, int, int]] = []

    # ---------------------------------------------------------------------
    # Step 1: Seed valid raster-edge cells
    # ---------------------------------------------------------------------
    # Seed all valid edge cells as initial outlets.
    # These cells have a guaranteed drainage path outside the DEM domain
    # and therefore define the starting points of the flood.
    for c in range(n_cols):
        # Top edge (row 0)
        if valid_mask[0, c] and not visited_mask[0, c]:
            heapq.heappush(priority_queue, (dem_filled[0, c], 0, c))
            visited_mask[0, c] = True
    
        # Bottom edge (last row)
        if valid_mask[n_rows - 1, c] and not visited_mask[n_rows - 1, c]:
            heapq.heappush(priority_queue, (dem_filled[n_rows - 1, c], n_rows - 1, c))
            visited_mask[n_rows - 1, c] = True
    
    for r in range(1, n_rows - 1):
        # Left edge (column 0, excluding corners)
        if valid_mask[r, 0] and not visited_mask[r, 0]:
            heapq.heappush(priority_queue, (dem_filled[r, 0], r, 0))
            visited_mask[r, 0] = True
    
        # Right edge (last column, excluding corners)
        if valid_mask[r, n_cols - 1] and not visited_mask[r, n_cols - 1]:
            heapq.heappush(priority_queue, (dem_filled[r, n_cols - 1], r, n_cols - 1))
            visited_mask[r, n_cols - 1] = True

    # ---------------------------------------------------------------------
    # Step 2: Optionally seed additional cells adjacent to masked internal voids
    # ---------------------------------------------------------------------
    # In finite or masked raster domains, internal NoData regions may be
    # intended to behave as open drainage boundaries rather than as closed
    # obstacles. Seeding their valid neighbours allows the flood to start
    # from these boundaries as additional outlets.
    if seed_internal_nodata_as_outlet:
        for r in range(n_rows):
            for c in range(n_cols):
                if not valid_mask[r, c] or visited_mask[r, c]:
                    continue

                for dr, dc in NEIGHBOR_OFFSETS_8:
                    rr, cc = r + dr, c + dc
                    if (not in_bounds(rr, cc)) or (not valid_mask[rr, cc]):
                        # As soon as one invalid neighbour is found, the cell is
                        # treated as adjacent to a drainage boundary and seeded once.
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

            # Mark the neighbour as visited at insertion time so that it enters
            # the priority queue only once. This matches the standard flood-style
            # traversal logic used in Priority-Flood.
            visited_mask[rr, cc] = True

            # If the neighbour lies below the lowest known drainage level reaching
            # this cell, raise it to that spill level so it becomes drainable.
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
