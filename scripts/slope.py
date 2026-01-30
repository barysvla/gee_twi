from __future__ import annotations
from typing import Any

import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os
import time
import shutil
import io
import contextlib
import logging

def compute_slope(dem):
    """
    Computation of terrain slope based on DEM.
    """
    slope = ee.Terrain.slope(dem).rename("Slope")
    
    return slope


def slope_ee_to_numpy(
    grid: dict[str, Any],
    ee_dem_grid: ee.Image,
    *,
    quiet: bool = True,
) -> np.ndarray:
    """
    Export EE slope computed from ee_dem_grid to a NumPy array on the SAME grid as DEM.

    The function:
      - computes slope in EE from ee_dem_grid,
      - exports it using the exact CRS/transform (or scale) from `grid`,
      - reads GeoTIFF back as float32,
      - converts NoData to NaN and applies grid["nodata_mask"].

    Returns
    -------
    slope_np : np.ndarray
        (H, W) float32 array, NaN = NoData.
    """
    # --- Validate required grid keys ---
    for key in ("projection_info", "region_used", "nodata_mask"):
        if key not in grid:
            raise KeyError(f"grid is missing required key: '{key}'")

    proj_info = grid["projection_info"]  # {'crs': str, 'transform': list|None}
    crs_str = proj_info["crs"]
    crs_transform_list = proj_info.get("transform", None)
    region_aligned = grid["region_used"]

    # --- Compute slope in EE (degrees) on the aligned grid ---
    slope_img = ee.Terrain.slope(ee_dem_grid).toFloat().rename("slope")

    export_kwargs = {
        "region": region_aligned,
        "file_per_band": False,
        "crs": crs_str,
    }
    if crs_transform_list is not None:
        export_kwargs["crs_transform"] = crs_transform_list
    else:
        if "scale_m" not in grid or grid["scale_m"] is None:
            raise KeyError("grid must contain 'scale_m' when projection_info.transform is None")
        export_kwargs["scale"] = float(grid["scale_m"])

    # --- Silence noisy logs if requested ---
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            lg = logging.getLogger(name)
            previous_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

    def _ee_export(img: ee.Image, filename: str, **kwargs) -> None:
        # Export a single-band EE image to GeoTIFF using geemap
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(img, filename=filename, **kwargs)
        else:
            geemap.ee_export_image(img, filename=filename, **kwargs)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slope_tif = os.path.join(tmp_dir, "slope.tif")

            # Export (geemap is expected to block until the file exists)
            _ee_export(slope_img, filename=slope_tif, **export_kwargs)

            if not os.path.exists(slope_tif):
                raise RuntimeError(f"EE export failed, file not created: {slope_tif}")

            # Read back -> NumPy
            with rasterio.open(slope_tif) as src:
                slope_np = src.read(1).astype(np.float32)
                nodata_val = src.nodata

    finally:
        # Restore logger levels
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)

    # Convert nodata to NaN if nodata is defined
    if nodata_val is not None:
        slope_np = np.where(np.isclose(slope_np, nodata_val), np.nan, slope_np)

    # Enforce DEM NoData mask
    slope_np = np.where(np.asarray(grid["nodata_mask"], dtype=bool), np.nan, slope_np)

    # Shape check against DEM array stored in grid
    dem_ref = grid.get("dem_elevations", None)
    if dem_ref is not None and slope_np.shape != dem_ref.shape:
        raise ValueError(f"Grid mismatch: slope {slope_np.shape} vs DEM {dem_ref.shape}")

    return slope_np
