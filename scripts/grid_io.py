# grid_io.py
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional, Union

import ee
import geemap
import numpy as np
import rasterio

from scripts.ee_export import export_ee_image_to_geotiff


def export_dem_and_area_to_arrays(
    src: Union[ee.image.Image, ee.imagecollection.ImageCollection, str],
    region_geom: ee.Geometry,
    *,
    band: Optional[str] = None,                  # e.g. 'DEM' for Copernicus; None for single-band
    resample_method: str = "bilinear",           # 'nearest' | 'bilinear' | 'bicubic'
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,
    tmp_dir: Optional[str] = None,
    dem_filename: str = "dem_elevations.tif",
    px_filename: str = "pixel_area.tif",
    quiet: bool = True,                          # suppress prints/logs from geemap / google clients
) -> Dict[str, Any]:
    """
    Export DEM and pixel area from Earth Engine to aligned GeoTIFFs and NumPy arrays.

    The procedure:
    - normalizes the source to a single ee.Image with a stable projection,
    - optionally snaps the export region to the DEM pixel grid,
    - exports DEM and pixel area on the same grid and projection,
    - reads both GeoTIFFs back as NumPy arrays and normalizes NoData to NaN.

    Returns
    -------
    dict with keys:
      - dem_elevations_np: (H, W) float64 array, NaN = NoData
      - pixel_area_m2_np:  (H, W) float64 array
      - transform: rasterio Affine
      - crs: rasterio CRS
      - nodata_mask: (H, W) bool
      - nodata_value_raw: float
      - projection_info: {'crs': str, 'transform': list|None}
      - scale_m: float | None
      - region_used: ee.Geometry (aligned region)
      - ee_dem_grid: ee.Image (masked, grid-locked)
      - ee_px_area_grid: ee.Image (grid-locked)
      - paths: {'dem_elevations': path, 'pixel_area': path}
      - tmp_dir: temp folder actually used
    """
    # Global verbosity control for Google/geemap clients (optional)
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    try:
        # 0) Normalize input to ee.Image and determine a stable projection
        if isinstance(src, ee.image.Image):
            dem_image = src if band is None else src.select([band])
            reference_image = dem_image
        else:
            dem_collection = ee.ImageCollection(src) if isinstance(src, str) else src
            if band is not None:
                dem_collection = dem_collection.select([band])
            reference_image = dem_collection.first()
            dem_image = dem_collection.filterBounds(region_geom).mosaic()

        proj = ee.Image(reference_image).projection()
        dem_image = ee.Image(dem_image).setDefaultProjection(proj)

        # 1) Optionally align the export region to the DEM grid
        if snap_region_to_grid:
            mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
            aligned_geometry_or_bbox = mask.geometry().transform(proj=proj, maxError=1)
            region_aligned = aligned_geometry_or_bbox.bounds(maxError=1, proj=proj)
        else:
            region_aligned = region_geom

        # 2) Extract CRS and affine transform (or fall back to nominal scale)
        proj_info = proj.getInfo()
        crs = proj_info["crs"]
        crs_transform = proj_info.get("transform", None)

        scale_m: Optional[float] = None
        if crs_transform is None:
            scale_m = float(ee.Image(dem_image).projection().nominalScale().getInfo())

        export_grid: dict[str, Any] = {
            "projection_info": {"crs": crs, "transform": crs_transform},
            "region_used": region_aligned,
            "scale_m": scale_m,
        }

        # 3) Apply resampling policy
        resample_method_norm = (resample_method or "").lower()
        if resample_method_norm in ("bilinear", "bicubic"):
            dem_resampled = ee.Image(dem_image).resample(resample_method_norm)
        elif resample_method_norm in ("nearest", "", None):
            dem_resampled = ee.Image(dem_image)
        else:
            raise ValueError(
                f"Invalid resample_method: {resample_method}. "
                "Use 'nearest', 'bilinear', or 'bicubic'."
            )

        # 4) Grid-lock DEM and pixel area to identical grid/projection
        ee_dem_grid = (
            ee.Image(dem_resampled)
            .toFloat()
            .reproject(crs=crs, crsTransform=crs_transform)
            .clip(region_aligned)
            .updateMask(ee.Image(dem_image).mask())
        )

        ee_px_area_grid = (
            ee.Image.pixelArea()
            .toFloat()
            .reproject(crs=crs, crsTransform=crs_transform)
            .updateMask(ee.Image(dem_image).mask())
            .clip(region_aligned)
        )

        # 5) Prepare output paths
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        dem_path = os.path.join(tmp_dir, dem_filename)
        pixel_area_path = os.path.join(tmp_dir, px_filename)

        # 6) Earth Engine -> GeoTIFF export (DEM uses explicit NoData materialization)
        export_ee_image_to_geotiff(
            ee_dem_grid.unmask(nodata_value),
            out_path=dem_path,
            grid=export_grid,
            quiet=quiet,
        )
        export_ee_image_to_geotiff(
            ee_px_area_grid,
            out_path=pixel_area_path,
            grid=export_grid,
            quiet=quiet,
        )

        # 7) Read GeoTIFFs back to NumPy and validate alignment
        with rasterio.open(dem_path) as src_dem:
            dem_raw_data = src_dem.read(1).astype("float64")
            transform = src_dem.transform
            out_crs = src_dem.crs
            nd_src = src_dem.nodata

        with rasterio.open(pixel_area_path) as src_px:
            pixel_area_data = src_px.read(1).astype("float64")
            if (
                src_px.transform != transform
                or src_px.crs != out_crs
                or src_px.width != dem_raw_data.shape[1]
                or src_px.height != dem_raw_data.shape[0]
            ):
                raise ValueError("pixel_area is not aligned with DEM (transform/CRS/shape mismatch).")

        # 8) Normalize NoData to NaN and build a NoData mask
        nodata_value_raw = nd_src if nd_src is not None else float(nodata_value)
        nodata_mask = (dem_raw_data == nodata_value_raw) | ~np.isfinite(dem_raw_data)

        dem_elevations = dem_raw_data.copy()
        dem_elevations[nodata_mask] = np.nan

        return {
            "dem_elevations_np": dem_elevations,
            "pixel_area_m2_np": pixel_area_data,
            "transform": transform,
            "crs": out_crs,
            "nodata_mask": nodata_mask,
            "nodata_value_raw": nodata_value_raw,
            "projection_info": {"crs": crs, "transform": crs_transform},
            "scale_m": scale_m,
            "region_used": region_aligned,
            "ee_dem_grid": ee_dem_grid,
            "ee_px_area_grid": ee_px_area_grid,
            "paths": {"dem_elevations": dem_path, "pixel_area": pixel_area_path},
            "tmp_dir": tmp_dir,
        }

    finally:
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)
