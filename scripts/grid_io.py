from __future__ import annotations

"""
Utilities for exporting grid-aligned raster data from Earth Engine.

This script provides helper functions for transferring raster layers from
Earth Engine to local GeoTIFF files and NumPy arrays while preserving a
shared spatial grid. It is used to prepare aligned DEM and pixel-area inputs
for the workflow and to export additional Earth Engine layers to the same
grid during later processing steps.

The main functions are `export_dem_grid`, which builds and exports the input
DEM grid, and `ee_to_tif`, which exports individual Earth Engine images using
a predefined grid definition.
"""

import contextlib
import io
import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional, Union

import ee
import geemap
import numpy as np
import rasterio


def export_dem_grid(
    src: Union[ee.image.Image, ee.imagecollection.ImageCollection, str],
    region_geom: ee.Geometry,
    *,
    band: Optional[str] = None,
    resample_method: str = "bilinear",
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,
    tmp_dir: Optional[str] = None,
    dem_filename: str = "dem_elevations.tif",
    px_filename: str = "pixel_area.tif",
    quiet: bool = True,
) -> Dict[str, Any]:
    """
    Export DEM elevation and pixel-area rasters from Earth Engine to aligned arrays.

    This function exports a DEM layer and the corresponding pixel-area
    layer from Earth Engine using a shared spatial grid. The workflow
    normalizes the input source to a single image, derives a stable
    projection, optionally aligns the export region to the raster grid,
    exports both layers as GeoTIFFs, and reads them back as NumPy arrays.

    The procedure consists of the following steps:

    Step 0
        Optionally suppress verbose logging from geemap and Google
        client libraries.

    Step 1
        Normalize the input source to a single Earth Engine image and
        derive a stable reference projection.

    Step 2
        Optionally align the export region to the DEM grid.

    Step 3
        Extract projection metadata and construct the export-grid
        definition.

    Step 4
        Apply the requested resampling method to the DEM.

    Step 5
        Reproject and clip the DEM and pixel-area layers to the same
        grid.

    Step 6
        Prepare output paths and export both layers to GeoTIFF.

    Step 7
        Read the exported rasters back to NumPy arrays and verify that
        both rasters are spatially aligned.

    Step 8
        Normalize DEM NoData values to NaN and construct the output
        dictionary.

    Parameters
    ----------
    src : ee.Image or ee.ImageCollection or str
        DEM source. A string is interpreted as an Earth Engine image
        collection identifier.
    region_geom : ee.Geometry
        Export region.
    band : str, optional
        Band name to select from the source image or collection.
    resample_method : {"nearest", "bilinear", "bicubic"}, default="bilinear"
        Resampling method applied to the DEM before export.
    nodata_value : float, default=-9999.0
        Numeric value used to materialize DEM NoData during export.
    snap_region_to_grid : bool, default=True
        If True, align the export region to the DEM grid.
    tmp_dir : str, optional
        Directory used for temporary GeoTIFF outputs. If None, a new
        temporary directory is created.
    dem_filename : str, default="dem_elevations.tif"
        Output filename for the DEM raster.
    px_filename : str, default="pixel_area.tif"
        Output filename for the pixel-area raster.
    quiet : bool, default=True
        If True, suppress verbose logging from geemap and Google client
        libraries during export.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
            - "dem_elevations_np": DEM array of shape (H, W), float64,
              with NaN as NoData
            - "pixel_area_m2_np": pixel-area array of shape (H, W), float64
            - "transform": rasterio Affine transform
            - "crs": raster CRS
            - "nodata_mask": boolean NoData mask of shape (H, W)
            - "nodata_value_raw": raw DEM NoData value read from GeoTIFF
            - "projection_info": {"crs": str, "transform": list | None}
            - "scale_m": export scale in metres or None
            - "region_used": export region actually used
            - "ee_dem_grid": grid-locked Earth Engine DEM image
            - "ee_px_area_grid": grid-locked Earth Engine pixel-area image
            - "paths": output paths for both GeoTIFF files
            - "tmp_dir": temporary directory actually used
    """
    # ---------------------------------------------------------------------
    # Step 0: Optionally suppress external library logging
    # ---------------------------------------------------------------------
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    try:
        # -----------------------------------------------------------------
        # Step 1: Normalize the source and derive a stable projection
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Step 2: Optionally align the export region to the DEM grid
        # -----------------------------------------------------------------
        if snap_region_to_grid:
            mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
            aligned_geometry = mask.geometry().transform(proj=proj, maxError=1)
            region_aligned = aligned_geometry.bounds(maxError=1, proj=proj)
        else:
            region_aligned = region_geom

        # -----------------------------------------------------------------
        # Step 3: Extract projection metadata and define the export grid
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Step 4: Apply the DEM resampling method
        # -----------------------------------------------------------------
        resample_method_norm = (resample_method or "").lower()

        if resample_method_norm in ("bilinear", "bicubic"):
            dem_resampled = ee.Image(dem_image).resample(resample_method_norm)
        elif resample_method_norm in ("nearest", ""):
            dem_resampled = ee.Image(dem_image)
        else:
            raise ValueError(
                f"Invalid resample_method: {resample_method}. "
                "Use 'nearest', 'bilinear', or 'bicubic'."
            )

        # -----------------------------------------------------------------
        # Step 5: Reproject and clip DEM and pixel-area rasters
        # -----------------------------------------------------------------
        if crs_transform is not None:
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
        else:
            ee_dem_grid = (
                ee.Image(dem_resampled)
                .toFloat()
                .reproject(crs=crs, scale=scale_m)
                .clip(region_aligned)
                .updateMask(ee.Image(dem_image).mask())
            )

            ee_px_area_grid = (
                ee.Image.pixelArea()
                .toFloat()
                .reproject(crs=crs, scale=scale_m)
                .updateMask(ee.Image(dem_image).mask())
                .clip(region_aligned)
            )

        # -----------------------------------------------------------------
        # Step 6: Prepare output paths and export GeoTIFF files
        # -----------------------------------------------------------------
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()

        dem_path = os.path.join(tmp_dir, dem_filename)
        pixel_area_path = os.path.join(tmp_dir, px_filename)

        ee_to_tif(
            ee_dem_grid.unmask(nodata_value),
            out_path=dem_path,
            grid=export_grid,
            quiet=quiet,
        )
        ee_to_tif(
            ee_px_area_grid,
            out_path=pixel_area_path,
            grid=export_grid,
            quiet=quiet,
        )

        # -----------------------------------------------------------------
        # Step 7: Read GeoTIFFs and verify raster alignment
        # -----------------------------------------------------------------
        with rasterio.open(dem_path) as src_dem:
            dem_raw_data = src_dem.read(1).astype(np.float64)
            transform = src_dem.transform
            out_crs = src_dem.crs
            nd_src = src_dem.nodata

        with rasterio.open(pixel_area_path) as src_px:
            pixel_area_data = src_px.read(1).astype(np.float64)

            if (
                src_px.transform != transform
                or src_px.crs != out_crs
                or src_px.width != dem_raw_data.shape[1]
                or src_px.height != dem_raw_data.shape[0]
            ):
                raise ValueError(
                    "pixel_area is not aligned with DEM "
                    "(transform, CRS, or shape mismatch)."
                )

        # -----------------------------------------------------------------
        # Step 8: Normalize DEM NoData values and build the output
        # -----------------------------------------------------------------
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
            "paths": {
                "dem_elevations": dem_path,
                "pixel_area": pixel_area_path,
            },
            "tmp_dir": tmp_dir,
        }

    finally:
        if quiet:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)


def ee_to_tif(
    img: ee.Image,
    *,
    out_path: str,
    grid: dict[str, Any],
    unmask_value: float | None = None,
    quiet: bool = True,
) -> str:
    """
    Export an Earth Engine image to a GeoTIFF aligned to a predefined grid.

    This function exports a single Earth Engine image using an explicit
    grid definition so that multiple raster layers can be written with
    identical CRS, transform or scale, and region extent.

    The procedure consists of the following steps:

    Step 0
        Validate the grid definition and derive export parameters.

    Step 1
        Ensure that the output directory exists.

    Step 2
        Optionally suppress verbose logging from geemap and Google
        client libraries.

    Step 3
        Prepare the export image, optionally materializing NoData by
        unmasking with a numeric value.

    Step 4
        Export the image to GeoTIFF and capture export logs when
        requested.

    Step 5
        Verify that the output file was created and raise a descriptive
        error if the export failed.

    Parameters
    ----------
    img : ee.Image
        Earth Engine image to export.
    out_path : str
        Target GeoTIFF path.
    grid : dict
        Grid definition. Required keys:
            - "projection_info": {"crs": str, "transform": list | None}
            - "region_used": ee.Geometry
        Optional keys:
            - "scale_m": float, required if transform is not available
    unmask_value : float, optional
        Value used in `img.unmask(unmask_value)` before export.
    quiet : bool, default=True
        If True, suppress verbose logging from geemap and Google client
        libraries during export.

    Returns
    -------
    out_path : str
        Path to the created GeoTIFF.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate the grid definition and derive export parameters
    # ---------------------------------------------------------------------
    for key in ("projection_info", "region_used"):
        if key not in grid:
            raise KeyError(f"grid is missing required key: '{key}'")

    proj_info = grid["projection_info"]
    crs_str = proj_info["crs"]
    crs_transform = proj_info.get("transform", None)
    region_aligned = grid["region_used"]

    export_kwargs: dict[str, Any] = {
        "region": region_aligned,
        "file_per_band": False,
    }

    if crs_transform is not None:
        export_kwargs["crs"] = crs_str
        export_kwargs["crs_transform"] = crs_transform
    else:
        scale_m = grid.get("scale_m", None)
        if scale_m is None:
            raise KeyError(
                "grid must contain 'scale_m' when projection_info.transform is None."
            )

        export_kwargs["crs"] = crs_str
        export_kwargs["scale"] = float(scale_m)

    # ---------------------------------------------------------------------
    # Step 1: Ensure that the output directory exists
    # ---------------------------------------------------------------------
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step 2: Optionally suppress external library logging
    # ---------------------------------------------------------------------
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    try:
        # -----------------------------------------------------------------
        # Step 3: Prepare the export image
        # -----------------------------------------------------------------
        export_img = img.unmask(unmask_value) if unmask_value is not None else img

        # -----------------------------------------------------------------
        # Step 4: Export the image and optionally capture logs
        # -----------------------------------------------------------------
        log_text = ""
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)
            log_text = sink.getvalue()
        else:
            geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)

        # -----------------------------------------------------------------
        # Step 5: Verify that the output file was created
        # -----------------------------------------------------------------
        if not os.path.exists(out_path):
            reported_total = None
            reported_limit = None

            match = re.search(
                (
                    r"Total request size\s*\((\d+)\s*bytes\)\s*"
                    r"must be less than or equal to\s*(\d+)\s*bytes"
                ),
                log_text,
            )
            if match:
                reported_total = int(match.group(1))
                reported_limit = int(match.group(2))
            else:
                match = re.search(
                    (
                        r"Total request size\s*must be less than or equal to\s*"
                        r"(\d+)\s*bytes"
                    ),
                    log_text,
                )
                if match:
                    reported_limit = int(match.group(1))

            msg = (
                f"Earth Engine export failed: '{out_path}' was not created.\n"
                "A common cause is exceeding the HTTP response size limit.\n"
            )

            if reported_total is not None and reported_limit is not None:
                msg += (
                    f"Reported by Earth Engine: total request size = {reported_total} bytes, "
                    f"limit = {reported_limit} bytes.\n"
                    f"Reduce the request by at least {reported_total - reported_limit} bytes "
                    "(smaller region and/or coarser resolution).\n"
                )
            elif reported_limit is not None:
                msg += (
                    "Reported by Earth Engine: total request size must be less than "
                    f"or equal to {reported_limit} bytes.\n"
                )
            elif log_text.strip():
                msg += "Captured Earth Engine or geemap log output:\n" + log_text + "\n"

            raise RuntimeError(msg)

        return out_path

    finally:
        if quiet:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)
