from __future__ import annotations

import io
import os
import ee
import geemap
import numpy as np
import rasterio
import tempfile
import logging
import contextlib
from typing import Any, Dict, Optional, Union


def export_dem_and_area_to_arrays(
    src: Union[ee.image.Image, ee.imagecollection.ImageCollection, str],
    region_geom: ee.Geometry,
    *,
    band: Optional[str] = None,  # e.g. 'DEM' for Copernicus; None for single-band
    resample_method: str = "bilinear",         # 'nearest' | 'bilinear' | 'bicubic'
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,
    tmp_dir: Optional[str] = None,
    dem_filename: str = "dem_elevations.tif",
    px_filename: str = "pixel_area.tif",
    quiet: bool = True,                        # suppress prints/logs from geemap / google clients
) -> Dict[str, Any]:
    """
    Export DEM and pixel area from Earth Engine to aligned GeoTIFFs and NumPy arrays.
    
    The function:
    - normalizes the source to a single ee.Image with a stable projection,
    - optionally snaps the region geometry to the DEM grid,
    - exports DEM and pixel area on the same grid and projection,
    - reads them back as NumPy arrays and harmonizes NoData to NaN.
    
    Returns:
        dict with:
            dem_elevations: (H, W) float64 ndarray, NaN = NoData
            pixel_area_m2: (H, W) float64 ndarray
            transform: rasterio Affine
            crs: rasterio CRS
            nodata_mask: (H, W) bool
            nodata_value_raw: float
            projection_info: {'crs': str, 'transform': list|None}
            scale_m: float | None
            region_used: ee.Geometry (aligned region)
            ee_dem_grid: ee.Image (masked, grid-locked)
            ee_px_area_grid: ee.Image (grid-locked)
            paths: {'dem_elevations': path, 'pixel_area': path}
            tmp_dir: temp folder actually used
    """
    # --- Verbosity control ----------------------------------------------------
    previous_levels = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    try:
        # --- 0) Normalize input to ee.Image with a stable projection ----------
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

        # --- 1) Optionally snap region to the DEM grid ------------------------
        # This ensures that the export region is aligned to the DEM pixel grid,
        # so all exported rasters (DEM, pixel area, derivatives) share the exact
        # same transform, CRS and shape.
        
        if snap_region_to_grid:
            mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
            aligned_geometry_or_bbox = mask.geometry().transform(proj=proj, maxError=1)
            region_aligned = aligned_geometry_or_bbox.bounds(maxError=1, proj=proj)
        else:
            region_aligned = region_geom

        # --- 2) Export parameters: CRS + transform / scale --------------------
        proj_info = proj.getInfo()
        crs = proj_info["crs"]
        crs_transform = proj_info.get("transform", None)

        scale_m: Optional[float] = None
        if crs_transform is None:
            # No explicit transform from EE → fall back to nominal scale
            scale_m = float(ee.Image(dem_image).projection().nominalScale().getInfo())

        # Helper: EE export via geemap.ee_export_image -------------------------
        def _ee_export(one_img: ee.Image, out_path: str) -> None:
            """
            Export a single-band image to GeoTIFF using geemap.ee_export_image.

            If the export fails and no file is created, it inspects the captured stderr/stdout
            and tries to parse the Earth Engine error message to extract request-size limits.
            Raises RuntimeError with a detailed message if the file was not created.
            """
            export_kwargs = {
                "region": region_aligned,
                "file_per_band": False,
            }

            if crs_transform is not None:
                export_kwargs["crs"] = crs
                export_kwargs["crs_transform"] = crs_transform
            elif scale_m is not None:
                export_kwargs["scale"] = scale_m

            # Capture stdout/stderr from geemap so we can inspect EE error messages
            log_text = ""
            if quiet:
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    geemap.ee_export_image(one_img, filename=out_path, **export_kwargs)
                log_text = sink.getvalue()
            else:
                # When quiet=False we do not redirect, so we only see errors on the console
                # and may not be able to parse them. Still try to run normally.
                geemap.ee_export_image(one_img, filename=out_path, **export_kwargs)

            # If EE export failed, no file will be created
            if not os.path.exists(out_path):
                import re

                reported_total = None
                reported_limit = None

                # Try to parse patterns like:
                # "Total request size (78765200 bytes) must be less than or equal to 50331648 bytes."
                m = re.search(
                    r"Total request size\s*\((\d+)\s*bytes\)\s*must be less than or equal to\s*(\d+)\s*bytes",
                    log_text,
                )
                if m:
                    reported_total = int(m.group(1))
                    reported_limit = int(m.group(2))
                else:
                    # Fallback: "Total request size must be less than or equal to 50331648 bytes."
                    m2 = re.search(
                        r"Total request size\s*must be less than or equal to\s*(\d+)\s*bytes",
                        log_text,
                    )
                    if m2:
                        reported_limit = int(m2.group(1))

                msg = (
                    f"Earth Engine export failed: '{out_path}' was not created.\n"
                    "A common cause for large downloads is the HTTP response size limit.\n"
                )

                if reported_total is not None and reported_limit is not None:
                    msg += (
                        f"Reported by Earth Engine: total request size = {reported_total} bytes, "
                        f"limit = {reported_limit} bytes.\n"
                        f"You need to reduce the request by at least "
                        f"{reported_total - reported_limit} bytes (lower the resolution/area).\n"
                    )
                elif reported_limit is not None:
                    msg += (
                        f"Reported by Earth Engine: total request size must be less than or equal "
                        f"to {reported_limit} bytes.\n"
                    )
                else:
                    # If we could not parse the message, at least attach the captured log
                    if log_text.strip():
                        msg += "Captured Earth Engine / geemap log output:\n" + log_text + "\n"

                msg += (
                    "Try reducing the region extent, using coarser resolution."
                )

                raise RuntimeError(msg)

        # --- 3) Resampling choice ---------------------------------------------
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

        # --- 4) Grid-lock DEM and pixelArea to the same grid ------------------
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

        dem_for_export = ee_dem_grid.unmask(nodata_value)

        # --- 5) Temp files ----------------------------------------------------
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        dem_path = os.path.join(tmp_dir, dem_filename)
        pixel_area_path = os.path.join(tmp_dir, px_filename)

        # --- 6) Earth Engine → GeoTIFF export --------------------------------
        _ee_export(dem_for_export, dem_path)
        _ee_export(ee_px_area_grid, pixel_area_path)

        # --- 7) Read back with rasterio ---------------------------------------
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
                raise ValueError(
                    "pixel_area is not aligned with DEM (transform/CRS/shape mismatch)."
                )

        # --- 8) Harmonize NoData in memory (NaN) ------------------------------
        nodata_value_raw = nd_src if nd_src is not None else float(nodata_value)
        nodata_mask = (dem_raw_data == nodata_value_raw) | ~np.isfinite(dem_raw_data)
        dem_elevations = dem_raw_data.copy()
        dem_elevations[nodata_mask] = np.nan

        return {
            "dem_elevations": dem_elevations,
            "pixel_area_m2": pixel_area_data,
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
