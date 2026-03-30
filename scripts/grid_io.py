from __future__ import annotations

"""
Utilities for exporting grid-aligned rasters from Earth Engine.

This module prepares a shared raster grid for the workflow and exports
Earth Engine images to local GeoTIFF files aligned to that grid.
Direct HTTP export is attempted first. If the request exceeds the
download-size limit, the export automatically falls back to tiled
download, local tile normalization, and raster assembly.

Public functions
----------------
export_dem_grid
    Build a grid-locked DEM and pixel-area raster and load them as
    aligned NumPy arrays.

ee_to_tif
    Export a single Earth Engine image to a GeoTIFF aligned to a
    predefined grid, with automatic fallback to tiled export when needed.
"""

import contextlib
import io
import logging
import math
import os
import re
import shutil
import tempfile
from typing import Any, Dict, Optional, Union

import ee
import geemap
import numpy as np
import rasterio
from rasterio.windows import Window


# Conservative defaults for direct HTTP export fallback planning.
_TILE_HARD_LIMIT_BYTES = 32 * 1024 * 1024
_TILE_SAFETY_FACTOR = 0.7
_TILE_MAX_DIM = 10000
_TILE_MAX_RETRIES = 4
_TILE_REFINE_MODE = "double"


class EarthEngineExportError(RuntimeError):
    """Raised when an Earth Engine export fails."""


class EarthEngineSizeLimitError(EarthEngineExportError):
    """Raised when a direct Earth Engine download exceeds the size limit."""


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
    Export a DEM and a matching pixel-area raster from Earth Engine to aligned NumPy arrays.
    
    This function implements the transition from Earth Engine to the local
    (NumPy-based) workflow. It converts a DEM source to a single image,
    derives a stable reference projection, optionally aligns the export
    region to the DEM grid, constructs grid-locked DEM and pixel-area
    images, exports both rasters to GeoTIFF, and loads them into aligned
    NumPy arrays.

    The pixel-area raster is exported on the same grid as the DEM and is
    used to convert flow accumulation to physical units (m² or km²), which
    is required because pixel area varies in geographic coordinate systems
    such as WGS84.
    
    Export uses `ee_to_tif()`, which first attempts a direct download and
    automatically switches to tiled export when the request exceeds the
    HTTP response-size limit.
    
    The procedure consists of the following steps:
    
    Step 0
        Normalize the DEM source to a single Earth Engine image and derive
        a reference projection.
    
    Step 1
        Optionally align the export region to the DEM grid.
    
    Step 2
        Build a reusable export-grid definition.
    
    Step 3
        Apply the requested DEM resampling method.
    
    Step 4
        Reproject DEM and pixel-area rasters to a shared grid.
    
    Step 5
        Export both rasters to GeoTIFF.
    
    Step 6
        Load exported GeoTIFF rasters into NumPy arrays and verify alignment.
    
    Step 7
        Normalize NoData values and assemble output structures.
    
    Parameters
    ----------
    src : ee.Image or ee.ImageCollection or str
        DEM source. A string is interpreted as an Earth Engine image
        collection identifier. If a collection is provided, it is filtered
        to the region and mosaicked.
    region_geom : ee.Geometry
        Region to export.
    band : str, optional
        Band name to select from the source image or collection.
    resample_method : {"nearest", "bilinear", "bicubic"}, default="bilinear"
        Resampling method applied to the DEM before reprojection.
    nodata_value : float, default=-9999.0
        Value used to materialize DEM NoData during export.
    snap_region_to_grid : bool, default=True
        If True, align the export region to the DEM grid.
    tmp_dir : str, optional
        Directory for GeoTIFF outputs. If None, a temporary directory
        is created.
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
        Dictionary with aligned DEM and pixel-area arrays, raster
        metadata, Earth Engine grid images, output paths, and export
        diagnostics. The main keys are:
            - "dem_elevations_np"
            - "pixel_area_m2_np"
            - "transform"
            - "crs"
            - "nodata_mask"
            - "ee_dem_grid"
            - "ee_px_area_grid"
            - "paths"
            - "export_info"
    """
    # Preserve original logger levels so they can be restored afterward.
    previous_levels: dict[str, int] = {}
    if quiet:
        # Temporarily suppress verbose logging from Google and geemap
        # during export.
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)
    
    try:
        # -----------------------------------------------------------------
        # Step 0: Normalize DEM source and derive reference projection
        # -----------------------------------------------------------------
        # Convert the input source to a single DEM image and use its
        # projection as the stable grid reference for export.
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
        # Step 1: Optionally align the region to the DEM grid
        # -----------------------------------------------------------------
        # Snap the export region to the DEM grid so that exported rasters
        # match the source grid boundaries exactly.
        if snap_region_to_grid:
            mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
            aligned_geometry = mask.geometry().transform(proj=proj, maxError=1)
            region_aligned = aligned_geometry.bounds(maxError=1, proj=proj)
        else:
            region_aligned = region_geom
    
        # -----------------------------------------------------------------
        # Step 2: Build the reusable export-grid definition
        # -----------------------------------------------------------------
        # Extract the projection definition needed to reproduce the DEM grid
        # consistently during both DEM and pixel-area export.
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
        # Step 3: Apply DEM resampling method
        # -----------------------------------------------------------------
        # Resample the DEM before reprojection so that export uses the
        # requested interpolation method.
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
        # Step 4: Reproject DEM and pixel-area rasters to the shared grid
        # -----------------------------------------------------------------
        # Construct grid-locked DEM and pixel-area images so that both
        # exported rasters are perfectly aligned.
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

        # Create a temporary local directory for exported rasters
        # if none was supplied.
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
    
        dem_path = os.path.join(tmp_dir, dem_filename)
        pixel_area_path = os.path.join(tmp_dir, px_filename)
    
        # -----------------------------------------------------------------
        # Step 5: Export DEM and pixel-area rasters to GeoTIFF
        # -----------------------------------------------------------------
        # Export the DEM with explicit NoData materialization and the
        # pixel-area raster on the same grid so both outputs remain aligned.
        dem_export_info = ee_to_tif(
            img=ee_dem_grid.unmask(nodata_value),
            out_path=dem_path,
            grid=export_grid,
            quiet=quiet,
            tile_prefix="dem_tile",
            tmp_dir=os.path.join(tmp_dir, "dem_tiles"),
        )
        px_export_info = ee_to_tif(
            img=ee_px_area_grid,
            out_path=pixel_area_path,
            grid=export_grid,
            quiet=quiet,
            tile_prefix="px_tile",
            tmp_dir=os.path.join(tmp_dir, "px_tiles"),
        )
    
        # -----------------------------------------------------------------
        # Step 6: Load exported GeoTIFF rasters into NumPy arrays and verify alignment
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
        # Step 7: Normalize NoData and assemble outputs
        # -----------------------------------------------------------------
        # Convert exported DEM NoData values to NaN and keep the original
        # boolean mask for subsequent local processing.
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
            "projection_info": {
                "crs": crs,
                "transform": crs_transform,
            },
            "scale_m": scale_m,
            "region_used": region_aligned,
            "ee_dem_grid": ee_dem_grid,
            "ee_px_area_grid": ee_px_area_grid,
            "paths": {
                "dem_elevations": dem_path,
                "pixel_area": pixel_area_path,
            },
            "export_info": {
                "dem_elevations": dem_export_info,
                "pixel_area": px_export_info,
            },
            "tmp_dir": tmp_dir,
        }
    
    finally:
        if quiet:
            # Restore original logger levels after export completes.
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)


def ee_to_tif(
    img: ee.Image,
    *,
    out_path: str,
    grid: dict[str, Any],
    unmask_value: float | None = None,
    quiet: bool = True,
    force_tiled: bool = False,
    force_standard: bool = False,
    tmp_dir: Optional[str] = None,
    tile_prefix: str = "tile",
    remove_tile_dirs: bool = False,
) -> Dict[str, Any]:
    """
    Export an Earth Engine image to a grid-aligned GeoTIFF with automatic fallback.
    
    This function provides a robust export mechanism from Earth Engine to a
    local GeoTIFF. It first attempts a standard direct export. If the request
    exceeds the Earth Engine response-size limit, it automatically switches
    to tiled export, reconstructs the full raster grid, and assembles the
    final GeoTIFF locally.
    
    The function is designed to ensure reliable export of large rasters by
    combining direct export, adaptive tiling, and retry logic.
    
    The procedure consists of the following steps:
    
    Step 0
        Validate export mode and input options.
    
    Step 1
        Attempt a standard direct export.
    
    Step 2
        If the direct export exceeds the size limit, reconstruct the full
        raster grid (transform and dimensions) from the grid definition.
    
    Step 3
        Estimate an initial tile grid based on raster size and a conservative
        byte budget.
    
    Step 4
        Export the image by tiles and assemble the final GeoTIFF.
    
    Step 5
        If tiled export fails, refine the tile grid and retry until the
        maximum number of attempts is reached.
    
    Step 6
        Return export diagnostics or raise an error if all attempts fail.
    
    Parameters
    ----------
    img : ee.Image
        Earth Engine image to export.
    out_path : str
        Target GeoTIFF path.
    grid : dict
        Grid definition with projection metadata and export region.
    unmask_value : float, optional
        Value used in `img.unmask(unmask_value)` before export.
    quiet : bool, default=True
        If True, suppress verbose logging during export.
    force_tiled : bool, default=False
        If True, skip direct export and use tiled export immediately.
        Intended mainly for testing.
    force_standard : bool, default=False
        If True, disable tiled fallback and raise the original direct
        export error instead. Intended mainly for testing.
    tmp_dir : str, optional
        Directory for temporary tile outputs when tiled export is used.
    tile_prefix : str, default="tile"
        Prefix used for temporary tile filenames.
    remove_tile_dirs : bool, default=False
        If True, remove temporary tiled-export directories after a
        successful tiled export.
    
    Returns
    -------
    result : dict
        Dictionary describing the export mode, output path, tile-grid
        configuration, and export diagnostics.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate export mode and input options
    # ---------------------------------------------------------------------
    if force_tiled and force_standard:
        raise ValueError("force_tiled and force_standard cannot both be True.")

    standard_error: Exception | None = None

    # ---------------------------------------------------------------------
    # Step 1: Attempt a standard direct export
    # ---------------------------------------------------------------------
    # Attempt a standard direct Earth Engine export (single request).
    if not force_tiled:
        try:
            _ee_to_tif_standard(
                img=img,
                out_path=out_path,
                grid=grid,
                unmask_value=unmask_value,
                quiet=quiet,
            )
            return {
                "mode": "standard",
                "out_path": out_path,
                "n_cols": 1,
                "n_rows": 1,
                "attempt": 0,
                "standard_error": None,
                "attempt_history": [],
            }
        except Exception as exc:
            standard_error = exc
            if force_standard:
                raise
            if not isinstance(exc, EarthEngineSizeLimitError):
                raise

    # ---------------------------------------------------------------------
    # Step 2: Derive full raster transform and shape from the grid
    # ---------------------------------------------------------------------
    # Reconstruct the full raster grid (transform and dimensions)
    # required for tiled export.
    full_transform, full_width, full_height = _grid_transform_and_shape(grid)

    # ---------------------------------------------------------------------
    # Step 3: Estimate an initial tile grid
    # ---------------------------------------------------------------------
    # Estimate an initial tile grid that satisfies memory constraints.
    plan = _suggest_tile_grid(
        width=full_width,
        height=full_height,
        band_count=1,
        bytes_per_pixel=4,
        hard_limit_bytes=_TILE_HARD_LIMIT_BYTES,
        safety_factor=_TILE_SAFETY_FACTOR,
        max_tile_dim=_TILE_MAX_DIM,
    )

    n_cols = plan["n_cols"]
    n_rows = plan["n_rows"]

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="ee_auto_export_")
    else:
        os.makedirs(tmp_dir, exist_ok=True)

    attempt_history: list[dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Step 4-5: Export tiles, assemble the raster, and refine if needed
    # ---------------------------------------------------------------------
    # Export the image in tiles, assemble the final GeoTIFF, and
    # iteratively refine the tile grid if export fails.
    for attempt in range(1, _TILE_MAX_RETRIES + 1):
        attempt_dir = os.path.join(tmp_dir, f"attempt_{attempt}_{n_rows}x{n_cols}")
        os.makedirs(attempt_dir, exist_ok=True)

        try:
            tiled_res = _export_ee_tiled(
                img=img,
                out_path=out_path,
                grid=grid,
                full_transform=full_transform,
                full_width=full_width,
                full_height=full_height,
                n_cols=n_cols,
                n_rows=n_rows,
                tmp_dir=attempt_dir,
                tile_prefix=tile_prefix,
                unmask_value=unmask_value,
                quiet=quiet,
            )

            result = {
                "mode": "tiled",
                "out_path": out_path,
                "n_cols": n_cols,
                "n_rows": n_rows,
                "attempt": attempt,
                "tmp_dir": attempt_dir,
                "tile_paths": tiled_res["tile_paths"],
                "standard_error": repr(standard_error) if standard_error is not None else None,
                "attempt_history": attempt_history,
                "plan": {
                    "tile_width": math.ceil(full_width / n_cols),
                    "tile_height": math.ceil(full_height / n_rows),
                    "tile_bytes_est": _estimate_raster_bytes(
                        math.ceil(full_width / n_cols),
                        math.ceil(full_height / n_rows),
                    ),
                    "total_bytes_est": _estimate_raster_bytes(full_width, full_height),
                },
            }

            if remove_tile_dirs:
                for name in os.listdir(tmp_dir):
                    path = os.path.join(tmp_dir, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)

            return result

        except Exception as exc:
            attempt_history.append(
                {
                    "attempt": attempt,
                    "n_cols": n_cols,
                    "n_rows": n_rows,
                    "error": repr(exc),
                }
            )
            n_cols, n_rows = _refine_tile_grid(n_cols, n_rows, mode=_TILE_REFINE_MODE)

    # ---------------------------------------------------------------------
    # Step 6: Raise a final error if all export attempts failed
    # ---------------------------------------------------------------------
    error_lines = ["Automatic EE export failed.", ""]

    if standard_error is not None:
        error_lines.extend(["Standard export error:", repr(standard_error), ""])
    else:
        error_lines.extend(["Standard export was skipped.", ""])

    error_lines.append("Tiled export attempts:")
    for item in attempt_history:
        error_lines.append(
            f"  attempt={item['attempt']}, grid={item['n_rows']}x{item['n_cols']}, "
            f"error={item['error']}"
        )

    raise RuntimeError("\n".join(error_lines))


def _ee_to_tif_standard(
    img: ee.Image,
    *,
    out_path: str,
    grid: dict[str, Any],
    unmask_value: float | None = None,
    quiet: bool = True,
) -> str:
    """
    Export an Earth Engine image to GeoTIFF using a single direct request.

    This internal function performs a standard Earth Engine export in a
    single HTTP request using a predefined grid definition. It prepares
    export arguments from the grid metadata, optionally applies unmasking,
    executes the export, and interprets captured log output to detect
    response-size failures and other export errors.

    The procedure consists of the following steps:

    Step 0
        Validate the grid definition and derive export parameters.

    Step 1
        Prepare the output directory and optional quiet logging mode.

    Step 2
        Build the export image and execute the direct export request.

    Step 3
        Detect successful output creation or interpret captured log output.

    Step 4
        Raise a specialized export error when the request exceeds the
        Earth Engine response-size limit or fails for another reason.

    Step 5
        Restore original logger levels.

    Parameters
    ----------
    img : ee.Image
        Earth Engine image to export.
    out_path : str
        Output GeoTIFF path.
    grid : dict
        Grid definition containing projection metadata and export region.
    unmask_value : float, optional
        Value used in `img.unmask(unmask_value)` before export.
    quiet : bool, default=True
        If True, suppress verbose logging and capture export output.

    Returns
    -------
    str
        Path to the created GeoTIFF.

    Raises
    ------
    KeyError
        If the required grid definition is incomplete.
    EarthEngineSizeLimitError
        If the direct export fails due to the Earth Engine response-size limit.
    EarthEngineExportError
        If the export fails for another reason.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate grid definition and derive export parameters
    # ---------------------------------------------------------------------
    for key in ("projection_info", "region_used"):
        if key not in grid:
            raise KeyError(f"grid is missing required key: '{key}'")
    
    proj_info = grid["projection_info"]
    crs_str = proj_info["crs"]
    crs_transform = proj_info.get("transform", None)
    region = grid["region_used"]
    
    export_kwargs: dict[str, Any] = {
        "region": region,
        "file_per_band": False,
    }
    
    # Use either the full affine transform or a scale-based export
    # depending on how the grid definition is represented.
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
    # Step 1: Prepare output path and optional quiet logging
    # ---------------------------------------------------------------------
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Preserve logger levels so they can be restored after export.
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)
    
    # ---------------------------------------------------------------------
    # Step 2: Build export image and execute direct export
    # ---------------------------------------------------------------------
    try:
        # Materialize masked pixels only when an explicit unmask value is requested.
        export_img = img.unmask(unmask_value) if unmask_value is not None else img

        # Remove any stale file so successful export is detected unambiguously.
        if os.path.exists(out_path):
            os.remove(out_path)
    
        log_text = ""
        if quiet:
            # Capture geemap / EE output so it can be inspected if export fails.
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)
            log_text = sink.getvalue()
        else:
            geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)

        # ---------------------------------------------------------------------
        # Step 3: Detect successful output creation or inspect export logs
        # ---------------------------------------------------------------------
        if os.path.exists(out_path):
            return out_path
    
        reported_total = None
        reported_limit = None

        # Try to extract the reported request size and limit from EE/geemap logs.
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
                r"Total request size\s*must be less than or equal to\s*(\d+)\s*bytes",
                log_text,
            )
            if match:
                reported_limit = int(match.group(1))

        # ---------------------------------------------------------------------
        # Step 4: Raise a specialized export error
        # ---------------------------------------------------------------------
        # Construct an informative error message based on the captured
        # Earth Engine / geemap output.
        msg = f"Earth Engine export failed: '{out_path}' was not created.\n"

        if reported_total is not None and reported_limit is not None:
            msg += (
                f"Reported by Earth Engine: total request size = {reported_total} bytes, "
                f"limit = {reported_limit} bytes.\n"
                f"Reduce the request by at least {reported_total - reported_limit} bytes.\n"
            )
            raise EarthEngineSizeLimitError(msg)

        if reported_limit is not None:
            msg += (
                "Reported by Earth Engine: total request size must be less than "
                f"or equal to {reported_limit} bytes.\n"
            )
            raise EarthEngineSizeLimitError(msg)

        if log_text.strip():
            msg += "Captured Earth Engine or geemap log output:\n" + log_text + "\n"

        raise EarthEngineExportError(msg)

    finally:
        # -----------------------------------------------------------------
        # Step 5: Restore original logger levels
        # -----------------------------------------------------------------
        if quiet:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)


def _estimate_raster_bytes(
    width: int,
    height: int,
    band_count: int = 1,
    bytes_per_pixel: int = 4,
) -> int:
    """Estimate raster size in bytes."""
    return int(width) * int(height) * int(band_count) * int(bytes_per_pixel)


def _compute_safe_tile_budget(
    hard_limit_bytes: int = _TILE_HARD_LIMIT_BYTES,
    safety_factor: float = _TILE_SAFETY_FACTOR,
) -> int:
    """Compute a conservative per-tile byte budget."""
    return int(hard_limit_bytes * safety_factor)


def _suggest_tile_grid(
    width: int,
    height: int,
    *,
    band_count: int = 1,
    bytes_per_pixel: int = 4,
    hard_limit_bytes: int = _TILE_HARD_LIMIT_BYTES,
    safety_factor: float = _TILE_SAFETY_FACTOR,
    max_tile_dim: int = _TILE_MAX_DIM,
) -> Dict[str, int]:
    """
    Suggest a near-square tile grid that satisfies byte and dimension limits.
    """
    max_tile_bytes = _compute_safe_tile_budget(hard_limit_bytes, safety_factor)

    total_bytes = _estimate_raster_bytes(
        width,
        height,
        band_count=band_count,
        bytes_per_pixel=bytes_per_pixel,
    )

    n_tiles = max(1, math.ceil(total_bytes / max_tile_bytes))
    n_cols = math.ceil(math.sqrt(n_tiles))
    n_rows = math.ceil(n_tiles / n_cols)

    while math.ceil(width / n_cols) > max_tile_dim:
        n_cols += 1
    while math.ceil(height / n_rows) > max_tile_dim:
        n_rows += 1

    tile_width = math.ceil(width / n_cols)
    tile_height = math.ceil(height / n_rows)

    return {
        "n_cols": n_cols,
        "n_rows": n_rows,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_bytes_est": _estimate_raster_bytes(
            tile_width,
            tile_height,
            band_count=band_count,
            bytes_per_pixel=bytes_per_pixel,
        ),
        "total_bytes_est": total_bytes,
        "max_tile_bytes": max_tile_bytes,
    }


def _refine_tile_grid(n_cols: int, n_rows: int, mode: str = _TILE_REFINE_MODE) -> tuple[int, int]:
    """Refine a tile grid for another retry attempt."""
    if mode == "double":
        return n_cols * 2, n_rows * 2
    if mode == "increment":
        return n_cols + 1, n_rows + 1
    raise ValueError("Unsupported refine mode. Use 'double' or 'increment'.")


def _grid_transform_and_shape(grid: dict[str, Any]) -> tuple[rasterio.Affine, int, int]:
    """
    Derive the cropped raster transform and shape from a grid definition.

    Tiled fallback requires an explicit projection transform. Grid
    definitions based only on scale are not supported here.
    """
    proj_info = grid["projection_info"]
    crs_str = proj_info["crs"]
    crs_transform = proj_info.get("transform", None)
    region = grid["region_used"]

    if crs_transform is None:
        raise ValueError(
            "Automatic tiled export requires projection_info['transform']. "
            "Grid definitions based only on scale are not supported."
        )

    base_transform = rasterio.Affine(*crs_transform)

    region_proj = region.bounds(maxError=1, proj=ee.Projection(crs_str))
    coords = region_proj.coordinates().getInfo()[0]

    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    col_min_f, row_top_f = ~base_transform * (xmin, ymax)
    col_max_f, row_bottom_f = ~base_transform * (xmax, ymin)

    tol = 1e-9

    col_min = int(math.floor(col_min_f + tol))
    col_max = int(math.ceil(col_max_f - tol))
    row_top = int(math.floor(row_top_f + tol))
    row_bottom = int(math.ceil(row_bottom_f - tol))

    width = col_max - col_min
    height = row_bottom - row_top

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid grid dimensions derived from region: width={width}, height={height}."
        )

    cropped_transform = base_transform * rasterio.Affine.translation(col_min, row_top)
    return cropped_transform, width, height


def _split_windows(width: int, height: int, n_cols: int, n_rows: int) -> list[dict[str, Any]]:
    """Split a raster shape into a regular grid of raster windows."""
    col_edges = np.linspace(0, width, n_cols + 1, dtype=int)
    row_edges = np.linspace(0, height, n_rows + 1, dtype=int)

    windows: list[dict[str, Any]] = []
    idx = 0

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            col0 = int(col_edges[col_idx])
            col1 = int(col_edges[col_idx + 1])
            row0 = int(row_edges[row_idx])
            row1 = int(row_edges[row_idx + 1])

            windows.append(
                {
                    "index": idx,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "window": Window(
                        col_off=col0,
                        row_off=row0,
                        width=col1 - col0,
                        height=row1 - row0,
                    ),
                }
            )
            idx += 1

    return windows


def _window_to_region(window: Window, transform: rasterio.Affine) -> ee.Geometry:
    """
    Convert a raster window to an EE rectangle.

    The rectangle is slightly shrunken inward to reduce the chance that
    Earth Engine returns an extra boundary pixel. Exact final placement
    is enforced later by cropping each downloaded tile to the expected
    raster window.
    """
    left, bottom, right, top = rasterio.windows.bounds(window, transform)

    px_w = abs(transform.a)
    px_h = abs(transform.e)

    eps_x = px_w * 1e-6
    eps_y = px_h * 1e-6

    return ee.Geometry.Rectangle(
        [left + eps_x, bottom + eps_y, right - eps_x, top - eps_y],
        proj=None,
        geodesic=False,
    )


def _export_ee_tiled(
    img: ee.Image,
    *,
    out_path: str,
    grid: dict[str, Any],
    full_transform: rasterio.Affine,
    full_width: int,
    full_height: int,
    n_cols: int,
    n_rows: int,
    tmp_dir: str,
    tile_prefix: str,
    unmask_value: float | None = None,
    quiet: bool = True,
) -> Dict[str, Any]:
    """
    Export an image by tiles, normalize each tile to the expected raster
    window, and assemble the final GeoTIFF locally.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    windows = _split_windows(full_width, full_height, n_cols=n_cols, n_rows=n_rows)

    assembled: np.ndarray | None = None
    profile: dict[str, Any] | None = None
    covered = np.zeros((full_height, full_width), dtype=bool)
    tile_paths: list[str] = []

    for tile in windows:
        idx = tile["index"]
        window = tile["window"]

        exp_h = int(window.height)
        exp_w = int(window.width)

        tile_grid = {
            "projection_info": grid["projection_info"],
            "region_used": _window_to_region(window, full_transform),
            "scale_m": grid.get("scale_m", None),
        }

        tile_path = os.path.join(tmp_dir, f"{tile_prefix}_{idx}.tif")

        _ee_to_tif_standard(
            img=img,
            out_path=tile_path,
            grid=tile_grid,
            unmask_value=unmask_value,
            quiet=quiet,
        )

        with rasterio.open(tile_path) as src:
            arr = src.read(1)
            if profile is None:
                profile = src.profile.copy()

        # Earth Engine may return one extra boundary row or column.
        # Crop to the expected raster window before assembly.
        arr = arr[:exp_h, :exp_w]

        if arr.shape != (exp_h, exp_w):
            raise ValueError(
                f"Tile {idx} could not be normalized to expected shape "
                f"{(exp_h, exp_w)}. Actual cropped shape: {arr.shape}."
            )

        if assembled is None:
            assembled = np.empty((full_height, full_width), dtype=arr.dtype)

        row0 = int(window.row_off)
        row1 = row0 + exp_h
        col0 = int(window.col_off)
        col1 = col0 + exp_w

        assembled[row0:row1, col0:col1] = arr
        covered[row0:row1, col0:col1] = True
        tile_paths.append(tile_path)

    if assembled is None or profile is None:
        raise RuntimeError("Tiled export failed before any tile was assembled.")
    if not np.all(covered):
        raise RuntimeError("Tiled export did not cover the full output raster.")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    profile.update(
        {
            "height": full_height,
            "width": full_width,
            "transform": full_transform,
            "count": 1,
        }
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(assembled, 1)

    return {
        "out_path": out_path,
        "tile_paths": tile_paths,
        "n_cols": n_cols,
        "n_rows": n_rows,
    }
