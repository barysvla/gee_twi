from __future__ import annotations

from typing import Any
import contextlib
import io
import logging
import os
import re

import ee
import geemap
import numpy as np
import rasterio


def export_ee_image_to_geotiff(
    img: ee.Image,
    *,
    out_path: str,
    grid: dict[str, Any],
    unmask_value: float | None = None,
    quiet: bool = True,
) -> str:
    """
    Export a single-band Earth Engine image to a GeoTIFF aligned to a predefined grid.

    The export is controlled by the grid definition to guarantee identical spatial
    alignment (CRS, transform/scale, region extent) across multiple exported layers.

    Parameters
    ----------
    img : ee.Image
        Input image to export (single band recommended).
    out_path : str
        Target GeoTIFF path.
    grid : dict
        Grid definition. Required keys:
          - "projection_info": {"crs": str, "transform": list | None}
          - "region_used": ee.Geometry
        Optional keys:
          - "scale_m": float (required if projection_info["transform"] is None)
    unmask_value : float or None
        If provided, `img.unmask(unmask_value)` is exported. This is typically used
        when NoData must be materialized as a numeric value in the GeoTIFF.
    quiet : bool
        If True, suppresses geemap/Google client logging during export.

    Returns
    -------
    str
        Path to the created GeoTIFF.
    """
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
            raise KeyError("grid must contain 'scale_m' when projection_info.transform is None")
        export_kwargs["scale"] = float(scale_m)

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Optional log suppression
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            lg = logging.getLogger(name)
            previous_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

    try:
        export_img = img.unmask(unmask_value) if unmask_value is not None else img

        log_text = ""
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)
            log_text = sink.getvalue()
        else:
            geemap.ee_export_image(export_img, filename=out_path, **export_kwargs)

        if not os.path.exists(out_path):
            # Parse common EE HTTP response size error if present
            reported_total = None
            reported_limit = None

            m = re.search(
                r"Total request size\s*\((\d+)\s*bytes\)\s*must be less than or equal to\s*(\d+)\s*bytes",
                log_text,
            )
            if m:
                reported_total = int(m.group(1))
                reported_limit = int(m.group(2))
            else:
                m2 = re.search(
                    r"Total request size\s*must be less than or equal to\s*(\d+)\s*bytes",
                    log_text,
                )
                if m2:
                    reported_limit = int(m2.group(1))

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
                    "Reported by Earth Engine: total request size must be less than or equal to "
                    f"{reported_limit} bytes.\n"
                )
            elif log_text.strip():
                msg += "Captured Earth Engine / geemap log output:\n" + log_text + "\n"

            raise RuntimeError(msg)

        return out_path

    finally:
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)


def read_geotiff_to_numpy(
    tif_path: str,
    *,
    nodata_mask: np.ndarray | None = None,
    as_float32: bool = True,
) -> np.ndarray:
    """
    Read a single-band GeoTIFF into a NumPy array and normalize NoData to NaN.

    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF.
    nodata_mask : np.ndarray or None
        Optional boolean mask (True = NoData) applied after reading.
        This is typically the DEM NoData mask used to enforce consistency.
    as_float32 : bool
        If True, returns float32; otherwise preserves rasterio dtype (except NaN normalization,
        which requires float).

    Returns
    -------
    np.ndarray
        (H, W) array with NoData represented as NaN.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata_val = src.nodata

    # Promote to float for NaN support
    if as_float32:
        arr = arr.astype(np.float32)
    else:
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)

    # Normalize source nodata to NaN
    if nodata_val is not None:
        arr = np.where(np.isclose(arr, nodata_val), np.nan, arr)

    # Normalize non-finite values to NaN
    arr = np.where(np.isfinite(arr), arr, np.nan)

    # Apply external nodata mask (e.g., DEM mask)
    if nodata_mask is not None:
        arr = np.where(np.asarray(nodata_mask, dtype=bool), np.nan, arr)

    return arr
