from __future__ import annotations

"""
Raster I/O utilities for local workflow processing.

This script provides functions for reading, writing, and clipping
GeoTIFF rasters in the local (NumPy-based) part of the workflow.
It ensures consistent handling of georeferencing, data types, and
NoData representation across all operations.

The functions provided are:
    - save_tif: writes a NumPy array to a GeoTIFF with consistent
      metadata, tiling, compression, and NoData handling
    - clip_tif: clips a GeoTIFF to a given geometry and preserves
      consistent output structure and metadata
    - read_tif: reads a GeoTIFF into a NumPy array with NoData
      normalized to NaN

NoData values are consistently represented as NaN in memory and,
when required, encoded using either explicit values or raster masks
in the output files.
"""

import os
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from rasterio.warp import transform_geom


def _ensure_dir(path: str) -> None:
    """Ensure that the parent directory of a file path exists."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _as_bool_mask(mask_arr: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """
    Return a boolean NoData mask and validate its shape.

    If mask_arr is None, return an all-False mask.
    """
    if mask_arr is None:
        return np.zeros(shape, dtype=bool)

    mask = np.asarray(mask_arr, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"nodata_mask must have shape {shape}, got {mask.shape}")

    return mask


def _default_profile(
    *,
    height: int,
    width: int,
    dtype: str,
    crs: Any,
    transform: Affine,
    compress: str = "LZW",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
    nodata_value: float = np.nan,
) -> dict:
    """
    Build a rasterio GeoTIFF profile for a single-band raster.

    If nodata_value is NaN, NoData is represented using a raster mask.
    """
    return {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "tiled": tiled,
        "blockxsize": int(blockxsize),
        "blockysize": int(blockysize),
        # When NoData is represented by NaN, store nodata as None and rely on the raster mask.
        "nodata": None if np.isnan(nodata_value) else float(nodata_value),
    }


def save_tif(
    arr: np.ndarray,
    transform: Affine,
    crs: Any,
    nodata_mask: Optional[np.ndarray] = None,
    filename: str = "output.tif",
    dtype: Optional[str] = None,
    compress: str = "LZW",
    nodata_value: float = np.nan,
    band_name: Optional[str] = None,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> str:
    """
    Save a two-dimensional NumPy array to a single-band GeoTIFF.

    This function writes a raster array together with its georeference,
    coordinate reference system, and NoData definition. If `nodata_value`
    is NaN, NoData is represented using an explicit raster mask.

    The procedure consists of the following steps:

    Step 0
        Validate the input array and prepare the output directory.

    Step 1
        Derive the effective NoData mask.

    Step 2
        Prepare the output array and assign NoData values where needed.

    Step 3
        Build the GeoTIFF profile.

    Step 4
        Write the raster and optional band metadata.

    Step 5
        Write an explicit raster mask when NoData is represented by NaN.

    Parameters
    ----------
    arr : np.ndarray
        Two-dimensional input array.
    transform : affine.Affine
        Affine transform describing raster georeferencing.
    crs : Any
        Coordinate reference system.
    nodata_mask : np.ndarray, optional
        Boolean mask indicating NoData cells (True = NoData). If None,
        non-finite values in `arr` are treated as NoData where possible.
    filename : str, default="output.tif"
        Output GeoTIFF path.
    dtype : str, optional
        Output data type. If None, the dtype of `arr` is preserved.
    compress : str, default="LZW"
        Compression method used for the GeoTIFF.
    nodata_value : float, default=np.nan
        NoData value written to the output raster. If NaN, NoData is
        represented through the raster mask.
    band_name : str, optional
        Optional band description.
    blockxsize : int, default=256
        Internal tile width.
    blockysize : int, default=256
        Internal tile height.

    Returns
    -------
    str
        Path to the created GeoTIFF.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate input and prepare output directory
    # ---------------------------------------------------------------------
    array = np.asarray(arr)
    if array.ndim != 2:
        raise ValueError("save_tif expects a 2D array (single band).")

    _ensure_dir(filename)

    out_dtype = dtype or str(array.dtype)

    # ---------------------------------------------------------------------
    # Step 1: Derive the effective NoData mask
    # ---------------------------------------------------------------------
    if nodata_mask is None:
        try:
            nodata = ~np.isfinite(array)
        except Exception:
            # If finiteness cannot be evaluated for the input dtype, fall
            # back to an all-valid mask and rely on an explicit NoData mask.
            nodata = np.zeros_like(array, dtype=bool)
    else:
        nodata = _as_bool_mask(nodata_mask, array.shape)

    # ---------------------------------------------------------------------
    # Step 2: Prepare the output array and assign NoData representation
    # ---------------------------------------------------------------------
    # Create a writable copy in the target dtype and encode NoData either
    # as an explicit value or, for floating-point outputs, preserve NaN.
    write_arr = array.astype(out_dtype, copy=True)

    if np.isfinite(nodata_value):
        write_arr[nodata] = nodata_value
    else:
        if np.issubdtype(write_arr.dtype, np.floating):
            write_arr[nodata] = np.nan

    # ---------------------------------------------------------------------
    # Step 3: Build the GeoTIFF profile
    # ---------------------------------------------------------------------
    # Assemble the GeoTIFF metadata, including raster dimensions,
    # georeferencing, tiling, compression, and NoData definition.
    profile = _default_profile(
        height=array.shape[0],
        width=array.shape[1],
        dtype=str(write_arr.dtype),
        crs=crs,
        transform=transform,
        compress=compress,
        tiled=True,
        blockxsize=blockxsize,
        blockysize=blockysize,
        nodata_value=nodata_value,
    )

    # ---------------------------------------------------------------------
    # Step 4: Write the raster and optional band metadata
    # ---------------------------------------------------------------------
    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(write_arr, 1)

        if band_name:
            dst.set_band_description(1, band_name)

        # -----------------------------------------------------------------
        # Step 5: Write an explicit raster mask for NaN-based NoData
        # -----------------------------------------------------------------
        if np.isnan(nodata_value):
            mask_bytes = (~nodata).astype("uint8") * 255
            dst.write_mask(mask_bytes)

    return filename


def clip_tif(
    input_tif: str,
    geojson_geom: dict,
    output_tif: str,
    band_name: Optional[str] = None,
    *,
    src_crs_assumed: str = "EPSG:4326",
    all_touched: bool = True,
    dtype: str = "float32",
    compress: str = "LZW",
    nodata_value: float = np.nan,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> str:
    """
    Clip a GeoTIFF by a GeoJSON geometry and write the result to a new raster.

    The input geometry is reprojected to the raster CRS, the raster is
    cropped and masked to the area of interest, and the output is written
    as a single-band GeoTIFF with consistent compression, tiling, and
    NoData handling.

    The procedure consists of the following steps:

    Step 0
        Prepare the output directory.

    Step 1
        Open the input raster and reproject the input geometry to the
        raster coordinate reference system.

    Step 2
        Mask and crop the raster to the area of interest.

    Step 3
        Normalize source NoData values in the clipped raster to a consistent
        internal representation.

    Step 4
        Build the output GeoTIFF profile.

    Step 5
        Write the clipped raster and optional raster mask.

    Parameters
    ----------
    input_tif : str
        Path to the input GeoTIFF.
    geojson_geom : dict
        Input GeoJSON geometry.
    output_tif : str
        Path to the output GeoTIFF.
    band_name : str, optional
        Optional band description.
    src_crs_assumed : str, default="EPSG:4326"
        CRS assumed for the input GeoJSON geometry.
    all_touched : bool, default=True
        Passed to `rasterio.mask.mask`.
    dtype : str, default="float32"
        Output data type.
    compress : str, default="LZW"
        Compression method used for the output GeoTIFF.
    nodata_value : float, default=np.nan
        NoData value written to the output raster. If NaN, NoData is
        represented through the raster mask.
    blockxsize : int, default=256
        Internal tile width.
    blockysize : int, default=256
        Internal tile height.

    Returns
    -------
    str
        Path to the created clipped GeoTIFF.
    """
    # ---------------------------------------------------------------------
    # Step 0: Prepare the output directory
    # ---------------------------------------------------------------------
    _ensure_dir(output_tif)

    with rasterio.open(input_tif) as src:
        src_meta = src.meta.copy()

        # Use the raster CRS as the target CRS for the clipping geometry.
        dst_crs_str = src.crs.to_string()

        # -----------------------------------------------------------------
        # Step 1: Reproject the input geometry to raster CRS
        # -----------------------------------------------------------------
        geom_in_raster_crs = transform_geom(
            src_crs=src_crs_assumed,
            dst_crs=dst_crs_str,
            geom=geojson_geom,
            precision=6,
        )

        # -----------------------------------------------------------------
        # Step 2: Mask and crop the raster to the AOI
        # -----------------------------------------------------------------
        out_image, out_transform = rio_mask(
            src,
            [geom_in_raster_crs],
            crop=True,
            all_touched=all_touched,
            nodata=nodata_value,
            filled=True,
        )

        out_image = out_image.astype(dtype, copy=False)

        # -----------------------------------------------------------------
        # Step 3: Normalize source NoData values
        # -----------------------------------------------------------------
        # Convert source NoData values to NaN so that invalid cells are handled
        # consistently before writing the output raster.
        in_nodata = src_meta.get("nodata")
        if in_nodata is not None and np.isfinite(in_nodata):
            out_image[out_image == in_nodata] = np.nan

        # -----------------------------------------------------------------
        # Step 4: Build the output profile
        # -----------------------------------------------------------------
        # Assemble the output GeoTIFF metadata using the clipped raster shape,
        # updated transform, and the requested storage settings.
        out_meta = src_meta.copy()
        out_meta.update(
            _default_profile(
                height=out_image.shape[1],
                width=out_image.shape[2],
                dtype=str(out_image.dtype),
                crs=src.crs,
                transform=out_transform,
                compress=compress,
                tiled=True,
                blockxsize=blockxsize,
                blockysize=blockysize,
                nodata_value=nodata_value,
            )
        )

        # -----------------------------------------------------------------
        # Step 5: Write the clipped raster and optional raster mask
        # -----------------------------------------------------------------
        # Write the clipped raster and, when NoData is represented by NaN,
        # store validity explicitly through the raster mask.
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(out_image)

            if band_name:
                dst.set_band_description(1, band_name)

            if np.isnan(nodata_value):
                valid = np.isfinite(out_image[0])
                dst.write_mask(valid.astype("uint8") * 255)

    return output_tif

def read_tif(
    tif_path: str,
    *,
    nodata_mask: np.ndarray | None = None,
    as_float32: bool = True,
) -> np.ndarray:
    """
    Read a single-band GeoTIFF to a NumPy array and normalize NoData to NaN.

    The raster is read from disk, optionally promoted to a floating type,
    source NoData values are converted to NaN, and an optional external
    NoData mask can be applied afterward.

    The procedure consists of the following steps:

    Step 0
        Read the raster and extract metadata.

    Step 1
        Ensure a floating-point representation for NaN support.

    Step 2
        Normalize source NoData values.

    Step 3
        Enforce a consistent NaN-based representation.

    Step 4
        Apply an optional external NoData mask.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF.
    nodata_mask : np.ndarray, optional
        Boolean mask indicating additional NoData cells (True = NoData).
        This is typically used to enforce consistency with an external
        DEM NoData mask.
    as_float32 : bool, default=True
        If True, return a float32 array. Otherwise preserve the raster
        dtype where possible, except where floating conversion is needed
        for NaN support.

    Returns
    -------
    np.ndarray
        Two-dimensional array with NoData represented as NaN.
    """
    # ---------------------------------------------------------------------
    # Step 0: Read raster and metadata
    # ---------------------------------------------------------------------
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata_val = src.nodata

    # ---------------------------------------------------------------------
    # Step 1: Ensure floating-point representation
    # ---------------------------------------------------------------------
    # Convert to float to allow NaN representation of NoData values.
    if as_float32:
        arr = arr.astype(np.float32)
    elif not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)

    # ---------------------------------------------------------------------
    # Step 2: Normalize source NoData values
    # ---------------------------------------------------------------------
    # Replace raster-defined NoData values with NaN.
    if nodata_val is not None:
        arr = np.where(np.isclose(arr, nodata_val), np.nan, arr)

    # ---------------------------------------------------------------------
    # Step 3: Enforce NaN-based representation
    # ---------------------------------------------------------------------
    # Ensure all invalid or non-finite values are consistently represented as NaN.
    arr = np.where(np.isfinite(arr), arr, np.nan)

    # ---------------------------------------------------------------------
    # Step 4: Apply external NoData mask
    # ---------------------------------------------------------------------
    # Combine with an optional mask propagated from earlier workflow steps.
    if nodata_mask is not None:
        arr = np.where(np.asarray(nodata_mask, dtype=bool), np.nan, arr)

    return arr
    
