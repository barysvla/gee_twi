# scripts/raster_io.py
from __future__ import annotations

import os
from typing import Optional, Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom


def _ensure_dir(path: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _as_bool_mask(mask_arr: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Validate/construct a boolean NoData mask (True = NoData)."""
    if mask_arr is None:
        return np.zeros(shape, dtype=bool)
    m = np.asarray(mask_arr, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"nodata_mask must have shape {shape}, got {m.shape}")
    return m


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
    """Build a consistent GeoTIFF profile for outputs."""
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
        # For NaN we store nodata as None and rely on raster mask
        "nodata": (None if np.isnan(nodata_value) else float(nodata_value)),
    }


def save_array_as_geotiff(
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
    Save a 2D NumPy array to GeoTIFF, preserving georeference/CRS and NoData mask.

    Conventions:
    - Output is single-band GeoTIFF.
    - If nodata_value is NaN (default), NoData is represented via raster mask.
    - If nodata_mask is None, non-finite values in 'arr' are treated as NoData.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("save_array_as_geotiff expects a 2D array (single band).")

    _ensure_dir(filename)

    out_dtype = dtype or str(a.dtype)

    # Determine NoData mask
    if nodata_mask is None:
        try:
            nodata = ~np.isfinite(a)
        except Exception:
            nodata = np.zeros_like(a, dtype=bool)
    else:
        nodata = _as_bool_mask(nodata_mask, a.shape)

    # Prepare data for write
    write_arr = a.astype(out_dtype, copy=True)

    if np.isfinite(nodata_value):
        write_arr[nodata] = nodata_value
    else:
        # If nodata_value is NaN, keep NaNs as-is; enforce NaN in nodata cells if needed
        if np.issubdtype(write_arr.dtype, np.floating):
            write_arr[nodata] = np.nan

    profile = _default_profile(
        height=a.shape[0],
        width=a.shape[1],
        dtype=str(write_arr.dtype),
        crs=crs,
        transform=transform,
        compress=compress,
        tiled=True,
        blockxsize=blockxsize,
        blockysize=blockysize,
        nodata_value=nodata_value,
    )

    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(write_arr, 1)
        if band_name:
            dst.set_band_description(1, band_name)

        # If nodata_value is NaN, write an explicit mask (255=valid, 0=nodata)
        if np.isnan(nodata_value):
            mask_bytes = (~nodata).astype("uint8") * 255
            dst.write_mask(mask_bytes)

    return filename


def clip_tif_by_geojson(
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
    Clip a GeoTIFF by a GeoJSON geometry. Pixels outside AOI -> NoData (NaN by default).

    Steps:
    - Reproject input geometry (assumed WGS84 by default) into raster CRS.
    - Mask + crop the raster to AOI.
    - Write output as GeoTIFF with consistent compression/tiling and band description.
    """
    _ensure_dir(output_tif)

    with rasterio.open(input_tif) as src:
        src_meta = src.meta.copy()
        dst_crs_str = src.crs.to_string()

        # Reproject GeoJSON geometry into raster CRS
        geom_in_raster_crs = transform_geom(
            src_crs=src_crs_assumed,
            dst_crs=dst_crs_str,
            geom=geojson_geom,
            precision=6,
        )

        # Mask raster to AOI
        out_image, out_transform = rio_mask(
            src,
            [geom_in_raster_crs],
            crop=True,
            all_touched=all_touched,
            nodata=nodata_value,
            filled=True,
        )

        # out_image shape is (bands, H, W); here we assume single band inputs/outputs
        out_image = out_image.astype(dtype, copy=False)

        # If input had a finite nodata value, normalize it to NaN in the output (robustness)
        in_nodata = src_meta.get("nodata")
        if in_nodata is not None and np.isfinite(in_nodata):
            out_image[out_image == in_nodata] = np.nan

        # Build output profile with unified settings
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

        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(out_image)
            if band_name:
                dst.set_band_description(1, band_name)

            # If nodata_value is NaN, write mask based on finite values
            if np.isnan(nodata_value):
                valid = np.isfinite(out_image[0])
                dst.write_mask(valid.astype("uint8") * 255)

    return output_tif
