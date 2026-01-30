import os
import numpy as np
import rasterio
from rasterio.transform import Affine

def save_array_as_geotiff(
    arr: np.ndarray,
    transform: Affine,
    crs,
    nodata_mask: np.ndarray = None,
    filename: str = "output.tif",
    dtype: str = None,
    compress: str = "LZW",
    nodata_value: float = np.nan,
    band_name: str = None,
) -> str:
    """
    Save a numpy array to GeoTIFF, preserving georeference, CRS, and nodata mask.

    Parameters:
      - arr: 2D numpy array (rows × cols)
      - transform: Affine transform for georeference
      - crs: coordinate reference system (e.g. "EPSG:xxxx" or rasterio CRS)
      - nodata_mask: boolean mask (True where nodata), optional
      - filename: path/name of output file
      - dtype: output data type (e.g. "float32"); if None, arr.dtype is used
      - compress: compression algorithm (e.g. "LZW")
      - nodata_value: value to use for nodata (e.g. np.nan or a numeric value)

    Returns:
      - the filename (path) of the written GeoTIFF
    """

    # Create output directory if not exists
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Determine the output dtype
    if dtype is None:
        out_dtype = arr.dtype
    else:
        out_dtype = dtype

    # Determine nodata mask
    if nodata_mask is None:
        try:
            # For float arrays, mask where values are not finite
            nodata_mask = ~np.isfinite(arr)
        except Exception:
            nodata_mask = np.zeros_like(arr, dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Prepare array to write: cast and set nodata
    write_arr = arr.astype(out_dtype).copy()
    if np.isfinite(nodata_value):
        write_arr[nodata_mask] = nodata_value
    else:
        # If nodata_value is NaN, we leave as is for NaN positions
        pass

    # Prepare profile/dataset metadata for rasterio
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": write_arr.dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "nodata": (None if np.isnan(nodata_value) else float(nodata_value)),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Write the GeoTIFF
    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(write_arr, 1)
        if band_name:
            dst.set_band_description(1, band_name)
        # If nodata_value is NaN, write mask based on nodata_mask
        if np.isnan(nodata_value) and nodata_mask is not None:
            mask = (~nodata_mask).astype("uint8") * 255
            dst.write_mask(mask)

    return filename
