from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, Optional

import ee
import numpy as np
import rasterio
from google.api_core.exceptions import BadRequest, Conflict, Forbidden, NotFound
from google.cloud import storage


def _get_or_create_bucket(
    storage_client: storage.Client,
    bucket_name: str,
    *,
    project_id: str,
    location: str = "US",
    storage_class: str = "STANDARD",
) -> storage.Bucket:
    """
    Return an existing GCS bucket or create it if it does not exist.

    The function ensures that the bucket is accessible and located in a
    region compatible with the Earth Engine GeoTIFF loading workflow.
    If the bucket does not exist, it is created with the specified
    location and storage class.

    Only a restricted set of locations is accepted in this workflow:
    'US' or 'US-CENTRAL1'.

    Parameters
    ----------
    storage_client : google.cloud.storage.Client
        Initialized GCS client.
    bucket_name : str
        Bucket name.
    project_id : str
        GCP project ID (used when creating a new bucket).
    location : str, default="US"
        Bucket location for creation.
    storage_class : str, default="STANDARD"
        Storage class for creation.

    Returns
    -------
    google.cloud.storage.Bucket
        Existing or newly created bucket.
    """
    bucket_name = bucket_name.lower().strip()

    try:
        # Try to fetch existing bucket
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()

    except NotFound:
        # Bucket does not exist → create it
        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class

        # Enable uniform bucket-level access (recommended for EE access)
        try:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        except Exception:
            pass

        bucket = storage_client.create_bucket(
            bucket,
            project=project_id,
            location=location,
        )
        bucket.reload()

    except Conflict:
        # Rare race condition: bucket was created between calls
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()

    except Forbidden as exc:
        raise RuntimeError(
            f"Forbidden to access bucket '{bucket_name}'. "
            "Missing permissions: storage.buckets.get and/or storage.buckets.create."
        ) from exc

    except BadRequest as exc:
        raise RuntimeError(
            f"Bad request when accessing or creating bucket '{bucket_name}': {exc}"
        ) from exc

    # Validate bucket location (EE constraint)
    loc = (bucket.location or "").upper().strip()
    if loc not in {"US", "US-CENTRAL1"}:
        raise RuntimeError(
            f"Bucket '{bucket_name}' is in location '{bucket.location}'. "
            "This workflow requires bucket location US or US-CENTRAL1."
        )

    return bucket

def _write_cog_local(
    arr_f32: np.ndarray,
    *,
    transform,
    crs: str,
    out_path: str,
    nodata_value: float,
    blocksize: int = 512,
    compress: str = "LZW",
) -> None:
    """
    Write a single-band Cloud Optimized GeoTIFF to local storage.

    This function writes a single-band raster as a Cloud Optimized
    GeoTIFF (COG) using rasterio.

    Parameters
    ----------
    arr_f32 : np.ndarray
        Two-dimensional float32 array containing raster values.
    transform : affine.Affine
        Affine transform describing raster georeferencing.
    crs : str
        Coordinate reference system identifier.
    out_path : str
        Output file path.
    nodata_value : float
        Numeric NoData value written to the GeoTIFF.
    blocksize : int, default=512
        Internal block size used by the COG writer.
    compress : str, default="LZW"
        Compression method used for the GeoTIFF.

    Returns
    -------
    None
    """
    if arr_f32.dtype != np.float32:
        raise ValueError("_write_cog_local expects a float32 input array.")
    if arr_f32.ndim != 2:
        raise ValueError("_write_cog_local expects a 2D array.")
    if not np.isfinite(nodata_value):
        raise ValueError("nodata_value must be finite.")

    profile = {
        "driver": "COG",
        "height": int(arr_f32.shape[0]),
        "width": int(arr_f32.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float(nodata_value),
        "compress": compress,
        "blocksize": int(blocksize),
        "overview_resampling": "average",
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_f32, 1)


def push_array_to_ee_geotiff(
    array_np: np.ndarray,
    *,
    transform,
    crs: str,
    bucket_name: str,
    project_id: str,
    band_name: str = "acc",
    nodata_mask: Optional[np.ndarray] = None,
    nodata_value: float = -9999.0,
    tmp_dir: Optional[str] = None,
    object_prefix: str = "twi_uploads",
    cleanup_local: bool = False,
) -> Dict[str, Any]:
    """
    Write a NumPy array to a local GeoTIFF, upload it to GCS, and load it into Earth Engine.

    This function converts a two-dimensional NumPy array to a local
    Cloud Optimized GeoTIFF, uploads the file to Google Cloud Storage,
    and loads it into Earth Engine as a float image.

    The procedure consists of the following steps:

    Step 0
        Validate the input array and normalize the NoData value.

    Step 1
        Construct the effective NoData mask.

    Step 2
        Convert the array to float32 and materialize NoData values.

    Step 3
        Prepare the local output directory and file path.

    Step 4
        Write the local Cloud Optimized GeoTIFF.

    Step 5
        Retrieve or create the target bucket and upload the GeoTIFF to
        Google Cloud Storage.

    Step 6
        Load the uploaded GeoTIFF into Earth Engine and return output
        metadata.

    Step 7
        Optionally clean up temporary local storage.

    Parameters
    ----------
    array_np : np.ndarray
        Two-dimensional input array.
    transform : affine.Affine
        Affine transform describing raster georeferencing.
    crs : str
        Coordinate reference system identifier.
    bucket_name : str
        Target Google Cloud Storage bucket name.
    project_id : str
        Google Cloud project identifier.
    band_name : str, default="acc"
        Output Earth Engine band name.
    nodata_mask : np.ndarray, optional
        Boolean mask indicating invalid cells (True = NoData). If None,
        non-finite values in `array_np` are treated as NoData.
    nodata_value : float, default=-9999.0
        Numeric value used to materialize NoData in the GeoTIFF.
    tmp_dir : str, optional
        Directory used for temporary local output. If None, a temporary
        directory is created.
    object_prefix : str, default="twi_uploads"
        Prefix used for the object name in Google Cloud Storage.
    cleanup_local : bool, default=False
        If True and `tmp_dir` is not provided, remove the temporary
        directory after upload.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
            - "image": loaded Earth Engine image
            - "gs_uri": Google Cloud Storage URI
            - "local_path": local GeoTIFF path
            - "bucket_object": object name inside the bucket

    Notes
    -----
    Earth Engine initialization must be performed by the caller before
    calling this function.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate input array and normalize NoData value
    # ---------------------------------------------------------------------
    if not np.isfinite(nodata_value):
        nodata_value = -9999.0

    arr = np.asarray(array_np)
    if arr.ndim != 2:
        raise ValueError("push_array_to_ee_geotiff expects a 2D array.")

    # ---------------------------------------------------------------------
    # Step 1: Construct the effective NoData mask
    # ---------------------------------------------------------------------
    if nodata_mask is None:
        mask = ~np.isfinite(arr)
    else:
        mask = np.asarray(nodata_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("nodata_mask must have the same shape as array_np.")

    # ---------------------------------------------------------------------
    # Step 2: Convert to float32 and materialize NoData values
    # ---------------------------------------------------------------------
    arr_f32 = arr.astype(np.float32, copy=False)

    if mask.any():
        arr_f32 = arr_f32.copy()
        arr_f32[mask] = float(nodata_value)
    elif np.isnan(arr_f32).any():
        arr_f32 = np.where(np.isnan(arr_f32), nodata_value, arr_f32).astype(
            np.float32,
            copy=False,
        )

    # ---------------------------------------------------------------------
    # Step 3: Prepare the local output directory and file path
    # ---------------------------------------------------------------------
    object_prefix = object_prefix.strip().strip("/")
    safe_band = band_name.strip() or "band"
    timestamp = int(time.time())
    tif_name = f"{safe_band}_{timestamp}.tif"

    temp_ctx = None
    if tmp_dir is None:
        if cleanup_local:
            temp_ctx = tempfile.TemporaryDirectory()
            tmp_dir = temp_ctx.name
        else:
            tmp_dir = tempfile.mkdtemp()

    local_path = os.path.join(tmp_dir, tif_name)

    try:
        # -----------------------------------------------------------------
        # Step 4: Write the local Cloud Optimized GeoTIFF
        # -----------------------------------------------------------------
        _write_cog_local(
            arr_f32,
            transform=transform,
            crs=crs,
            out_path=local_path,
            nodata_value=float(nodata_value),
            blocksize=512,
            compress="LZW",
        )

        # -----------------------------------------------------------------
        # Step 5: Upload the GeoTIFF to Google Cloud Storage
        # -----------------------------------------------------------------
        storage_client = storage.Client(project=project_id)
        bucket = _get_or_create_bucket(
            storage_client,
            bucket_name,
            project_id=project_id,
            location="US",
            storage_class="STANDARD",
        )

        object_name = f"{object_prefix}/{tif_name}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path, content_type="image/tiff")
        gs_uri = f"gs://{bucket.name}/{object_name}"

        # -----------------------------------------------------------------
        # Step 6: Load the uploaded GeoTIFF into Earth Engine
        # -----------------------------------------------------------------
        ee_img = ee.Image.loadGeoTIFF(gs_uri).rename(safe_band).toFloat()

        return {
            "image": ee_img,
            "gs_uri": gs_uri,
            "local_path": local_path,
            "bucket_object": object_name,
        }

    finally:
        # -----------------------------------------------------------------
        # Step 7: Optionally clean up temporary local storage
        # -----------------------------------------------------------------
        if temp_ctx is not None:
            temp_ctx.cleanup()
