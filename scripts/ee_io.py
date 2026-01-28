from __future__ import annotations

import os
import time
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import ee
import rasterio
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden, Conflict, BadRequest


def _get_or_create_bucket(
    storage_client: storage.Client,
    bucket_name: str,
    *,
    project_id: str,
    location: str = "US",
    storage_class: str = "STANDARD",
) -> storage.Bucket:
    """
    Get an existing GCS bucket or create it if missing.

    Notes
    -----
    Earth Engine's ee.Image.loadGeoTIFF requires the bucket to be in an EE-compatible location.
    This function enforces a safe subset: US multi-region or us-central1.
    """
    bucket_name = bucket_name.lower().strip()

    try:
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()
    except NotFound:
        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class

        # Optional (recommended): uniform bucket-level access
        try:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        except Exception:
            pass

        bucket = storage_client.create_bucket(bucket, project=project_id, location=location)
        bucket.reload()
    except Conflict:
        # Rare race: bucket created between lookup and create
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()
    except Forbidden as e:
        raise RuntimeError(
            f"Forbidden to access bucket '{bucket_name}'. "
            "Missing permissions: storage.buckets.get and/or storage.buckets.create."
        ) from e
    except BadRequest as e:
        raise RuntimeError(f"Bad request when accessing/creating bucket '{bucket_name}': {e}") from e

    # Enforce a strict, predictable location policy (avoid substring checks)
    loc = (bucket.location or "").upper().strip()
    if loc not in {"US", "US-CENTRAL1"}:
        raise RuntimeError(
            f"Bucket '{bucket_name}' is in location '{bucket.location}'. "
            "ee.Image.loadGeoTIFF requires US multi-region or US-CENTRAL1 (safe subset enforced here)."
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
    Write a single-band Cloud Optimized GeoTIFF (COG).

    Requirements
    ------------
    - arr_f32 must be float32
    - NaNs must already be replaced with nodata_value
    """
    if arr_f32.dtype != np.float32:
        raise ValueError("_write_cog_local expects float32 input array.")
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
    Write a 2D numpy array to a local COG, upload it to GCS, and load it into Earth Engine.

    Returns
    -------
    dict with keys:
      - image: ee.Image
      - gs_uri: str
      - local_path: str
      - bucket_object: str
    """
    if not np.isfinite(nodata_value):
        nodata_value = -9999.0

    arr = np.asarray(array_np)
    if arr.ndim != 2:
        raise ValueError("push_array_to_ee_geotiff expects a 2D array (single band).")

    # Prepare nodata mask
    if nodata_mask is None:
        mask = ~np.isfinite(arr)
    else:
        mask = np.asarray(nodata_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("nodata_mask must have the same shape as array_np.")

    # Convert to float32 and replace NaNs/non-finite with nodata_value
    arr_f32 = arr.astype(np.float32, copy=False)
    if mask.any():
        arr_f32 = arr_f32.copy()
        arr_f32[mask] = float(nodata_value)
    else:
        # Still ensure no NaNs sneak in
        if np.isnan(arr_f32).any():
            arr_f32 = np.where(np.isnan(arr_f32), nodata_value, arr_f32).astype(np.float32, copy=False)

    # Prepare local output path
    object_prefix = object_prefix.strip().strip("/")
    safe_band = band_name.strip() or "band"
    tstamp = int(time.time())
    tif_name = f"{safe_band}_{tstamp}.tif"

    temp_ctx = None
    if tmp_dir is None:
        if cleanup_local:
            temp_ctx = tempfile.TemporaryDirectory()
            tmp_dir = temp_ctx.name
        else:
            tmp_dir = tempfile.mkdtemp()

    local_path = os.path.join(tmp_dir, tif_name)

    # Write local COG
    try:
        _write_cog_local(
            arr_f32,
            transform=transform,
            crs=crs,
            out_path=local_path,
            nodata_value=float(nodata_value),
            blocksize=512,
            compress="LZW",
        )

        # Upload to GCS (ensure bucket exists and location is compatible)
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

        # Load into EE (ee.Initialize() must be done by caller)
        ee_img = ee.Image.loadGeoTIFF(gs_uri).rename(safe_band).toFloat()

        return {
            "image": ee_img,
            "gs_uri": gs_uri,
            "local_path": local_path,
            "bucket_object": object_name,
        }
    finally:
        # Optional cleanup for TemporaryDirectory mode
        if temp_ctx is not None:
            temp_ctx.cleanup()
