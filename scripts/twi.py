from __future__ import annotations

"""
Topographic Wetness Index computation for the final workflow output.

This script computes the Topographic Wetness Index (TWI) from flow
accumulation and slope rasters on a shared grid. It represents the final
analytical step of the workflow.

The functions provided are:
    - twi_ee: computes TWI for Earth Engine images (server-side evaluation)
    - twi_np: computes TWI for NumPy arrays (local evaluation)

These functions enable consistent TWI computation in both cloud-based
(GEE) and local (NumPy) processing modes.
"""

from typing import Optional

import ee
import numpy as np


def twi_ee(
    flow_acc: ee.Image,
    slope_deg: ee.Image,
    *,
    min_slope_deg: float = 0.1,
) -> ee.Image:
    """
    Compute the Topographic Wetness Index (TWI) in Earth Engine.

    The index is computed as

        TWI = ln(A / tan(beta))

    where `A` is the contributing area and `beta` is the local slope
    angle in radians. TWI expresses the tendency of water accumulation
    controlled by the balance between contributing area and local slope.

    The procedure consists of the following steps:

    Step 0
        Validate input parameters.

    Step 1
        Define the valid computational domain (implicitly handled by EE masks).

    Step 2
        Enforce a minimum slope threshold to ensure numerical stability.

    Step 3
        Convert slope from degrees to radians and evaluate tan(beta).

    Step 4
        Compute TWI as ln(A / tan(beta)).


    Parameters
    ----------
    flow_acc : ee.Image
        Contributing-area raster on the target grid.
    slope_deg : ee.Image
        Slope raster in degrees.
    min_slope_deg : float, default=0.1
        Minimum slope threshold in degrees.

    Returns
    -------
    ee.Image
        Single-band image named `"TWI"`.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate input parameters
    # ---------------------------------------------------------------------
    # Ensure that the minimum slope threshold is physically meaningful.
    if min_slope_deg < 0.0:
        raise ValueError("min_slope_deg must be non-negative.")

    # ---------------------------------------------------------------------
    # Step 1: Define the valid computational domain (implicit in EE)
    # ---------------------------------------------------------------------
    # Earth Engine propagates masks automatically, so invalid pixels
    # (e.g. NoData) are excluded from all subsequent operations.

    # ---------------------------------------------------------------------
    # Step 2: Enforce minimum slope threshold
    # ---------------------------------------------------------------------
    # Prevent tan(beta) → 0 in flat areas, which would lead to division
    # by zero and undefined logarithm values.
    safe_slope = slope_deg.max(ee.Number(float(min_slope_deg)))

    # ---------------------------------------------------------------------
    # Step 3: Convert slope and compute tan(beta)
    # ---------------------------------------------------------------------
    # Convert slope from degrees to radians and evaluate the tangent,
    # which represents the local gradient.
    tan_slope = safe_slope.multiply(np.pi).divide(180.0).tan()

    # ---------------------------------------------------------------------
    # Step 4: Compute TWI
    # ---------------------------------------------------------------------
    # Evaluate the TWI formula as the logarithm of the ratio between
    # contributing area and local slope.
    twi = flow_acc.divide(tan_slope).log()

    return twi.rename("TWI")


def twi_np(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    min_slope_deg: float = 0.1,
    nodata_mask: Optional[np.ndarray] = None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Compute the Topographic Wetness Index (TWI) from NumPy arrays.

    The index is computed as

        TWI = ln(A / tan(beta))

    where `A` is the contributing area and `beta` is the local slope
    angle in radians. TWI expresses the tendency of water accumulation
    controlled by the balance between contributing area and local slope.

    The procedure consists of the following steps:

    Step 0
        Validate inputs and ensure grid consistency.

    Step 1
        Define the valid computational domain using masks and constraints.

    Step 2
        Enforce a minimum slope threshold to ensure numerical stability.

    Step 3
        Convert slope from degrees to radians and evaluate tan(beta).

    Step 4
        Compute TWI for valid cells.


    Parameters
    ----------
    acc_np : np.ndarray
        Contributing-area raster on the target grid.
    slope_deg_np : np.ndarray
        Slope raster in degrees on the same grid.
    min_slope_deg : float, default=0.1
        Minimum slope threshold in degrees.
    nodata_mask : np.ndarray, optional
        Boolean mask indicating invalid cells (True = NoData).
    out_dtype : str, default="float32"
        Output data type.

    Returns
    -------
    np.ndarray
        Two-dimensional TWI array with NoData represented as NaN.
    """
    # ---------------------------------------------------------------------
    # Step 0: Validate inputs and normalize arrays
    # ---------------------------------------------------------------------
    # Promote inputs to float64 for numerically stable division and logarithm.
    acc = np.asarray(acc_np, dtype=np.float64)
    slope_deg = np.asarray(slope_deg_np, dtype=np.float64)

    # Ensure both rasters share the same grid.
    if acc.shape != slope_deg.shape:
        raise ValueError(f"Shape mismatch: acc {acc.shape} vs slope {slope_deg.shape}")

    # Ensure that the minimum slope threshold is physically meaningful.
    if min_slope_deg < 0.0:
        raise ValueError("min_slope_deg must be non-negative.")

    # ---------------------------------------------------------------------
    # Step 1: Define the valid computational domain
    # ---------------------------------------------------------------------
    # Identify invalid cells explicitly, since NumPy does not propagate masks.
    invalid = ~np.isfinite(acc) | ~np.isfinite(slope_deg)

    # Incorporate external NoData mask if provided.
    if nodata_mask is not None:
        mask = np.asarray(nodata_mask, dtype=bool)
        if mask.shape != acc.shape:
            raise ValueError("nodata_mask must have the same shape as the input arrays.")
        invalid |= mask

    # Exclude cells with non-positive contributing area.
    invalid |= acc <= 0.0

    # ---------------------------------------------------------------------
    # Step 2: Enforce minimum slope threshold
    # ---------------------------------------------------------------------
    # Prevent tan(beta) → 0 in flat areas, which would lead to division
    # by zero and undefined logarithm values.
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))

    # ---------------------------------------------------------------------
    # Step 3: Convert slope and compute tan(beta)
    # ---------------------------------------------------------------------
    # Convert slope from degrees to radians and evaluate the tangent,
    # which represents the local gradient.
    tan_slope = np.tan(np.deg2rad(slope_deg_safe))

    # ---------------------------------------------------------------------
    # Step 4: Compute TWI
    # ---------------------------------------------------------------------
    # Evaluate the TWI formula only for valid cells to avoid invalid operations.
    twi = np.full(acc.shape, np.nan, dtype=np.float64)
    valid = ~invalid
    twi[valid] = np.log(acc[valid] / tan_slope[valid])

    return twi.astype(out_dtype, copy=False)
