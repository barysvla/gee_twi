from __future__ import annotations

from typing import Optional

import ee
import numpy as np


def compute_twi(
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
    angle in radians. A minimum slope threshold is applied to avoid
    numerical instability.

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
    if min_slope_deg < 0.0:
        raise ValueError("min_slope_deg must be non-negative.")

    # Prevent tan(beta) → 0 for flat areas.
    # Without this, division by zero would lead to +inf and undefined log.
    safe_slope = slope_deg.max(ee.Number(float(min_slope_deg)))

    # Convert degrees → radians and evaluate tan(beta).
    tan_slope = safe_slope.multiply(np.pi).divide(180.0).tan()

    # Compute TWI = ln(A / tan(beta)).
    twi = flow_acc.divide(tan_slope).log()

    return twi.rename("TWI")


def compute_twi_numpy(
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
    angle in radians. A minimum slope threshold is applied to avoid
    numerical instability.

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
    # Promote to float64 for stable division and logarithm.
    # float32 would be more memory-efficient but less stable numerically.
    acc = np.asarray(acc_np, dtype=np.float64)
    slope_deg = np.asarray(slope_deg_np, dtype=np.float64)

    # Grid consistency is required — TWI is defined cell-wise.
    if acc.shape != slope_deg.shape:
        raise ValueError(f"Shape mismatch: acc {acc.shape} vs slope {slope_deg.shape}")
    if min_slope_deg < 0.0:
        raise ValueError("min_slope_deg must be non-negative.")

    # Base invalid mask:
    # - non-finite values (NaN, inf)
    invalid = ~np.isfinite(acc) | ~np.isfinite(slope_deg)

    # Optional external mask (e.g., DEM mask propagated through pipeline)
    if nodata_mask is not None:
        mask = np.asarray(nodata_mask, dtype=bool)
        if mask.shape != acc.shape:
            raise ValueError("nodata_mask must have the same shape as the input arrays.")
        invalid |= mask

    # Contributing area must be strictly positive.
    # Zero or negative values lead to log(0) or undefined behaviour.
    invalid |= acc <= 0.0

    # Enforce minimum slope:
    # prevents tan(beta) → 0 and stabilizes flat areas.
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))

    # Convert slope to radians and compute tan(beta).
    tan_slope = np.tan(np.deg2rad(slope_deg_safe))

    # Allocate output filled with NaN (default NoData representation).
    twi = np.full(acc.shape, np.nan, dtype=np.float64)

    # Compute only on valid cells to avoid warnings and invalid math.
    valid = ~invalid
    twi[valid] = np.log(acc[valid] / tan_slope[valid])

    # Cast to target dtype at the end (after stable computation).
    return twi.astype(out_dtype, copy=False)
