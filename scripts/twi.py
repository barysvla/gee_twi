from __future__ import annotations

from typing import Literal, Optional

import ee
import numpy as np


def compute_twi(
    flow_acc: ee.Image,
    slope_deg: ee.Image,
    *,
    min_slope_deg: float = 0.1
) -> ee.Image:
    """
    Compute the Topographic Wetness Index (TWI) in Earth Engine.

    The index is calculated according to the standard definition:

        TWI = ln( A / tan(beta) )

    where:
        A     ... contributing area
        beta  ... local slope angle (in radians)

    Parameters
    ----------
    flow_acc : ee.Image
        Raster of contributing area (must be on the target grid).
    slope_deg : ee.Image
        Slope raster in degrees.
    min_slope_deg : float, optional
        Minimum slope threshold (in degrees) used to prevent
        numerical instability caused by tan(0).

    Returns
    -------
    ee.Image
        Single-band image named "TWI".
    """

    # Enforce a minimum slope to avoid division by zero
    # (tan(0) = 0 → undefined logarithm)
    safe_slope = slope_deg.max(ee.Number(float(min_slope_deg)))

    # Convert slope from degrees to radians and compute tan(beta)
    tan_slope = safe_slope.multiply(np.pi).divide(180.0).tan()

    # Compute TWI = ln(A / tan(beta))
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

    The index is calculated according to the standard definition:

        TWI = ln( A / tan(beta) )

    where:
        A     ... contributing area
        beta  ... local slope angle (in radians)

    Parameters
    ----------
    acc_np : np.ndarray
        Contributing area raster defined on the target grid.
        Units must already correspond to the chosen TWI definition.
    slope_deg_np : np.ndarray
        Slope raster in degrees on the same grid.
    min_slope_deg : float, optional
        Minimum slope threshold (in degrees) used to prevent
        numerical instability caused by tan(0).
    nodata_mask : np.ndarray or None
        Optional boolean mask (True = NoData). Masked cells are set to NaN.
    out_dtype : str
        Output dtype (default: float32).

    Returns
    -------
    np.ndarray
        (H, W) array representing TWI with NoData encoded as NaN.
    """

    # Promote inputs to float64 for numerical stability
    acc = np.asarray(acc_np, dtype=np.float64)
    slope_deg = np.asarray(slope_deg_np, dtype=np.float64)

    # Validate grid consistency
    if acc.shape != slope_deg.shape:
        raise ValueError(f"Shape mismatch: acc {acc.shape} vs slope {slope_deg.shape}")

    # Initialize invalid mask (non-finite inputs treated as NoData)
    invalid = ~np.isfinite(acc) | ~np.isfinite(slope_deg)

    # Apply optional external NoData mask
    if nodata_mask is not None:
        m = np.asarray(nodata_mask, dtype=bool)
        if m.shape != acc.shape:
            raise ValueError("nodata_mask must have the same shape as inputs")
        invalid |= m

    # Enforce minimum slope to avoid division by zero (tan(0) = 0)
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))

    # Convert slope from degrees to radians and compute tan(beta)
    tan_slope = np.tan(np.deg2rad(slope_deg_safe))

    # Allocate output array (NaN by default)
    twi = np.full(acc.shape, np.nan, dtype=np.float64)

    # Compute TWI only on valid cells
    valid = ~invalid
    twi[valid] = np.log(acc[valid] / tan_slope[valid])

    return twi.astype(out_dtype, copy=False)
