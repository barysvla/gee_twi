from __future__ import annotations

from typing import Literal, Optional

import ee
import numpy as np


def compute_twi(flow_acc: ee.Image, slope_deg: ee.Image, *, min_slope_deg: float = 0.1) -> ee.Image:
    """
    Compute TWI in Earth Engine from flow accumulation and slope (degrees).

    Notes:
    - slope is thresholded to avoid division by zero
    - output is an ee.Image named "TWI"
    """
    # Numerical guard: avoid tan(0) by enforcing a minimum slope (degrees)
    safe_slope = slope_deg.max(ee.Number(float(min_slope_deg)))

    # Convert degrees -> radians and compute tan(beta)
    tan_slope = safe_slope.multiply(np.pi).divide(180.0).tan()

    # Convert contributing area from km² to m² to match the standard TWI definition
    #flow_acc_m2 = flow_acc.multiply(1e6) 
    
    # TWI = ln(A / tan(beta))
    return flow_acc.divide(tan_slope).log().rename("TWI")


def compute_twi_numpy(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    min_slope_deg: float = 0.1,
    nodata_mask: Optional[np.ndarray] = None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Compute TWI from NumPy arrays: TWI = ln( A / tan(beta) )

    Parameters
    ----------
    acc_np : np.ndarray
        Contributing area / accumulation raster on the target grid (must be same grid as slope).
        Units must already be consistent with your chosen TWI definition.
    slope_deg_np : np.ndarray
        Slope raster in degrees on the same grid.
    min_slope_deg : float
        Minimum slope (degrees) used to avoid tan(0).
    nodata_mask : np.ndarray or None
        Optional boolean mask (True = NoData). Masked cells are set to NaN.
    out_dtype : str
        Output dtype, default "float32".

    Returns
    -------
    np.ndarray
        (H, W) array of TWI with NoData as NaN.
    """
    acc = np.asarray(acc_np, dtype=np.float64)
    slope_deg = np.asarray(slope_deg_np, dtype=np.float64)

    if acc.shape != slope_deg.shape:
        raise ValueError(f"Shape mismatch: acc {acc.shape} vs slope {slope_deg.shape}")

    # Start with invalid where inputs are non-finite
    invalid = ~np.isfinite(acc) | ~np.isfinite(slope_deg)

    # Optional external NoData mask
    if nodata_mask is not None:
        m = np.asarray(nodata_mask, dtype=bool)
        if m.shape != acc.shape:
            raise ValueError("nodata_mask must have the same shape as inputs")
        invalid |= m

    # Numerical guard: enforce minimum slope (degrees)
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))

    # tan(beta) where beta in radians
    tan_slope = np.tan(np.deg2rad(slope_deg_safe))

    # Compute TWI; keep NaN for invalid cells
    twi = np.full(acc.shape, np.nan, dtype=np.float64)
    valid = ~invalid
    twi[valid] = np.log(acc[valid] / tan_slope[valid])

    return twi.astype(out_dtype, copy=False)
    
