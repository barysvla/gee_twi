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
    flow_acc_m2 = flow_acc.multiply(1e6) 
    
    # TWI = ln(A / tan(beta))
    return flow_acc_m2.divide(tan_slope).log().rename("TWI")


def compute_twi_numpy(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    acc_is_area: bool,
    acc_units: Literal["m2", "km2"] = "m2",
    cell_area: Optional[float | np.ndarray] = None,
    min_slope_deg: float = 0.1,
    min_tan: float = 1e-6,
    nodata_mask: Optional[np.ndarray] = None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Compute TWI from NumPy arrays on a shared grid.

    Notes:
    - slope is in degrees
    - if acc_is_area=True, acc_np is contributing area (m² or km² via acc_units)
    - NoData is returned as NaN
    """
    # Use float64 for numerical stability during log/tan operations
    acc = np.asarray(acc_np, dtype=np.float64)
    slope_deg = np.asarray(slope_deg_np, dtype=np.float64)

    # Basic shape validation (both rasters must be on the same grid)
    if acc.shape != slope_deg.shape:
        raise ValueError(f"Shape mismatch: acc {acc.shape} vs slope {slope_deg.shape}")

    # Build/validate NoData mask
    if nodata_mask is None:
        mask = np.zeros(acc.shape, dtype=bool)
    else:
        mask = np.asarray(nodata_mask, dtype=bool)
        if mask.shape != acc.shape:
            raise ValueError("nodata_mask must have the same shape as inputs")

    # Treat any non-finite inputs as NoData
    mask |= ~np.isfinite(acc) | ~np.isfinite(slope_deg)

    # Convert accumulation to contributing area 'a' in m²:
    # - either already an area raster (m² / km²)
    # - or cell-count accumulation converted by cell_area (scalar or per-cell raster)
    if acc_is_area:
        a = acc.copy()
        if acc_units == "km2":
            a *= 1e6
        elif acc_units != "m2":
            raise ValueError(f"Unsupported acc_units: {acc_units}")
    else:
        if cell_area is None:
            raise ValueError("cell_area must be provided when acc_is_area=False")

        # Support scalar cell area or per-cell area raster
        if np.isscalar(cell_area):
            a = acc * float(cell_area)
        else:
            ca = np.asarray(cell_area, dtype=np.float64)
            if ca.shape != acc.shape:
                raise ValueError("cell_area array must match input shape")
            mask |= ~np.isfinite(ca)
            a = acc * ca

    # Log argument must be strictly positive
    mask |= (a <= 0.0)

    # Numerical guard: avoid tan(0) by enforcing a minimum slope in degrees
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))
    slope_rad = np.deg2rad(slope_deg_safe)

    # Numerical guard: enforce a small positive floor for tan(beta)
    tan_beta = np.tan(slope_rad)
    tan_beta = np.maximum(tan_beta, float(min_tan))
    mask |= ~np.isfinite(tan_beta) | (tan_beta <= 0.0)

    # Allocate output and compute only on valid cells
    twi = np.full(acc.shape, np.nan, dtype=np.float64)
    valid = ~mask
    twi[valid] = np.log(a[valid] / tan_beta[valid])

    return twi.astype(out_dtype, copy=False)
