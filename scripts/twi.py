import ee
import numpy as np

def compute_twi(flow_accumulation, slope):
    """
    Výpočet Topographic Wetness Index (TWI).
    """
    safe_slope = slope.where(slope.eq(0), 0.1)
    tan_slope = safe_slope.divide(180).multiply(ee.Number(3.14159265359)).tan()
    twi = flow_accumulation.divide(tan_slope).log().rename("TWI")
    #scaled_twi = twi.multiply(1e8).toInt().rename("TWI_scaled")

    return twi

def compute_twi_numpy(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    acc_is_area: bool,
    cell_area: float = None,
    min_slope_deg: float = 0.1,
    nodata_mask: np.ndarray = None,
    out_dtype: str = "float32"
) -> np.ndarray:
    """
    Compute TWI (Topographic Wetness Index) from numpy arrays.

    Formula:
      TWI = ln( a / tan(beta) )
    where:
      - a = upslope contributing area [m²]
      - beta = slope angle in radians

    Parameters:
      - acc_np: numpy array of accumulation (either count of cells or area)
      - slope_deg_np: numpy array of slopes in degrees
      - acc_is_area: True if acc_np already gives area in m²; False if it is number of cells
      - cell_area: area of each cell in m² (required if acc_is_area=False)
      - min_slope_deg: minimum slope (in degrees) to avoid tan(0) or extremely steep slopes
      - nodata_mask: boolean mask array (True where nodata / invalid)
      - out_dtype: output dtype (e.g. "float32")

    Returns:
      - numpy array of TWI (float) with same shape as inputs
    """

    # Convert to float64 for stable computations
    acc = np.array(acc_np, dtype=np.float64)
    slope_deg = np.array(slope_deg_np, dtype=np.float64)

    # Handle nodata_mask: if provided, mask out those cells
    if nodata_mask is not None:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)
    else:
        nodata_mask = np.zeros(acc.shape, dtype=bool)

    # Compute upslope area a in m²
    if acc_is_area:
        a = acc
    else:
        if cell_area is None:
            raise ValueError("cell_area must be provided if acc_is_area=False")
        a = acc * float(cell_area)

    # Replace non-finite or negative values in a with a small positive to avoid log zeros
    # But preserve nodata_mask separately
    a_safe = np.where((~nodata_mask) & (a > 0) & np.isfinite(a), a, np.nan)

    # Enforce minimum slope degree to avoid tan(0)
    slope_deg_safe = np.maximum(slope_deg, min_slope_deg)
    # Convert to radians
    slope_rad = np.deg2rad(slope_deg_safe)

    # Compute tangent
    tan_beta = np.tan(slope_rad)
    # Avoid zeros or negative (shouldn't be), set very low floor
    tan_beta = np.where((~nodata_mask) & (tan_beta > 0) & np.isfinite(tan_beta), tan_beta, 1e-6)

    # Compute TWI
    twi = np.log(a_safe / tan_beta)

    # Apply nodata mask: set to nan where nodata
    twi = np.where(nodata_mask, np.nan, twi)

    # Cast to output dtype
    twi = twi.astype(out_dtype)

    return twi
