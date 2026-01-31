# scripts/visualization.py
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio

LayerSpec = Tuple[ee.Image, dict, str]


def visualize_map(layers: Sequence[LayerSpec]) -> geemap.Map:
    """
    Create an interactive map and add EE image layers.

    Parameters
    ----------
    layers
        Sequence of (image, vis_params, name) tuples.
        - image: ee.Image
        - vis_params: dict compatible with Map.addLayer()
        - name: layer name shown in the layer control

    Returns
    -------
    geemap.Map
        Interactive map instance with added layers.
    """
    m = geemap.Map(basemap="Esri.WorldImagery")
    m.add_basemap("Esri.WorldTopoMap")

    for item in layers:
        try:
            image, vis_params, name = item
        except Exception as e:
            raise ValueError(
                "Each layer must be a tuple: (ee.Image, vis_params: dict, name: str)."
            ) from e

        if not isinstance(image, ee.image.Image):
            # Keep this a warning; visualization should not hard-fail if one layer is wrong.
            print(f"⚠ Warning: Layer '{name}' is not an ee.Image and will be skipped.")
            continue

        m.addLayer(image, vis_params, name)

    return m


def vis_2sigma(
    image: ee.Image,
    band: str,
    region: ee.Geometry,
    scale: float,
    *,
    k: float = 2.0,
    palette: Optional[Sequence[str]] = None,
    clamp_to_pct: Optional[Tuple[int, int]] = None,
    best_effort: bool = True,
    max_pixels: float = 1e13,
    tile_scale: int = 4,
) -> dict:
    """
    Build visualization parameters using a μ ± k·σ stretch over a region.

    Notes
    -----
    - Statistics are computed over 'region' using reduceRegion.
    - Masked pixels are ignored by reducers.
    - If the region is fully masked (no valid pixels), this function falls back
      to a conservative default range (0..1) to avoid runtime errors.

    Parameters
    ----------
    image
        Source image.
    band
        Band name to visualize.
    region
        Geometry used for statistics.
    scale
        Pixel scale (meters) used for reduceRegion.
    k
        Sigma multiplier for the stretch; default 2.0.
    palette
        Optional color palette for visualization.
    clamp_to_pct
        Optional percentile clamp, e.g. (2, 98), to make the stretch more robust.
        The final min/max are clamped to the percentile range.
    best_effort, max_pixels, tile_scale
        Performance/robustness controls for reduceRegion.

    Returns
    -------
    dict
        Map.addLayer()-compatible visualization parameters:
        {'bands': [band], 'min': <float>, 'max': <float>, 'palette': [...]}.
    """
    img = image.select([band])

    # Compute mean and standard deviation over the region (masked pixels ignored).
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region,
        scale=scale,
        bestEffort=best_effort,
        maxPixels=max_pixels,
        tileScale=tile_scale,
    )

    mu = ee.Number(stats.get(f"{band}_mean"))
    sig = ee.Number(stats.get(f"{band}_stdDev"))

    # Guard against missing stats (e.g., fully masked region).
    # If mu or sig is null, getInfo() would fail.
    mu_is_null = stats.get(f"{band}_mean").eq(None)
    sig_is_null = stats.get(f"{band}_stdDev").eq(None)

    vmin = ee.Number(0)
    vmax = ee.Number(1)

    # If stats are present, compute μ ± k·σ.
    vmin = ee.Algorithms.If(
        mu_is_null.Or(sig_is_null),
        vmin,
        mu.subtract(sig.multiply(k)),
    )
    vmax = ee.Algorithms.If(
        mu_is_null.Or(sig_is_null),
        vmax,
        mu.add(sig.multiply(k)),
    )

    vmin = ee.Number(vmin)
    vmax = ee.Number(vmax)

    # Optional percentile clamp for robustness (reduces influence of extreme outliers).
    if clamp_to_pct is not None:
        lo, hi = clamp_to_pct
        p = img.reduceRegion(
            reducer=ee.Reducer.percentile([lo, hi]),
            geometry=region,
            scale=scale,
            bestEffort=best_effort,
            maxPixels=max_pixels,
            tileScale=tile_scale,
        )
        pmin = ee.Number(p.get(f"{band}_p{lo}"))
        pmax = ee.Number(p.get(f"{band}_p{hi}"))

        # Guard against missing percentiles as well.
        pmin_is_null = p.get(f"{band}_p{lo}").eq(None)
        pmax_is_null = p.get(f"{band}_p{hi}").eq(None)

        vmin = ee.Algorithms.If(pmin_is_null, vmin, vmin.max(pmin))
        vmax = ee.Algorithms.If(pmax_is_null, vmax, vmax.min(pmax))

        vmin = ee.Number(vmin)
        vmax = ee.Number(vmax)

    # Bring min/max to client side for geemap Map.addLayer().
    params = {"bands": [band], "min": float(vmin.getInfo()), "max": float(vmax.getInfo())}
    if palette:
        params["palette"] = list(palette)

    return params

def plot_tif(
    tif_path: str,
    p_low: float = 2.0,
    p_high: float = 98.0,
    label: str = "TWI",
    title: str | None = None,
):
    """
    Continuous visualization of a single-band GeoTIFF.
    Low values = red, high values = blue (RdYlBu colormap).
    The display range is derived from the [p_low, p_high] percentiles
    (values outside this range are clipped to the extreme colors).
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("Raster has no valid (finite) values.")

    # Percentile-based stretch
    vmin = float(np.nanpercentile(valid, p_low))
    vmax = float(np.nanpercentile(valid, p_high))

    # Degeneracy protection
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="RdYlBu")
    cbar = plt.colorbar(im)
    cbar.set_label(label)

    if title is None:
        title = tif_path
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
