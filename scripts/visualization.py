from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence, Tuple

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio

LayerSpec = Tuple[ee.Image, dict, str]


def visualize_map(layers: Sequence[LayerSpec]) -> geemap.Map:
    """
    Create an interactive geemap instance and add Earth Engine image layers.

    Parameters
    ----------
    layers : sequence of tuple
        Sequence of `(image, vis_params, name)` tuples, where:
            - `image` is an `ee.Image`
            - `vis_params` is a dictionary accepted by `Map.addLayer()`
            - `name` is the layer name shown in the layer control

    Returns
    -------
    geemap.Map
        Interactive map with the requested layers.
    """
    map_obj = geemap.Map(basemap="Esri.WorldImagery")
    map_obj.add_basemap("Esri.WorldTopoMap")

    for item in layers:
        try:
            image, vis_params, name = item
        except Exception as exc:
            raise ValueError(
                "Each layer must be a tuple: (ee.Image, vis_params: dict, name: str)."
            ) from exc

        if not isinstance(image, ee.image.Image):
            warnings.warn(
                f"Layer '{name}' is not an ee.Image and will be skipped.",
                stacklevel=2,
            )
            continue

        map_obj.addLayer(image, vis_params, name)

    return map_obj


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

    Statistics are computed over the supplied region using Earth Engine
    reducers. Masked pixels are ignored. If the region contains no valid
    pixels, the function falls back to a conservative default range.

    The procedure consists of the following steps:

    Step 0
        Select the target band and validate optional percentile clamp
        settings.

    Step 1
        Count valid pixels in the region to detect empty or fully masked
        inputs.

    Step 2
        Compute mean and standard deviation over the region.

    Step 3
        Derive the sigma-based stretch and fall back to a default range
        when the result is degenerate.

    Step 4
        Optionally clamp the sigma stretch to a percentile range.

    Step 5
        Convert the final min and max values to client-side parameters
        compatible with `Map.addLayer()`.

    Parameters
    ----------
    image : ee.Image
        Source image.
    band : str
        Band name to visualize.
    region : ee.Geometry
        Geometry used for statistics.
    scale : float
        Pixel scale used for `reduceRegion`.
    k : float, default=2.0
        Sigma multiplier for the stretch.
    palette : sequence of str, optional
        Optional colour palette for visualization.
    clamp_to_pct : tuple of int, optional
        Optional percentile clamp, for example `(2, 98)`. The final
        sigma-based stretch is constrained to this percentile interval.
    best_effort : bool, default=True
        Passed to `reduceRegion`.
    max_pixels : float, default=1e13
        Passed to `reduceRegion`.
    tile_scale : int, default=4
        Passed to `reduceRegion`.

    Returns
    -------
    dict
        Visualization dictionary compatible with `Map.addLayer()`.
    """
    # ---------------------------------------------------------------------
    # Step 0: Select the target band and validate optional inputs
    # ---------------------------------------------------------------------
    img = image.select([band])

    if clamp_to_pct is not None:
        lo, hi = clamp_to_pct
        if not (0 <= lo < hi <= 100):
            raise ValueError("clamp_to_pct must satisfy 0 <= low < high <= 100.")

    # ---------------------------------------------------------------------
    # Step 1: Count valid pixels in the target region
    # ---------------------------------------------------------------------
    # If the region is empty or fully masked, reduceRegion(count) may
    # return null. A default value of 0 is therefore supplied.
    n_valid = ee.Number(
        img.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=scale,
            bestEffort=best_effort,
            maxPixels=max_pixels,
            tileScale=tile_scale,
        ).get(band, 0)
    )

    # ---------------------------------------------------------------------
    # Step 2: Compute regional mean and standard deviation
    # ---------------------------------------------------------------------
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region,
        scale=scale,
        bestEffort=best_effort,
        maxPixels=max_pixels,
        tileScale=tile_scale,
    )

    # Provide safe defaults so that ee.Number() never receives null.
    mu = ee.Number(stats.get(f"{band}_mean", 0))
    sigma = ee.Number(stats.get(f"{band}_stdDev", 0))

    # ---------------------------------------------------------------------
    # Step 3: Derive the sigma-based stretch
    # ---------------------------------------------------------------------
    default_min = ee.Number(0)
    default_max = ee.Number(1)

    vmin_sigma = mu.subtract(sigma.multiply(k))
    vmax_sigma = mu.add(sigma.multiply(k))

    use_sigma = n_valid.gt(0).And(vmax_sigma.neq(vmin_sigma)).And(sigma.gt(0))

    vmin = ee.Number(ee.Algorithms.If(use_sigma, vmin_sigma, default_min))
    vmax = ee.Number(ee.Algorithms.If(use_sigma, vmax_sigma, default_max))

    # ---------------------------------------------------------------------
    # Step 4: Optionally clamp the stretch by percentiles
    # ---------------------------------------------------------------------
    if clamp_to_pct is not None:
        percentile_stats = img.reduceRegion(
            reducer=ee.Reducer.percentile([lo, hi]),
            geometry=region,
            scale=scale,
            bestEffort=best_effort,
            maxPixels=max_pixels,
            tileScale=tile_scale,
        )

        pmin = ee.Number(percentile_stats.get(f"{band}_p{lo}", vmin))
        pmax = ee.Number(percentile_stats.get(f"{band}_p{hi}", vmax))

        vmin = ee.Number(ee.Algorithms.If(n_valid.gt(0), vmin.max(pmin), vmin))
        vmax = ee.Number(ee.Algorithms.If(n_valid.gt(0), vmax.min(pmax), vmax))

    # ---------------------------------------------------------------------
    # Step 5: Convert to client-side visualization parameters
    # ---------------------------------------------------------------------
    params = {
        "bands": [band],
        "min": float(vmin.getInfo()),
        "max": float(vmax.getInfo()),
    }
    if palette:
        params["palette"] = list(palette)

    return params


def plot_tif(
    tif_path: str,
    p_low: float = 2.0,
    p_high: float = 98.0,
    label: str = "TWI",
    title: str | None = None,
) -> None:
    """
    Display a single-band GeoTIFF using a percentile-based contrast stretch.

    The display range is derived from the `[p_low, p_high]` percentiles
    computed over valid raster values. Values outside this interval are
    clipped to the extreme colours of the colormap.

    This function is intended for local, static inspection of raster
    outputs.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF.
    p_low : float, default=2.0
        Lower percentile used for contrast stretching.
    p_high : float, default=98.0
        Upper percentile used for contrast stretching.
    label : str, default="TWI"
        Colourbar label.
    title : str, optional
        Plot title. If None, the input filename is used.
    """
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100.")

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

    # Compute contrast stretch only from valid raster values.
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("Raster contains no valid finite values.")

    vmin = float(np.nanpercentile(valid, p_low))
    vmax = float(np.nanpercentile(valid, p_high))

    # Fall back to the full valid range if percentile values collapse.
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))

    plt.figure(figsize=(8, 6))
    image = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="RdYlBu")
    colorbar = plt.colorbar(image)
    colorbar.set_label(label)

    if title is None:
        title = os.path.basename(tif_path)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
