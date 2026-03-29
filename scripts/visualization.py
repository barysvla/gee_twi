from __future__ import annotations

"""
Visualization utilities for inspecting workflow outputs.

This script provides functions for visualizing raster outputs generated
by the workflow in both cloud and local modes. It includes utilities for
deriving visualization parameters from Earth Engine images, displaying
interactive map layers, and rendering GeoTIFF rasters using matplotlib.

The functions provided are:
    - vis_sigma: builds visualization parameters for Earth Engine images
      using a μ ± k·σ stretch (server-side visualization via geemap)
    - show_map: creates an interactive map and adds EE image layers
    - plot_raster: renders a GeoTIFF using percentile-based contrast
      stretching (local visualization of NumPy arrays or GeoTIFF files)

These utilities are intended for exploratory analysis and visual
validation of intermediate and final workflow results.
"""

import os
import warnings
from typing import Optional, Sequence, Tuple

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio

LayerSpec = Tuple[ee.Image, dict, str]


def show_map(layers: Sequence[LayerSpec]) -> geemap.Map:
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
    # Initialize the map with imagery and topographic basemaps.
    map_obj = geemap.Map(basemap="Esri.WorldImagery")
    map_obj.add_basemap("Esri.WorldTopoMap")

    for item in layers:
        try:
            image, vis_params, name = item
        except Exception as exc:
            raise ValueError(
                "Each layer must be a tuple: (ee.Image, vis_params: dict, name: str)."
            ) from exc

        # Skip invalid layer objects but keep the map creation running.
        if not isinstance(image, ee.image.Image):
            warnings.warn(
                f"Layer '{name}' is not an ee.Image and will be skipped.",
                stacklevel=2,
            )
            continue

        map_obj.addLayer(image, vis_params, name)

    return map_obj

def vis_sigma(
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
    pixels or the sigma-based range is degenerate, the function falls 
    back to the default interval [0, 1].

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
    # Step 3: Derive the sigma-based stretch and apply fallback if needed
    # ---------------------------------------------------------------------
    default_min = ee.Number(0)
    default_max = ee.Number(1)

    vmin_sigma = mu.subtract(sigma.multiply(k))
    vmax_sigma = mu.add(sigma.multiply(k))

    # Use the sigma stretch only when the region contains valid pixels
    # and the resulting range is non-degenerate.
    use_sigma = n_valid.gt(0).And(vmax_sigma.neq(vmin_sigma)).And(sigma.gt(0))

    vmin = ee.Number(ee.Algorithms.If(use_sigma, vmin_sigma, default_min))
    vmax = ee.Number(ee.Algorithms.If(use_sigma, vmax_sigma, default_max))

    # ---------------------------------------------------------------------
    # Step 4: Optionally clamp the stretch by percentiles
    # ---------------------------------------------------------------------
    # Percentile clamping prevents the sigma stretch from being dominated
    # by extreme values in the tails of the distribution.
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
    # Convert server-side EE numbers to client-side Python floats so the
    # result can be passed directly to Map.addLayer().
    params = {
        "bands": [band],
        "min": float(vmin.getInfo()),
        "max": float(vmax.getInfo()),
    }
    if palette:
        params["palette"] = list(palette)

    return params


def plot_raster(
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

    The procedure consists of the following steps:

    Step 0
        Validate input parameters.

    Step 1
        Load raster data and handle NoData values.

    Step 2
        Define the valid computational domain.

    Step 3
        Compute percentile-based contrast stretch.

    Step 4
        Handle degenerate percentile ranges.

    Step 5
        Render the raster using matplotlib.

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
    # ---------------------------------------------------------------------
    # Step 0: Validate input parameters
    # ---------------------------------------------------------------------
    # Ensure percentile range is valid and ordered correctly.
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100.")

    # ---------------------------------------------------------------------
    # Step 1: Load raster and handle NoData values
    # ---------------------------------------------------------------------
    # Read the first band and convert to float for NaN support.
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

        # Replace NoData values with NaN for consistent processing.
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

    # ---------------------------------------------------------------------
    # Step 2: Define the valid computational domain
    # ---------------------------------------------------------------------
    # Extract only finite values for statistics.
    valid = arr[np.isfinite(arr)]

    # Ensure the raster contains valid data.
    if valid.size == 0:
        raise ValueError("Raster contains no valid finite values.")

    # ---------------------------------------------------------------------
    # Step 3: Compute percentile-based contrast stretch
    # ---------------------------------------------------------------------
    # Derive visualization range from the selected percentiles.
    vmin = float(np.nanpercentile(valid, p_low))
    vmax = float(np.nanpercentile(valid, p_high))

    # ---------------------------------------------------------------------
    # Step 4: Handle degenerate percentile ranges
    # ---------------------------------------------------------------------
    # If percentiles collapse or are invalid, fall back to full data range.
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))

    # ---------------------------------------------------------------------
    # Step 5: Render raster
    # ---------------------------------------------------------------------
    # Display the raster using the computed visualization range.
    plt.figure(figsize=(8, 6))
    image = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="RdYlBu")

    # Add colourbar with label.
    colorbar = plt.colorbar(image)
    colorbar.set_label(label)

    # Use filename as default title if not provided.
    if title is None:
        title = os.path.basename(tif_path)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
