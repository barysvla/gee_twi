import geemap
import ee
from IPython.display import display
import numpy as np
import rasterio
#import leafmap.leafmap as leafmap
import matplotlib.pyplot as plt

def visualize_map(layers):
    """
    Vytvoření interaktivní mapy.
    :param layers: Seznam tuple vrstvy a jejich vizualizačních parametrů.
    """
    Map = geemap.Map(basemap="Esri.WorldImagery")
    Map.add_basemap("Esri.WorldTopoMap")
    
    for layer in layers:
        image, vis_params, name = layer
        if isinstance(image, ee.Image):  # Ověření, že je image správný typ
            Map.addLayer(image, vis_params, name)
        else:
            print(f"⚠ Varování: Vrstva '{name}' není typu ee.Image a nebude zobrazena!")
    
    return Map
    
# def visualize_map_leaf(layers, center=None, zoom=10, basemaps=None):
#     """
#     Create an interactive map using leafmap, similar to geemap version.
    
#     :param layers: list of tuples (img_or_path, vis_params, name)
#                    img_or_path can be ee.Image or a path to a raster file
#     :param center: optional (lat, lon) to center map initially
#     :param zoom: initial zoom level
#     :param basemaps: optional list of basemap names (strings) to add
#     :return: leafmap.Map instance
#     """
#     # Create base map
#     if center is None:
#         m = leafmap.Map()
#     else:
#         m = leafmap.Map(center=center, zoom=zoom)
    
#     # Add extra basemaps if given
#     if basemaps:
#         for bm in basemaps:
#             m.add_basemap(bm)
    
#     # Add each layer
#     for img_or_path, vis_params, name in layers:
#         if isinstance(img_or_path, ee.Image):
#             # Add EE layer
#             m.add_ee_layer(img_or_path, vis_params, name)
#         else:
#             # Treat as local raster
#             cmap = vis_params.get("palette") or vis_params.get("colormap")
#             vmin = vis_params.get("min", None)
#             vmax = vis_params.get("max", None)
#             nodata = vis_params.get("nodata", None)
#             m.add_raster(
#                 img_or_path,
#                 layer_name=name,
#                 colormap=cmap,
#                 vmin=vmin,
#                 vmax=vmax,
#                 nodata=nodata
#             )
    
#     # Add inspector / click tool if available
#     # try:
#     #     m.add_inspector_gui()
#     # except Exception:
#     #     pass
    
#     return m
    
def vis_2sigma(image, band, region, scale, k=2.0, palette=None, clamp_to_pct=None):
    """
    Build visualization params for Map.addLayer() using a μ ± k·σ stretch.

    Parameters
    ----------
    image : ee.Image
        Source image.
    band : str
        Band name to visualize.
    region : ee.Geometry
        Region for statistics (masked pixels are ignored).
    scale : float
        Pixel scale (meters) for reduceRegion.
    k : float, optional
        Sigma multiplier; default 2.0.
    palette : list[str] | None
        Optional color palette for visualization.
    clamp_to_pct : tuple[int, int] | None
        Optional percentile clamp (e.g., (2, 98)) to make the stretch more robust.

    Returns
    -------
    dict
        Dictionary with 'bands', 'min', 'max' (and 'palette' if provided) for Map.addLayer().
    """
    img = image.select([band])

    # Mean and standard deviation over the region (masked pixels are ignored)
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
    )
    mu  = ee.Number(stats.get(f"{band}_mean"))
    sig = ee.Number(stats.get(f"{band}_stdDev"))

    vmin = mu.subtract(sig.multiply(k))
    vmax = mu.add(sig.multiply(k))

    # Optionally clamp by percentiles for a more robust stretch
    if clamp_to_pct is not None:
        lo, hi = clamp_to_pct
        p = img.reduceRegion(
            reducer=ee.Reducer.percentile([lo, hi]),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
        )
        pmin = ee.Number(p.get(f"{band}_p{lo}"))
        pmax = ee.Number(p.get(f"{band}_p{hi}"))
        vmin = vmin.max(pmin)
        vmax = vmax.min(pmax)

    params = {"bands": [band], "min": vmin.getInfo(), "max": vmax.getInfo()}
    if palette:
        params["palette"] = palette
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

# def vis_2sigma_safe(image, band, region, scale, k=2.0, palette=None, clamp_to_pct=None):
#     """
#     Build visParams with min/max = μ ± k·σ.
#     Robust to null stats: falls back to percentiles if mean/stdDev are missing or degenerate.
#     """
#     img = image.select([band])

#     # Count valid pixels to catch empty/masked regions early
#     n_valid = ee.Number(
#         img.reduceRegion(
#             reducer=ee.Reducer.count(),
#             geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
#         ).get(band, 0)
#     )

#     # Compute mean+std; use defaults to avoid nulls
#     stats = ee.Dictionary(
#         img.reduceRegion(
#             reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
#             geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
#         )
#     )
#     mu  = ee.Number(stats.get(f"{band}_mean", 0))
#     sig = ee.Number(stats.get(f"{band}_stdDev", 0))

#     # Proposed min/max from μ ± k·σ
#     vmin_sigma = mu.subtract(sig.multiply(k))
#     vmax_sigma = mu.add(sig.multiply(k))

#     # Fallback percentiles (robust)
#     lo, hi = (2, 98) if clamp_to_pct is None else clamp_to_pct
#     p = ee.Dictionary(
#         img.reduceRegion(
#             reducer=ee.Reducer.percentile([lo, hi]),
#             geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
#         )
#     )
#     pmin = ee.Number(p.get(f"{band}_p{lo}", 0))
#     pmax = ee.Number(p.get(f"{band}_p{hi}", 1))

#     # Use sigma-stretch only if we have data and a non-degenerate stdDev range
#     use_sigma = n_valid.gt(0).And(vmax_sigma.neq(vmin_sigma)).And(sig.gt(0))

#     vmin = ee.Number(ee.Algorithms.If(use_sigma, vmin_sigma, pmin))
#     vmax = ee.Number(ee.Algorithms.If(use_sigma, vmax_sigma, pmax))

#     # Bring to client (numbers, not ee.Number)
#     params = {"bands": [band], "min": vmin.getInfo(), "max": vmax.getInfo()}
#     if palette:
#         params["palette"] = palette
#     return params

# def vis_2sigma_tif(
#     tif_path: str,
#     clamp_to_pct: tuple[int, int] = None,
#     k: float = 2.0,
#     palette: list[str] = None,
# ):
#     """
#     Compute visualization parameters (min, max, palette) for a GeoTIFF using μ ± k·σ stretch.
#     Fallback to percentiles if data are degenerate.
    
#     :param tif_path: str, path to the GeoTIFF file
#     :param clamp_to_pct: tuple (lo_pct, hi_pct), e.g. (2, 98)
#     :param k: float, multiplier for σ (default 2.0)
#     :param palette: list of hex colors (optional)
#     :return: dict with keys: "min", "max", optionally "palette"
#     """
#     with rasterio.open(tif_path) as src:
#         arr = src.read(1).astype(float)
#         nod = src.nodata
#         if nod is not None:
#             arr = np.where(arr == nod, np.nan, arr)
    
#     # Flatten valid values (finite)
#     valid = arr[np.isfinite(arr)]
#     if valid.size == 0:
#         # No valid data
#         vmin = 0.0
#         vmax = 1.0
#     else:
#         mu = np.nanmean(valid)
#         sigma = np.nanstd(valid)
#         vmin_sigma = mu - k * sigma
#         vmax_sigma = mu + k * sigma
        
#         # Fallback percentiles
#         if clamp_to_pct is not None:
#             lo, hi = clamp_to_pct
#         else:
#             lo, hi = 2, 98

#         lo_val = np.nanpercentile(valid, lo)
#         hi_val = np.nanpercentile(valid, hi)

#         # Clamp the sigma-derived bounds
#         vmin = max(vmin_sigma, lo_val)
#         vmax = min(vmax_sigma, hi_val)
        
#         # If degenerate (vmin >= vmax), fallback to percentile extremes
#         if vmin >= vmax:
#             vmin = lo_val
#             vmax = hi_val
    
#     params = {"min": float(vmin), "max": float(vmax)}
#     if palette is not None:
#         params["palette"] = palette
#     return params
