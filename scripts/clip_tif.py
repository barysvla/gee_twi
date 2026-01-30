import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
import numpy as np
import os

def clip_tif_by_geojson(input_tif: str,
                        geojson_geom: dict,
                        output_tif: str,
                        band_name: str = None) -> str:
    """
    Clip input_tif by geojson_geom. Outside geometry → NaN.
    """
    with rasterio.open(input_tif) as src:
        src_crs = src.crs
        src_meta = src.meta.copy()

        # Reproject geojson geometry into raster CRS
        geom_in_r = transform_geom(
            src_crs="EPSG:4326",
            dst_crs=src_crs.to_string(),
            geom=geojson_geom,
            precision=6
        )

        out_image, out_transform = mask(
            src,
            [geom_in_r],
            crop=True,
            all_touched=True,
            nodata=np.nan,
            filled=True
        )

    # Convert to float32 if not already
    out_image = out_image.astype(np.float32)

    # Also, if the input had a nodata value defined (e.g. -9999), replace those
    nod_val = src_meta.get("nodata")
    if nod_val is not None:
        out_image[out_image == nod_val] = np.nan

    # Update metadata
    out_meta = src_meta
    out_meta.update({
        "dtype": "float32",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": np.nan
    })

    with rasterio.open(output_tif, "w", **out_meta) as dst:
        dst.write(out_image)
        if band_name:
            dst.set_band_description(1, band_name)
    return output_tif
