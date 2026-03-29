from __future__ import annotations

"""
Main entry point of the TWI workflow.

This script coordinates the complete processing pipeline from DEM selection
and grid export to hydrological conditioning, flow routing, flow accumulation,
slope computation, and final TWI generation. It also prepares reference layers
used for comparison and provides two output branches: a local mode producing
GeoTIFF files and a cloud mode returning Earth Engine layers and an interactive map.

The main function is `run_pipeline`, while the remaining helper functions handle
DEM selection, reference-layer preparation, flow computation, map assembly,
and mode-specific output processing.
"""

from typing import Any
from datetime import datetime

import os

import ee
import numpy as np

from scripts.fill_depressions import fill_depressions
from scripts.flow_accumulation import flow_acc
from scripts.flow_direction_d8 import flow_dir_d8
from scripts.flow_direction_mfd import flow_dir_mfd_quinn_1991
from scripts.geotiff_io import clip_tif, read_tif, save_tif
from scripts.grid_io import export_dem_grid, ee_to_tif
from scripts.numpy_to_ee import np_to_ee
from scripts.resolve_flats import resolve_flats_barnes_2014
from scripts.twi import twi_ee, twi_np
from scripts.visualization import plot_raster, show_map, vis_sigma


def _select_dem_src(
    dem_source: str,
) -> ee.Image | ee.ImageCollection:
    """Return the requested DEM source as an Earth Engine image or image collection."""
    if dem_source == "FABDEM":
        return ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    if dem_source == "GLO30":
        return ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    if dem_source == "AW3D30":
        return ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select("DSM")
    if dem_source == "SRTMGL1_003":
        return ee.Image("USGS/SRTMGL1_003").select("elevation")
    if dem_source == "NASADEM_HGT":
        return ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    if dem_source == "ASTER_GDEM":
        return ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").select("b1")
    if dem_source == "CGIAR_SRTM90":
        return ee.Image("CGIAR/SRTM90_V4").select("elevation")
    if dem_source == "MERIT_DEM":
        return ee.Image("MERIT/DEM/v1_0_3").select("dem")
    if dem_source == "MERIT_Hydro":
        return ee.Image("MERIT/Hydro/v1_0_1").select("elv")

    raise ValueError(f"Unsupported dem_source: {dem_source}")


def _build_ref_layers(
    dem_ee: ee.Image,
    geom: ee.Geometry,
    scale: ee.Number,
) -> dict[str, ee.Image]:
    """Build reference layers aligned to the pipeline grid."""
    merit_upa_grid = (
        ee.Image("MERIT/Hydro/v1_0_1")
        .select("upa")
        .reproject(dem_ee.projection())
        .rename("MERIT_flow_accumulation_upa")
    )
    merit_upa = merit_upa_grid.clip(geom)

    cti_grid = (
        ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti")
        .mosaic()
        .toFloat()
        .divide(ee.Number(1e8))
        .translate(0, scale.multiply(-1))
        .reproject(dem_ee.projection())
        .rename("CTI")
    )
    cti = cti_grid.clip(geom)

    return {
        "merit_upa_grid": merit_upa_grid,
        "merit_upa": merit_upa,
        "cti_grid": cti_grid,
        "cti": cti,
    }


def _compute_flow(
    dem_np,
    transform,
    nodata_mask,
    px_area_np,
    flow_method: str,
) -> dict[str, Any]:
    """
    Compute flow direction and flow accumulation for the selected routing method.

    Depending on the selected option, the function runs either D8 or MFD
    flow routing and then derives flow accumulation in square kilometres.
    """
    if flow_method == "mfd_quinn_1991":
        dir_out = flow_dir_mfd_quinn_1991(
            dem_np,
            transform,
            nodata_mask=nodata_mask,
        )
        print("Flow direction computed.")

        acc_km2 = flow_acc(
            flow_weights=dir_out,
            nodata_mask=nodata_mask,
            pixel_area_m2=px_area_np,
            out="km2",
        )
        print("Flow accumulation computed.")

        return {
            "dir": dir_out,
            "acc_km2": acc_km2,
        }

    if flow_method == "d8":
        dir_out = flow_dir_d8(
            dem_np,
            transform,
            nodata_mask=nodata_mask,
        )
        print("Flow direction computed.")

        acc_km2 = flow_acc(
            dir_idx=dir_out,
            nodata_mask=nodata_mask,
            pixel_area_m2=px_area_np,
            out="km2",
        )
        print("Flow accumulation computed.")

        return {
            "dir": dir_out,
            "acc_km2": acc_km2,
        }

    raise ValueError(f"Unsupported flow_method: {flow_method}")


def _build_cloud_map(
    *,
    geom: ee.Geometry,
    scale: ee.Number,
    slope_ee: ee.Image,
    acc_km2_ee: ee.Image,
    merit_upa: ee.Image,
    cti: ee.Image,
    twi: ee.Image,
):
    """Create an interactive map for cloud-mode outputs."""
    vis_twi = vis_sigma(
        twi,
        "TWI",
        geom,
        scale,
        k=2.0,
        palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"],
    )
    vis_cti = vis_sigma(
        cti,
        "CTI",
        geom,
        scale,
        k=2.0,
        palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"],
    )
    vis_acc_km2 = vis_sigma(
        acc_km2_ee,
        "flow_accumulation_km2",
        geom,
        scale,
        k=2.0,
        palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"],
    )
    vis_merit = vis_sigma(
        merit_upa,
        "MERIT_flow_accumulation_upa",
        geom,
        scale,
        k=2.0,
        palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"],
    )
    vis_slope = vis_sigma(
        slope_ee,
        "Slope",
        geom,
        scale,
        k=2.0,
        palette=[
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
            "#6baed6", "#4292c6", "#2171b5", "#084594",
        ],
    )

    map_obj = show_map(
        [
            (slope_ee, vis_slope, "Slope (°)"),
            (merit_upa, vis_merit, "(reference) Flow accumulation - MERIT (km²)"),
            (acc_km2_ee, vis_acc_km2, "Flow accumulation (km²)"),
            (cti, vis_cti, "(reference) CTI - Hydrography90m"),
            (twi, vis_twi, "TWI"),
        ]
    )
    map_obj.centerObject(geom, 12)

    return map_obj


def _run_cloud(
    *,
    project_id: str,
    geom: ee.Geometry,
    geom_acc: ee.Geometry,
    grid: dict[str, Any],
    transform,
    crs,
    nodata_mask,
    dem_ee: ee.Image,
    slope_ee: ee.Image,
    scale: ee.Number,
    acc_km2,
    ref_layers: dict[str, ee.Image],
) -> dict[str, Any]:
    """
    Run the cloud output branch of the workflow.

    This function uploads the locally computed flow-accumulation raster
    to Earth Engine, computes TWI in cloud mode, and prepares the final
    interactive map output.

    The procedure consists of the following steps:

    Step 0
        Upload flow accumulation raster to Earth Engine.

    Step 1
        Compute and clip the Topographic Wetness Index (TWI).

    Step 2
        Build the interactive map and return the final outputs.

    Returns
    -------
    dict
        Dictionary containing Earth Engine outputs and map object.
    """
    # ---------------------------------------------------------------------
    # Step 0: Upload flow accumulation raster to Earth Engine
    # ---------------------------------------------------------------------
    acc_km2_res = np_to_ee(
        acc_km2,
        transform=transform,
        crs=crs,
        nodata_mask=nodata_mask,
        bucket_name=f"{project_id}-ee-uploads",
        project_id=project_id,
        band_name="flow_accumulation_km2",
        tmp_dir=grid.get("tmp_dir", None),
        object_prefix="twi_uploads",
        nodata_value=-9999.0,
    )
    acc_km2_ee_full = acc_km2_res["image"]
    acc_km2_ee = acc_km2_ee_full.clip(geom)

    # ---------------------------------------------------------------------
    # Step 1: Compute and clip TWI
    # ---------------------------------------------------------------------
    twi = twi_ee(acc_km2_ee, slope_ee).clip(geom)
    print("TWI computed.")

    # ---------------------------------------------------------------------
    # Step 2: Build map and return final outputs
    # ---------------------------------------------------------------------
    map_obj = _build_cloud_map(
        geom=geom,
        scale=scale,
        slope_ee=slope_ee,
        acc_km2_ee=acc_km2_ee,
        merit_upa=ref_layers["merit_upa"],
        cti=ref_layers["cti"],
        twi=twi,
    )

    return {
        "dem": dem_ee.clip(geom),
        "mode": "cloud",
        "slope": slope_ee,
        "flow_accumulation_km2": acc_km2_ee,
        "flow_accumulation_km2_full": acc_km2_ee_full,
        "MERIT_flow_accumulation_upa": ref_layers["merit_upa"],
        "twi": twi,
        "cti_Hydrography90m": ref_layers["cti"],
        "geometry": geom,
        "geometry_accum": geom_acc,
        "scale": scale,
        "map": map_obj,
    }


def _run_local(
    *,
    geom: ee.Geometry,
    grid: dict[str, Any],
    output_dir: str,
    transform,
    crs,
    nodata_mask,
    slope_grid_ee: ee.Image,
    acc_km2,
    ref_layers: dict[str, ee.Image],
) -> dict[str, Any]:
    """
    Run the local output branch of the workflow.

    This function performs local post-processing of the workflow outputs,
    including export of slope, computation of TWI, raster saving, clipping,
    and visualization.

    The procedure consists of the following steps:

    Step 0
        Prepare output directory structure for the current run.

    Step 1
        Export slope raster from Earth Engine and load it as a NumPy array.

    Step 2
        Compute the Topographic Wetness Index (TWI) from local arrays.

    Step 3
        Save flow accumulation and TWI rasters as GeoTIFF files.

    Step 4
        Clip all rasters to the target geometry.

    Step 5
        Export reference rasters (MERIT UPA and CTI).

    Step 6
        Visualize the final TWI raster.

    Returns
    -------
    dict
        Dictionary containing paths to local output rasters and metadata.
    """
    # ---------------------------------------------------------------------
    # Step 0: Prepare local output paths
    # ---------------------------------------------------------------------
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    slope_tif_path = os.path.join(out_dir, "slope.tif")
    acc_km2_tif_path = os.path.join(out_dir, "flow_accumulation_km2.tif")
    twi_tif_path = os.path.join(out_dir, "twi.tif")

    dem_clip_tif_path = os.path.join(out_dir, "dem_clipped.tif")
    acc_km2_clip_tif_path = os.path.join(out_dir, "acc_km2_clipped.tif")
    slope_clip_tif_path = os.path.join(out_dir, "slope_clipped.tif")
    twi_clip_tif_path = os.path.join(out_dir, "twi_clipped.tif")

    merit_upa_tif_path = os.path.join(out_dir, "merit_upa.tif")
    cti_tif_path = os.path.join(out_dir, "cti.tif")

    # ---------------------------------------------------------------------
    # Step 1: Export slope raster from Earth Engine and read it locally
    # ---------------------------------------------------------------------
    slope_export = ee_to_tif(
        slope_grid_ee,
        out_path=slope_tif_path,
        grid=grid,
        unmask_value=None,
        quiet=True,
        tmp_dir=os.path.join(tiles_dir, "slope"),
        tile_prefix="slope_tile",
    )
    slope_tif = slope_export["out_path"]

    slope_np = read_tif(
        slope_tif,
        nodata_mask=grid["nodata_mask"],
    )

    # ---------------------------------------------------------------------
    # Step 2: Compute TWI locally
    # ---------------------------------------------------------------------
    twi_arr = twi_np(
        acc_np=acc_km2,
        slope_deg_np=slope_np,
        min_slope_deg=0.1,
        nodata_mask=nodata_mask,
        out_dtype="float32",
    )
    print("TWI computed.")

    # ---------------------------------------------------------------------
    # Step 3: Save local rasters
    # ---------------------------------------------------------------------
    dem_tif = grid["paths"]["dem_elevations"]

    acc_km2_tif = save_tif(
        acc_km2,
        transform,
        crs,
        nodata_mask,
        filename=acc_km2_tif_path,
        band_name="Flow accumulation (km2)",
    )
    twi_tif = save_tif(
        twi_arr,
        transform,
        crs,
        nodata_mask,
        filename=twi_tif_path,
        band_name="TWI",
    )

    # ---------------------------------------------------------------------
    # Step 4: Clip rasters to the target geometry
    # ---------------------------------------------------------------------
    geom_wgs84 = geom.getInfo()

    dem_clip_tif = clip_tif(
        dem_tif,
        geom_wgs84,
        dem_clip_tif_path,
        band_name="DEM",
    )
    acc_km2_clip_tif = clip_tif(
        acc_km2_tif,
        geom_wgs84,
        acc_km2_clip_tif_path,
        band_name="Flow accumulation (km2)",
    )
    slope_clip_tif = clip_tif(
        slope_tif,
        geom_wgs84,
        slope_clip_tif_path,
        band_name="Slope",
    )
    twi_clip_tif = clip_tif(
        twi_tif,
        geom_wgs84,
        twi_clip_tif_path,
        band_name="TWI",
    )

    # ---------------------------------------------------------------------
    # Step 5: Export reference rasters
    # ---------------------------------------------------------------------
    merit_upa_export = ee_to_tif(
        img=ref_layers["merit_upa_grid"],
        out_path=merit_upa_tif_path,
        grid=grid,
        quiet=True,
        tmp_dir=os.path.join(tiles_dir, "merit"),
        tile_prefix="merit_tile",
    )
    merit_upa_tif = merit_upa_export["out_path"]

    cti_export = ee_to_tif(
        img=ref_layers["cti_grid"],
        out_path=cti_tif_path,
        grid=grid,
        quiet=True,
        tmp_dir=os.path.join(tiles_dir, "cti"),
        tile_prefix="cti_tile",
    )
    cti_tif = cti_export["out_path"]

    # ---------------------------------------------------------------------
    # Step 6: Plot final TWI
    # ---------------------------------------------------------------------
    print("Plotting TWI (local mode, percentile stretch).")
    plot_raster(
        twi_clip_tif,
        p_low=2.0,
        p_high=98.0,
        label="TWI",
        title="Topographic Wetness Index",
    )

    return {
        "mode": "local",
        "output_dir": out_dir,
        "dem": dem_clip_tif,
        "slope": slope_clip_tif,
        "flow_accumulation_km2": acc_km2_clip_tif,
        "MERIT_flow_accumulation_upa": merit_upa_tif,
        "twi": twi_clip_tif,
        "cti_Hydrography90m": cti_tif,
        "transform": transform,
        "crs": crs,
        "nodata_mask": nodata_mask,
        "export_info": {
            "slope": slope_export,
            "MERIT_flow_accumulation_upa": merit_upa_export,
            "cti_Hydrography90m": cti_export,
        },
    }


def run_pipeline(
    project_id: str | None = None,
    geometry: ee.Geometry | None = None,
    accum_geometry: ee.Geometry | None = None,
    dem_source: str = "FABDEM",
    flow_method: str = "mfd_quinn_1991",
    use_bucket: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run the complete TWI workflow for the selected DEM source and routing method.

    The pipeline performs DEM export, hydrological conditioning, flow
    direction and accumulation computation, slope derivation, and final
    TWI generation. Accumulation may be computed on a buffered geometry,
    while final outputs are clipped to the original geometry.

    The procedure consists of the following steps:

    Step 0
        Initialize Earth Engine and validate the input geometries.

    Step 1
        Select the DEM source and export the DEM grid and pixel-area grid
        over the accumulation geometry.

    Step 2
        Build grid-aligned reference layers used for validation and
        visual comparison.

    Step 3
        Perform hydrological conditioning on the DEM in local arrays.

    Step 4
        Compute flow direction and flow accumulation.

    Step 5
        Compute slope on the Earth Engine grid.

    Step 6
        Compute the Topographic Wetness Index (TWI) and produce final outputs
        using either the cloud-mode (Earth Engine) or local-mode (NumPy) branch.

    Parameters
    ----------
    project_id : str, optional
        Earth Engine and Google Cloud project identifier.
    geometry : ee.Geometry, optional
        Original unbuffered region of interest used for final clipping.
    accum_geometry : ee.Geometry, optional
        Optional buffered region used for accumulation. If None, the
        original geometry is used.
    dem_source : str, default="FABDEM"
        DEM source identifier.
    flow_method : {"mfd_quinn_1991", "d8"}, default="mfd_quinn_1991"
        Flow-routing method.
    use_bucket : bool, default=False
        If True, upload local rasters to Cloud Storage and continue in
        Earth Engine. If False, keep outputs as local GeoTIFF files.
    output_dir : str, optional
        Output directory for local results. If None, a timestamped
        directory is created in `outputs/`.

    Returns
    -------
    dict
        Dictionary containing pipeline outputs for either cloud mode or
        local mode.
    """
    # ---------------------------------------------------------------------
    # Step 0: Initialize Earth Engine and validate inputs
    # ---------------------------------------------------------------------
    ee.Initialize(project=project_id)

    if geometry is None:
        raise ValueError("Missing required parameter: geometry")

    geom = geometry
    geom_acc = accum_geometry if accum_geometry is not None else geom

    if output_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", run_name)

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Step 1: Select DEM source and export aligned input grids
    # ---------------------------------------------------------------------
    dem_src = _select_dem_src(dem_source)

    grid = export_dem_grid(
        src=dem_src,
        region_geom=geom_acc,
        band=None,
        resample_method="bilinear",
        nodata_value=-9999.0,
        snap_region_to_grid=True,
    )

    dem_np = grid["dem_elevations_np"]
    px_area_np = grid["pixel_area_m2_np"]
    transform = grid["transform"]
    nodata_mask = grid["nodata_mask"]
    crs = grid["crs"]
    dem_ee = grid["ee_dem_grid"]
    scale = ee.Number(dem_ee.projection().nominalScale())

    # ---------------------------------------------------------------------
    # Step 2: Build grid-aligned reference layers
    # ---------------------------------------------------------------------
    ref_layers = _build_ref_layers(
        dem_ee=dem_ee,
        geom=geom,
        scale=scale,
    )

    # ---------------------------------------------------------------------
    # Step 3: Perform hydrological conditioning
    # ---------------------------------------------------------------------
    dem_fill_np = fill_depressions(
        dem_np,
        seed_internal_nodata_as_outlet=True,
        return_fill_depth=False,
    )
    print("Depression filling completed.")

    dem_res_np, _, _, _, _ = resolve_flats_barnes_2014(
        dem_fill_np,
        nodata=np.nan,
        equal_tol=0.0,
        lower_tol=0.0,
        treat_oob_as_lower=True,
        apply_to_dem="epsilon",
        epsilon=1e-5,
    )
    print("Flat resolution completed.")

    # ---------------------------------------------------------------------
    # Step 4: Compute flow direction and flow accumulation
    # ---------------------------------------------------------------------
    flow_res = _compute_flow(
        dem_np=dem_res_np,
        transform=transform,
        nodata_mask=nodata_mask,
        px_area_np=px_area_np,
        flow_method=flow_method,
    )
    acc_km2 = flow_res["acc_km2"]

    # ---------------------------------------------------------------------
    # Step 5: Compute slope on the aligned Earth Engine grid
    # ---------------------------------------------------------------------
    slope_grid_ee = ee.Terrain.slope(dem_ee).toFloat().rename("Slope")
    slope_ee = slope_grid_ee.clip(geom)
    print("Slope computed.")

    # ---------------------------------------------------------------------
    # Step 6: Compute TWI and generate outputs (cloud or local mode)
    # ---------------------------------------------------------------------
    if use_bucket:
        return _run_cloud(
            project_id=project_id,
            geom=geom,
            geom_acc=geom_acc,
            grid=grid,
            transform=transform,
            crs=crs,
            nodata_mask=nodata_mask,
            dem_ee=dem_ee,
            slope_ee=slope_ee,
            scale=scale,
            acc_km2=acc_km2,
            ref_layers=ref_layers,
        )

    return _run_local(
        geom=geom,
        grid=grid,
        output_dir=output_dir,
        transform=transform,
        crs=crs,
        nodata_mask=nodata_mask,
        slope_grid_ee=slope_grid_ee,
        acc_km2=acc_km2,
        ref_layers=ref_layers,
    )


if __name__ == "__main__":
    _ = run_pipeline()
