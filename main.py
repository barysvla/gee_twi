from __future__ import annotations

from typing import Any

import ee
import numpy as np

from scripts.fill_depressions import fill_depressions
from scripts.flow_accumulation import flow_acc
from scripts.flow_direction_d8 import flow_dir_d8
from scripts.flow_direction_mfd import flow_dir_mfd_quinn_1991
from scripts.geotiff_io import clip_tif, read_tif, save_tif
from scripts.grid_io import ee_to_tif, export_dem_grid
from scripts.numpy_to_ee import np_to_ee
from scripts.resolve_flats import resolve_flats_barnes_2014
from scripts.twi import twi_ee, twi_np
from scripts.visualization import plot_raster, show_map, vis_sigma


def _select_dem_src(dem_source: str):
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
    """Compute flow direction and accumulation products for the selected routing method."""
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
        acc_cells = flow_acc(
            flow_weights=dir_out,
            nodata_mask=nodata_mask,
            out="cells",
        )
        print("Flow accumulation computed.")

        return {
            "dir": dir_out,
            "acc_km2": acc_km2,
            "acc_cells": acc_cells,
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
        acc_cells = flow_acc(
            dir_idx=dir_out,
            nodata_mask=nodata_mask,
            out="cells",
        )
        print("Flow accumulation computed.")

        return {
            "dir": dir_out,
            "acc_km2": acc_km2,
            "acc_cells": acc_cells,
        }

    raise ValueError(f"Unsupported flow_method: {flow_method}")


def _build_cloud_map(
    *,
    geom: ee.Geometry,
    scale: ee.Number,
    slope_ee: ee.Image,
    acc_km2_ee: ee.Image,
    acc_cells_ee: ee.Image,
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
    vis_acc_cells = vis_sigma(
        acc_cells_ee,
        "flow_accumulation_cells",
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
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594",
        ],
    )

    map_obj = show_map(
        [
            (slope_ee, vis_slope, "Slope (°)"),
            (acc_cells_ee, vis_acc_cells, "Flow accumulation (cells)"),
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
    acc_cells,
    ref_layers: dict[str, ee.Image],
) -> dict[str, Any]:
    """Run the cloud-mode branch and return Earth Engine outputs."""
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

    acc_cells_res = np_to_ee(
        acc_cells,
        transform=transform,
        crs=crs,
        nodata_mask=nodata_mask,
        bucket_name=f"{project_id}-ee-uploads",
        project_id=project_id,
        band_name="flow_accumulation_cells",
        tmp_dir=grid.get("tmp_dir", None),
        object_prefix="twi_uploads",
        nodata_value=-9999.0,
    )
    acc_cells_ee_full = acc_cells_res["image"]
    acc_cells_ee = acc_cells_ee_full.clip(geom)

    twi = twi_ee(acc_km2_ee, slope_ee).clip(geom)
    print("TWI computed.")

    map_obj = _build_cloud_map(
        geom=geom,
        scale=scale,
        slope_ee=slope_ee,
        acc_km2_ee=acc_km2_ee,
        acc_cells_ee=acc_cells_ee,
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
        "flow_accumulation_cells": acc_cells_ee,
        "flow_accumulation_cells_full": acc_cells_ee_full,
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
    transform,
    crs,
    nodata_mask,
    slope_grid_ee: ee.Image,
    acc_km2,
    acc_cells,
    ref_layers: dict[str, ee.Image],
) -> dict[str, Any]:
    """Run the local-mode branch and return file-based outputs."""
    slope_tif = ee_to_tif(
        slope_grid_ee,
        out_path="slope.tif",
        grid=grid,
        unmask_value=None,
        quiet=True,
    )

    slope_np = read_tif(
        slope_tif,
        nodata_mask=grid["nodata_mask"],
    )

    twi_arr = twi_np(
        acc_np=acc_km2,
        slope_deg_np=slope_np,
        min_slope_deg=0.1,
        nodata_mask=nodata_mask,
        out_dtype="float32",
    )
    print("TWI computed.")

    dem_tif = grid["paths"]["dem_elevations"]

    acc_km2_tif = save_tif(
        acc_km2,
        transform,
        crs,
        nodata_mask,
        filename="flow_accumulation_km2.tif",
        band_name="Flow accumulation (km2)",
    )
    acc_cells_tif = save_tif(
        acc_cells,
        transform,
        crs,
        nodata_mask,
        filename="flow_accumulation_cells.tif",
        band_name="Flow accumulation (cells)",
    )
    twi_tif = save_tif(
        twi_arr,
        transform,
        crs,
        nodata_mask,
        filename="twi.tif",
        band_name="TWI",
    )

    geom_wgs84 = geom.getInfo()

    dem_clip_tif = clip_tif(
        dem_tif,
        geom_wgs84,
        "dem_clipped.tif",
        band_name="DEM",
    )
    acc_km2_clip_tif = clip_tif(
        acc_km2_tif,
        geom_wgs84,
        "acc_km2_clipped.tif",
        band_name="Flow accumulation (km2)",
    )
    acc_cells_clip_tif = clip_tif(
        acc_cells_tif,
        geom_wgs84,
        "acc_cells_clipped.tif",
        band_name="Flow accumulation (cells)",
    )
    slope_clip_tif = clip_tif(
        slope_tif,
        geom_wgs84,
        "slope_clipped.tif",
        band_name="Slope",
    )
    twi_clip_tif = clip_tif(
        twi_tif,
        geom_wgs84,
        "twi_clipped.tif",
        band_name="TWI",
    )

    merit_upa_tif = ee_to_tif(
        ref_layers["merit_upa_grid"],
        out_path="merit_upa.tif",
        grid=grid,
        quiet=True,
    )
    cti_tif = ee_to_tif(
        ref_layers["cti_grid"],
        out_path="cti.tif",
        grid=grid,
        quiet=True,
    )

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
        "dem": dem_clip_tif,
        "slope": slope_clip_tif,
        "flow_accumulation_km2": acc_km2_clip_tif,
        "flow_accumulation_cells": acc_cells_clip_tif,
        "MERIT_flow_accumulation_upa": merit_upa_tif,
        "twi": twi_clip_tif,
        "cti_Hydrography90m": cti_tif,
        "transform": transform,
        "crs": crs,
        "nodata_mask": nodata_mask,
    }


def run_pipeline(
    project_id: str | None = None,
    geometry: ee.Geometry | None = None,
    accum_geometry: ee.Geometry | None = None,
    dem_source: str = "FABDEM",
    flow_method: str = "mfd_quinn_1991",
    use_bucket: bool = False,
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
        Run either the cloud-mode or local-mode output branch.

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

    dem_res_np, flat_mask, labels, flowdirs, stats = resolve_flats_barnes_2014(
        dem_fill_np,
        nodata=np.nan,
        equal_tol=0.0,
        lower_tol=0.0,
        treat_oob_as_lower=True,
        apply_to_dem="epsilon",
        epsilon=1e-5,
    )
    print("Flat resolution completed.")

    _ = flat_mask, labels, flowdirs, stats

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
    acc_cells = flow_res["acc_cells"]

    # ---------------------------------------------------------------------
    # Step 5: Compute slope on the aligned Earth Engine grid
    # ---------------------------------------------------------------------
    slope_grid_ee = ee.Terrain.slope(dem_ee).toFloat().rename("Slope")
    slope_ee = slope_grid_ee.clip(geom)
    print("Slope computed.")

    # ---------------------------------------------------------------------
    # Step 6: Run the selected output branch
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
            acc_cells=acc_cells,
            ref_layers=ref_layers,
        )

    return _run_local(
        geom=geom,
        grid=grid,
        transform=transform,
        crs=crs,
        nodata_mask=nodata_mask,
        slope_grid_ee=slope_grid_ee,
        acc_km2=acc_km2,
        acc_cells=acc_cells,
        ref_layers=ref_layers,
    )


if __name__ == "__main__":
    _ = run_pipeline()
