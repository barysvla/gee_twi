from __future__ import annotations

from typing import Any
from datetime import datetime
import time
import os
import gc

import ee
import numpy as np

from scripts.fill_depressions import fill_depressions
from scripts.flow_accumulation import flow_acc
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
):
    timings = {}

    if flow_method == "mfd_quinn_1991":
        t0 = time.perf_counter()
        flow_weights = flow_dir_mfd_quinn_1991(
            dem_np,
            transform,
            nodata_mask=nodata_mask,
        )
        dt_dir = time.perf_counter() - t0
        timings["flow_direction_s"] = dt_dir
        print(f"Flow direction computed. ({dt_dir:.2f} s)")

        t0 = time.perf_counter()
        acc_km2 = flow_acc(
            flow_weights=flow_weights,
            nodata_mask=nodata_mask,
            pixel_area_m2=px_area_np,
            out="km2",
        )
        dt_acc = time.perf_counter() - t0
        timings["flow_accumulation_s"] = dt_acc
        print(f"Flow accumulation computed. ({dt_acc:.2f} s)")

        del flow_weights
        gc.collect()

        return {
            "acc_km2": acc_km2,
            "timings": timings,
        }

    if flow_method == "d8":
        t0 = time.perf_counter()
        dir_out = flow_dir_d8(
            dem_np,
            transform,
            nodata_mask=nodata_mask,
        )
        dt_dir = time.perf_counter() - t0
        timings["flow_direction_s"] = dt_dir
        print(f"Flow direction computed. ({dt_dir:.2f} s)")

        t0 = time.perf_counter()
        acc_km2 = flow_acc(
            dir_idx=dir_out,
            nodata_mask=nodata_mask,
            pixel_area_m2=px_area_np,
            out="km2",
        )
        dt_acc = time.perf_counter() - t0
        timings["flow_accumulation_s"] = dt_acc
        print(f"Flow accumulation computed. ({dt_acc:.2f} s)")

        del dir_out
        gc.collect()

        return {
            "acc_km2": acc_km2,
            "timings": timings,
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
    cell_count_total,
    cell_count_valid,
    cell_count_nodata,
    timings,
) -> dict[str, Any]:
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

    t0 = time.perf_counter()
    twi = twi_ee(acc_km2_ee, slope_ee).clip(geom)
    dt = time.perf_counter() - t0
    timings["twi_s"] = dt
    print(f"TWI computed. ({dt:.2f} s)")

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
        "cell_count_total": cell_count_total,
        "cell_count_valid": cell_count_valid,
        "cell_count_nodata": cell_count_nodata,
        "timings": timings,
    }


def _run_local(
    *,
    geom: ee.Geometry,
    geom_acc: ee.Geometry,
    grid: dict[str, Any],
    output_dir: str,
    transform,
    crs,
    nodata_mask,
    slope_grid_ee: ee.Image,
    acc_km2,
    ref_layers: dict[str, ee.Image],
    cell_count_total,
    cell_count_valid,
    cell_count_nodata,
    timings,
) -> dict[str, Any]:
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

    t0 = time.perf_counter()
    twi_arr = twi_np(
        acc_np=acc_km2,
        slope_deg_np=slope_np,
        min_slope_deg=0.1,
        nodata_mask=nodata_mask,
        out_dtype="float32",
    )
    dt = time.perf_counter() - t0
    timings["twi_s"] = dt
    print(f"TWI computed. ({dt:.2f} s)")

    del slope_np
    gc.collect()

    dem_tif = grid["paths"]["dem_elevations"]

    acc_km2_tif = save_tif(
        acc_km2,
        transform,
        crs,
        nodata_mask,
        filename=acc_km2_tif_path,
        band_name="Flow accumulation (km2)",
    )

    del acc_km2
    gc.collect()

    twi_tif = save_tif(
        twi_arr,
        transform,
        crs,
        nodata_mask,
        filename=twi_tif_path,
        band_name="TWI",
    )

    del twi_arr
    gc.collect()

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
        "geometry": geom,
        "geometry_accum": geom_acc,
        "transform": transform,
        "crs": crs,
        "nodata_mask": nodata_mask,
        "export_info": {
            "slope": slope_export,
            "MERIT_flow_accumulation_upa": merit_upa_export,
            "cti_Hydrography90m": cti_export,
        },
        "cell_count_total": cell_count_total,
        "cell_count_valid": cell_count_valid,
        "cell_count_nodata": cell_count_nodata,
        "timings": timings,
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
    ee.Initialize(project=project_id)

    if geometry is None:
        raise ValueError("Missing required parameter: geometry")

    geom = geometry
    geom_acc = accum_geometry if accum_geometry is not None else geom

    if output_dir is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", run_name)

    os.makedirs(output_dir, exist_ok=True)

    timings = {}

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

    cell_count_total = int(dem_np.size)
    cell_count_valid = int((~nodata_mask).sum())
    cell_count_nodata = int(nodata_mask.sum())

    ref_layers = _build_ref_layers(
        dem_ee=dem_ee,
        geom=geom,
        scale=scale,
    )

    t0 = time.perf_counter()
    dem_fill_np = fill_depressions(
        dem_np,
        seed_internal_nodata_as_outlet=True,
        return_fill_depth=False,
    )
    dt = time.perf_counter() - t0
    timings["depression_filling_s"] = dt
    print(f"Depression filling completed. ({dt:.2f} s)")

    del dem_np
    gc.collect()

   t0 = time.perf_counter()
    dem_res_np, _, _, _ = resolve_flats_barnes_2014(
        dem_fill_np,
        nodata=np.nan,
        equal_tol=0.0,
        lower_tol=0.0,
        treat_oob_as_lower=True,
        apply_to_dem="epsilon",
        epsilon=1e-5,
    )
    dt = time.perf_counter() - t0
    timings["flat_resolution_s"] = dt
    print(f"Flat resolution completed. ({dt:.2f} s)")
    
    del dem_fill_np
    gc.collect()
    
    flow_res = _compute_flow(
        dem_np=dem_res_np,
        transform=transform,
        nodata_mask=nodata_mask,
        px_area_np=px_area_np,
        flow_method=flow_method,
    )
    
    acc_km2 = flow_res["acc_km2"]
    timings.update(flow_res["timings"])
    
    del dem_res_np
    del px_area_np
    gc.collect()

    t0 = time.perf_counter()
    slope_grid_ee = ee.Terrain.slope(dem_ee).toFloat().rename("Slope")
    slope_ee = slope_grid_ee.clip(geom)
    dt = time.perf_counter() - t0
    timings["slope_s"] = dt
    print(f"Slope computed. ({dt:.2f} s)")

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
            cell_count_total=cell_count_total,
            cell_count_valid=cell_count_valid,
            cell_count_nodata=cell_count_nodata,
            timings=timings,
        )

    return _run_local(
        geom=geom,
        geom_acc=geom_acc,
        grid=grid,
        output_dir=output_dir,
        transform=transform,
        crs=crs,
        nodata_mask=nodata_mask,
        slope_grid_ee=slope_grid_ee,
        acc_km2=acc_km2,
        ref_layers=ref_layers,
        cell_count_total=cell_count_total,
        cell_count_valid=cell_count_valid,
        cell_count_nodata=cell_count_nodata,
        timings=timings,
    )


if __name__ == "__main__":
    _ = run_pipeline()
