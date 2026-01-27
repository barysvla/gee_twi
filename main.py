# main.py
import ee
import geemap
import numpy as np
import tempfile
import os

from scripts.io_grid import export_dem_and_area_to_arrays
from scripts.save_array_as_geotiff import save_array_as_geotiff
from scripts.twi_np import compute_twi_numpy

from scripts.fill_depressions import priority_flood_fill
from scripts.resolve_flats import resolve_flats_barnes_tie

from scripts.flow_direction_quinn_cit import compute_flow_direction_quinn_cit
from scripts.flow_direction_quinn_1991 import compute_flow_direction_quinn_1991
from scripts.flow_direction_sfd_inf import compute_flow_direction_sfd_inf
from scripts.flow_direction_dz_mfd import compute_flow_direction_dz_mfd
from scripts.flow_direction_md_inf import compute_flow_direction_md_infinity
from scripts.flow_direction_qin_2007 import compute_flow_direction_qin_2007

from scripts.flow_accumulation_quinn_cit import compute_flow_accumulation_quinn_cit
from scripts.flow_accumulation_sfd_inf import compute_flow_accumulation_sfd_inf
from scripts.flow_accumulation_quinn_1991 import compute_flow_accumulation_quinn_1991
from scripts.flow_accumulation_md_inf import compute_flow_accumulation_md_infinity
from scripts.flow_accumulation_qin_2007 import compute_flow_accumulation_qin_2007

from scripts.push_to_ee import push_array_to_ee_geotiff
from scripts.clip_tif import clip_tif_by_geojson
from scripts.slope import compute_slope, slope_ee_to_numpy_on_grid
from scripts.twi import compute_twi
from scripts.visualization import visualize_map, vis_2sigma, plot_tif

from google.colab import files

def run_pipeline(
    project_id: str = None,
    geometry: ee.Geometry = None,                # ORIGINAL, UNBUFFERED ROI (for clipping)
    accum_geometry: ee.Geometry = None,          # BUFFERED ROI FOR ACCUMULATION (optional; falls back to geometry)
    dem_source: str = "FABDEM",
    flow_method: str = "quinn_1991",
    use_bucket: bool = False,
) -> dict:
    """
    Compute DEM conditioning, flow direction/accumulation, slope (on buffered ROI) and TWI (on unbuffered ROI).
    Returns:
        dict with:
            - ee_flow_accumulation        (ee.Image)  # clipped to unbuffered ROI
            - ee_flow_accumulation_full   (ee.Image)  # full accumulation over buffered ROI
            - geometry                    (ee.Geometry)        # unbuffered ROI
            - geometry_accum              (ee.Geometry)        # buffered ROI actually used for accumulation
            - scale                       (ee.Number)
            - slope                       (ee.Image)          # clipped to unbuffered ROI
            - twi                         (ee.Image)          # clipped to unbuffered ROI
            - map                         (geemap.Map)
    """
    # --- Initialize Earth Engine ---
    ee.Initialize(project=project_id)

    # --- Regions of interest ---
    if geometry is None:
    # Fail fast: the caller must pass a valid ee.Geometry; do not fall back to defaults
        raise ValueError("Missing required parameter: geometry")

    # Use the same ROI for accumulation if no separate (buffered) geometry was provided
    if accum_geometry is None:
        accum_geometry = geometry  # default: no buffer

    # --- DEM source selection ---
    if dem_source == "FABDEM":
        dem_raw = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    elif dem_source == "GLO30":
        dem_raw = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")  # DSM
    elif dem_source == "AW3D30":
        dem_raw = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select("DSM")  # DSM
    elif dem_source == "SRTMGL1_003":
        dem_raw = ee.Image("USGS/SRTMGL1_003").select("elevation")
    elif dem_source == "NASADEM_HGT":
        dem_raw = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    elif dem_source == "ASTER_GDEM":
        dem_raw = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").select("b1")
    elif dem_source == "CGIAR_SRTM90":
        dem_raw = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    elif dem_source == "MERIT_Hydro":
        dem_raw = ee.Image("MERIT/Hydro/v1_0_1").select("elv")
    else:
        raise ValueError(f"Unsupported dem_source: {dem_source}")

    # --- Export DEM to numpy grid over the ACCUMULATION geometry (buffered) ---
    grid = export_dem_and_area_to_arrays(
        src=dem_raw,
        region_geom=accum_geometry,      # compute grids over buffered extent to reduce boundary effects
        band=None,
        resample_method="bilinear",
        nodata_value=-9999.0,
        snap_region_to_grid=True,
    )

    dem_np       = grid["dem_elevations"]        # DEM in numpy
    px_area_np   = grid["pixel_area_m2"]         # pixel area (numpy)
    transform    = grid["transform"]
    nodata_mask  = grid["nodata_mask"]
    crs          = grid["crs"]
    ee_dem_grid  = grid["ee_dem_grid"]           # DEM (Earth Engine grid-locked)


    scale = ee.Number(ee_dem_grid.projection().nominalScale())
    # print("nominalScale [m]:", scale.getInfo())

    # --- Hydrologic conditioning (client-side arrays) ---
    dem_filled, fill_depth = priority_flood_fill(
        dem_np,
        seed_internal_nodata_as_outlet=True,
        return_fill_depth=True,
    )
    print("✅ Fill pits completed.")

    dem_resolved, flatmask, labels, stats = resolve_flats_barnes_tie(
        dem_filled,
        nodata=np.nan,
        epsilon=2e-5,
        equal_tol=1e-3,
        lower_tol=0.0,
        treat_oob_as_lower=True,
        require_low_edge_only=True,
        force_all_flats=False,
        include_equal_ties=True,
    )
    print("✅ Flats resolved.")

    # --- Flow direction (on buffered grid) ---
    if flow_method == "sfd_inf":
        flow_direction = compute_flow_direction_sfd_inf(
            dem_resolved, transform, nodata_mask=nodata_mask
        )
    elif flow_method == "dz_mfd":
        flow_direction = compute_flow_direction_dz_mfd(
            dem_resolved, p=1.6, nodata_mask=nodata_mask
        )
    elif flow_method == "quinn_cit":
        flow_direction = compute_flow_direction_quinn_cit(
            dem_resolved, transform, p=1.0, nodata_mask=nodata_mask
        )
    elif flow_method == "quinn_1991":
        flow_direction = compute_flow_direction_quinn_1991(
            dem_resolved, transform, p=1.0, nodata_mask=nodata_mask
        )
    elif flow_method == "md_infinity":
        flow_direction = compute_flow_direction_md_infinity(
            dem_resolved, transform, nodata_mask=nodata_mask
        )
    elif flow_method == "qin_2007":
        flow_direction = compute_flow_direction_qin_2007(
            dem_resolved, transform, nodata_mask=nodata_mask
        )
    else:
        raise ValueError(f"Unsupported flow_method: {flow_method}")
    print("✅ Flow direction computed.")

    # --- Flow accumulation on buffered domain ---
    # Example: Qin 2007; adjust to match selected flow_method if you have variants
    # acc_cells = compute_flow_accumulation_qin_2007(
    #     flow_direction, nodata_mask=nodata_mask, out="cells"
    # )
    acc_km2 = compute_flow_accumulation_qin_2007(
        flow_direction, pixel_area_m2=px_area_np, nodata_mask=nodata_mask, out="km2"
    )
    print("✅ Flow accumulation computed.")

    # Branch: cloud mode vs local mode
    if use_bucket:
        # Upload arrays to EE via bucket
        # dict_acc_cells = push_array_to_ee_geotiff(
        #     acc_cells,
        #     transform=transform,
        #     crs=crs,
        #     nodata_mask=nodata_mask,
        #     bucket_name=f"{project_id}-ee-uploads",
        #     project_id=project_id,
        #     band_name="flow_accumulation_cells",
        #     tmp_dir=grid.get("tmp_dir", None),
        #     nodata_value=np.nan,
        # )
        dict_acc = push_array_to_ee_geotiff(
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
            # dtype="float32",
            # build_mask_from_nodata=True,
        )
        # ee_flow_accumulation_cells_full = dict_acc_cells["image"]
        ee_flow_accumulation_full = dict_acc["image"]

        # Clip to original ROI
        # ee_flow_accumulation_cells = ee_flow_accumulation_cells_full.clip(geometry)
        ee_flow_accumulation = ee_flow_accumulation_full.clip(geometry)

        # Slope & TWI via EE
        slope_full = compute_slope(ee_dem_grid)
        slope = slope_full.clip(geometry)
        print("✅ Slope computed.")
        twi = compute_twi(ee_flow_accumulation, slope).clip(geometry)
        print("✅ Twi computed.")

        # CTI reference
        cti_ic = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti")
        cti = cti_ic.mosaic().toFloat().divide(ee.Number(1e8)).rename("CTI").clip(geometry)

        # Visualization
        vis_twi = vis_2sigma(
            twi, "TWI", geometry, scale, k=2.0,
            palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
        )
        vis_cti = vis_2sigma(
            cti, "CTI", geometry, scale, k=2.0,
            palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
        )
        vis_acc = vis_2sigma(
            ee_flow_accumulation, "flow_accumulation_km2", geometry, scale, k=2.0,
            palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
        )
        # vis_acc_cells = vis_2sigma(
        #     ee_flow_accumulation_cells, "flow_accumulation_cells", geometry, scale, k=2.0,
        #     palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
        # )

        Map = visualize_map([
            # (ee_flow_accumulation_cells, vis_acc_cells, "Flow accumulation (cells)"),
            (ee_flow_accumulation, vis_acc, "Flow accumulation (km²)"),
            (cti, vis_cti, "CTI - reference (Hydrography90m)"),
            (twi, vis_twi, "TWI"),
        ])
        Map.centerObject(geometry, 12)

        return {
            "mode": "cloud",
            "ee_flow_accumulation": ee_flow_accumulation,
            "ee_flow_accumulation_full": ee_flow_accumulation_full,
            "geometry": geometry,
            "geometry_accum": accum_geometry,
            "scale": scale,
            "slope": slope,
            "twi": twi,
            "map": Map,
        }

    else:
        # Local mode: compute slope & TWI in numpy, save TIFFs
        # Slope via EE → numpy
        slope_np = slope_ee_to_numpy_on_grid(grid, ee_dem_grid)
        print("✅ Slope computed.")

        # Compute twi numpy
        # Here we assume acc_km2 is area (m^2). If not, use acc_cells and cell area = px_area_np.
        twi_np = compute_twi_numpy(
            acc_np=acc_km2,
            slope_deg_np=slope_np,
            acc_is_area=True,
            cell_area=None,
            min_slope_deg=0.1,
            nodata_mask=nodata_mask,
            out_dtype="float32"
        )
        print("✅ Twi computed.")

        # Save arrays to GeoTIFFs
        geotiff_acc_km2 = save_array_as_geotiff(
            acc_km2, transform, crs, nodata_mask,
            filename="flow_accumulation_km2.tif", band_name="Flow accumulation (km2)"
        )
        geotiff_slope = save_array_as_geotiff(
            slope_np, transform, crs, nodata_mask,
            filename="slope.tif", band_name="Slope"
        )
        geotiff_twi = save_array_as_geotiff(
            twi_np, transform, crs, nodata_mask,
            filename="twi.tif", band_name="TWI"
        )

        geometry_wgs84 = geometry.getInfo()          

        acc_km2_clipped = clip_tif_by_geojson(geotiff_acc_km2, geometry_wgs84, "acc_km2_clipped.tif", band_name="Flow accumulation (km2)")
        slope_clipped = clip_tif_by_geojson(geotiff_slope, geometry_wgs84, "slope_clipped.tif", band_name="Slope")
        twi_clipped = clip_tif_by_geojson(geotiff_twi, geometry_wgs84, "twi_clipped.tif", band_name="TWI")

        # vis_twi = vis_2sigma_tif(twi_clipped, clamp_to_pct=(2,98), k=2.0,
        #                 palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"])
        # vis_slope = vis_2sigma_tif(slope_clipped, clamp_to_pct=(2,98), k=2.0,
        #                 palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"])
        # vis_acc = vis_2sigma_tif(acc_km2_clipped, clamp_to_pct=(2,98), k=2.0,
        #                 palette=["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"])

        # Plot TWI as a static figure from the clipped GeoTIFF
        print("🖼 Plotting TWI (local mode, percentile stretch)…")
        plot_tif(
            twi_clipped,
            p_low=2.0,
            p_high=98.0,
            label="TWI",
            title="Topographic Wetness Index",
        )

        # Return metadata and file paths
        return {
            "mode": "local",
            "acc_km2_array": acc_km2_clipped,
            "slope_array": slope_clipped,
            "twi_array": twi_clipped,
            "geotiff_acc_km2_path": geotiff_acc_km2,
            "geotiff_slope_path": geotiff_slope,
            "geotiff_twi_path": geotiff_twi,
            "transform": transform,
            "crs": crs,
            "nodata_mask": nodata_mask,
        }

        # Map = visualize_map_leaf([
        #     #(ee_flow_accumulation_cells, vis_acc_cells, "Flow accumulation (cells)"),
        #     (acc_km2_clipped, vis_acc, "Flow accumulation (km²)"),
        #     # (cti, vis_cti, "CTI - reference (Hydrography90m)"),
        #     (twi_clipped, vis_twi, "TWI"),
        # ], basemaps=["Esri.WorldImagery", "Esri.WorldTopoMap"])
        
        # return {
        #     "mode": "local",
        #     "acc_km2_array": acc_km2_clipped,
        #     "slope_array": slope_clipped,
        #     "twi_array": twi_clipped,
        #     "geotiff_acc_km2_path": geotiff_acc_km2,
        #     "geotiff_slope_path": geotiff_slope,
        #     "geotiff_twi_path": geotiff_twi,
        #     "transform": transform,
        #     "crs": crs,
        #     "nodata_mask": nodata_mask,
        #     "map": Map,
        # }

if __name__ == "__main__":
    _ = run_pipeline()
