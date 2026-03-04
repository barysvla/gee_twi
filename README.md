# TWI Calculator (Google Earth Engine + Colab)

Interactive workflow for computing the **Topographic Wetness Index (TWI)**
from global DEM datasets using Google Earth Engine and Google Colab.

The workflow allows the user to select an area of interest, choose the DEM 
source and flow routing method, and export terrain-derived outputs 
such as slope, flow accumulation, and TWI.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barysvla/gee_twi/blob/main/notebooks/gee_twi_workflow.ipynb)


## Features

- Interactive **AOI selection** (draw on map or upload vector file)
- Support for multiple **global DEM datasets** (30 m and 90 m resolution)
- **Hydrological conditioning** of DEM (depression filling and flat resolution)
- Selectable **flow routing algorithms** (D8, MFD – Quinn 1991)
- Optional **buffer around AOI** to reduce edge effects in accumulation and slope calculation
- Computation of **flow accumulation**, **slope**, and **Topographic Wetness Index**
- Optional **reference layers** for comparison (MERIT Hydro UPA, Hydrography90m CTI)
- Export of outputs as **GeoTIFF** (Google Drive or local download)
- Interactive **visualization in Google Earth Engine map interface**

## Workflow overview

The workflow consists of several sequential steps executed in the Colab notebook:

1. **Authentication and environment setup**  
   The user authenticates with Google Earth Engine and provides a Cloud Project ID.  
   The notebook clones the repository and installs required Python dependencies.

2. **Area of Interest (AOI) definition**  
   The AOI can be defined by drawing a polygon on the interactive map or by uploading a vector file (GeoJSON, GPKG, KML/KMZ, or Shapefile).

3. **DEM selection**  
   A global DEM dataset is selected from the available sources (e.g., FABDEM, Copernicus GLO-30, MERIT DEM, SRTM).

4. **Hydrological conditioning of DEM**  
   The DEM is hydrologically conditioned to ensure correct surface drainage representation.  
   Depressions are filled using the **Priority-Flood algorithm** (Barnes et al., 2014).  
   Flat areas are then resolved using the **improved flat-resolution method** proposed by Barnes et al. (2015), which constructs a drainage field by superimposing a gradient away from higher terrain with a gradient toward lower terrain. This approach improves flow convergence compared to earlier methods such as Garbrecht and Martz (1997).

5. **Flow routing and accumulation computation**  
   Flow routing is calculated using either the D8 or MFD (Quinn 1991) algorithm, followed by computation of upslope contributing area.

6. **Slope and TWI computation**  
   Terrain slope is calculated from the DEM and combined with flow accumulation to compute the Topographic Wetness Index:

`TWI = ln(a / tan β)`

where  
- `a` — upslope contributing area per unit contour width  
- `β` — slope (radians)
  
7. **Visualization and export**  
   Results are visualized in an interactive Earth Engine map and can be exported as GeoTIFF files either to Google Drive or to local storage.


## References

Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-Flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*.

Barnes, R., Lehman, C., & Mulla, D. (2015). An efficient assignment of drainage direction over flat surfaces in raster digital elevation models. *Computers & Geosciences*.
