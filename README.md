# TWI Calculator (Google Earth Engine + Colab)

Interactive workflow for computing the **Topographic Wetness Index (TWI)**
from global DEM datasets using Google Earth Engine (GEE) and Google Colab.

The workflow allows the user to select an area of interest, choose the DEM 
source and flow routing method, and export terrain-derived outputs 
such as slope, flow accumulation, and TWI.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barysvla/gee_twi/blob/main/notebooks/gee_twi_workflow.ipynb)

Open the notebook in Google Colab using the badge above, save a copy to your Google Drive, and follow the step-by-step instructions provided directly in the notebook.

## Workflow overview

The workflow consists of several sequential steps executed in the Colab notebook:

1. **Authentication and environment setup**  
   The user authenticates with GEE and provides a Cloud Project ID.  
   The notebook clones the repository and installs required Python dependencies.

2. **Area of Interest (AOI) definition**  
   The AOI can be defined by drawing a polygon on the interactive map or by uploading a vector file (GeoJSON, GPKG, KML/KMZ, or Shapefile).

3. **DEM selection**  
   A global DEM dataset is selected from the available sources (e.g., FABDEM, Copernicus GLO-30, MERIT DEM, SRTM).

4. **Hydrological conditioning of DEM**  
   Depressions are filled using **Priority-Flood (Barnes et al., 2014)** and flats are resolved using the method of **Barnes et al. (2015)**.
   
6. **Flow routing and accumulation computation**  
   Flow routing is computed using either the **D8 (single-flow direction)** algorithm or the **multiple-flow direction (MFD)** method proposed by Quinn et al. (1991). The resulting flow distribution is then used to compute the upslope contributing area.

7. **Slope and TWI computation**  
   Terrain slope is calculated from the DEM and combined with flow accumulation to compute the Topographic Wetness Index:

      `TWI = ln(a / tan β)`
      
      where  
      - `a` — upslope contributing area per unit contour width  
      - `β` — slope (radians)
  
7. **Visualization and export**  
   Results are visualized in an interactive Earth Engine map and can be exported as GeoTIFF files either to Google Drive or to local storage.

## Execution model

The workflow combines **server-side processing in GEE** with **local computation in the Colab environment**.

The division of computation is as follows:

**Operations executed in Google Earth Engine**
- loading the selected DEM dataset
- slope computation using `ee.Terrain.slope`
- optional cloud-based visualization
- TWI computation in cloud mode

**Operations executed locally in Colab (NumPy)**
- export of DEM data from GEE
- hydrological conditioning of the DEM  
  (depression filling and flat resolution)
- flow routing and flow accumulation computation
- TWI computation in local mode

This hybrid approach is used because some hydrological algorithms rely on iterative topological operations that are difficult to implement efficiently within the GEE raster framework.

## References

Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-Flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*.

Barnes, R., Lehman, C., & Mulla, D. (2015). An efficient assignment of drainage direction over flat surfaces in raster digital elevation models. *Computers & Geosciences*.

Quinn, P., Beven, K., Chevallier, P., & Planchon, O. (1991). The prediction of hillslope flow paths for distributed hydrological modelling using digital terrain models. *Hydrological Processes*, 5(1), 59–79.
