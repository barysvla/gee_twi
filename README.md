# TWI Calculator (Google Earth Engine + Colab)

Interactive workflow for computing the **Topographic Wetness Index (TWI)**
from global DEM datasets using Google Earth Engine (GEE) and Google Colab.

The workflow allows the user to select an area of interest, choose the DEM 
source and flow routing method, and export terrain-derived outputs 
such as slope, flow accumulation, and TWI.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barysvla/gee_twi/blob/main/notebooks/gee_twi_workflow.ipynb) 

Open the notebook in Google Colab using the badge above, save a copy to your Google Drive, and follow the step-by-step instructions provided directly in the notebook.

## Processing architecture

The workflow combines server-side processing in Google Earth Engine with local computation in the Colab environment.

DEM conditioning, flow routing, and flow accumulation are performed locally using NumPy, while terrain slope is derived in Earth Engine.

The workflow branches into cloud or local execution only at the TWI computation stage.

## Project structure

- `main.py`  
  Main entry point coordinating the workflow.

- `scripts/`  
  Core processing modules:  
  - `fill_depressions.py`, `resolve_flats.py` – DEM conditioning  
  - `flow_direction_d8.py`, `flow_direction_mfd.py` – flow routing  
  - `flow_accumulation.py` – contributing area computation  
  - `twi.py` – TWI calculation  
  - `grid_io.py`, `geotiff_io.py`, `numpy_to_ee.py` – data transfer and export  
  - `visualization.py` – visualization utilities  

- `notebooks/gee_twi_workflow.ipynb`  
  Interactive Colab notebook for running the workflow.

## Requirements

- Google Earth Engine account
- Google Cloud Project registered in GEE
- Enabled APIs:
  - Google Earth Engine API
  - Cloud Billing API
- Billing account (required for server-side TWI computation in GEE)

Local execution does not require billing.

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
   Depressions are filled using **Priority-Flood (Barnes et al., 2014a)** and flats are resolved using the method of **Barnes et al. (2014b)**.
   
5. **Flow routing and accumulation computation**  
   Flow routing is computed using either the **D8 (single-flow direction)** algorithm or the **MFD (multiple-flow direction )** method proposed by Quinn et al. (1991). The resulting flow distribution is then used to compute the upslope contributing area.

6. **Slope computation**  
   Terrain slope is derived from the DEM using the Earth Engine function `ee.Terrain.slope`, which computes slope in degrees on the target grid.

7. **TWI computation**  
   After slope computation, the workflow branches into two execution modes depending on the availability of an active Google Cloud billing account:

   - **Cloud mode (GEE)** – TWI is computed server-side using Earth Engine  
   - **Local mode (NumPy)** – DEM-derived data are exported and TWI is computed locally in the Colab environment

    TWI is defined as:

$$
\mathrm{TWI} = \ln\left(\frac{a}{\tan \beta}\right)
$$

   where $a$ is the total upslope contributing area derived from flow accumulation (km²) and $\beta$ is slope (radians).

9. **Visualization and export**  
   Results are visualized either in the interactive Earth Engine map or locally, and can be exported as GeoTIFF files to Google Drive or local storage.

## Execution model

The division of computation is as follows:

**Operations executed in Google Earth Engine**
- loading the selected DEM dataset
- slope computation using `ee.Terrain.slope`
- visualization in Earth Engine (interactive map)
- TWI computation in cloud mode

**Operations executed locally in Colab (NumPy)**
- export of DEM data from GEE
- hydrological conditioning of the DEM  
  (depression filling and flat resolution)
- flow routing and flow accumulation computation
- TWI computation in local mode

This hybrid approach is used because some hydrological algorithms rely on iterative topological operations that are difficult to implement efficiently within the GEE raster framework.

## References

Barnes, R., Lehman, C., & Mulla, D. (2014a). Priority-Flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*, 62, 117–127.

Barnes, R., Lehman, C., & Mulla, D. (2014b). An efficient assignment of drainage direction over flat surfaces in raster digital elevation models. *Computers & Geosciences*, 62, 128–135.

Barták, V. (2008). Algoritmy pro zpracování digitálních modelů terénu s aplikacemi v hydrologickém modelování. Diplomová práce, Česká zemědělská univerzita v Praze.

O'Callaghan, J. F., & Mark, D. M. (1984). The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28(3), 323–344.

Quinn, P., Beven, K., Chevallier, P., & Planchon, O. (1991). The prediction of hillslope flow paths for distributed hydrological modelling using digital terrain models. *Hydrological Processes*, 5(1), 59–79.
