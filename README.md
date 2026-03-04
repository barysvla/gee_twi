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
## Workflow Architecture

The workflow integrates:

- **Google Earth Engine** — data access, server-side raster operations, and visualization  
- **Google Colab (Python environment)** — workflow orchestration and interactive interface  
- **NumPy-based algorithms** — hydrological conditioning, flow direction, and flow accumulation

TWI is defined as:

`TWI = ln(a / tan β)`

where  
- `a` — upslope contributing area per unit contour width  
- `β` — slope (radians)
