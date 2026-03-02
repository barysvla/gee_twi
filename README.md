# TWI Calculator — Google Earth Engine + Colab

Python-based workflow for computing the **Topographic Wetness Index (TWI)** using Google Earth Engine (GEE) and Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barysvla/gee_twi/blob/main/notebooks/gee_twi_workflow.ipynb)

---

The tool combines cloud-based raster processing with locally executed topological algorithms to enable flexible and reproducible TWI computation over user-defined areas.

---

## Features

- Multiple DEM sources (30 m and 90 m; DTM and DSM variants)
- Hydrological conditioning (depression filling, flat resolution)
- Flow direction methods: **D8** and **MFD** (Quinn, 1991)
- **Two execution modes for final steps:**
  - **Cloud mode:** TWI computed and visualized in **GEE** (interactive map)
  - **Local mode:** TWI computed and visualized **locally** (static map)
- GeoTIFF export of selected outputs (TWI, slope, flow accumulation)

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
