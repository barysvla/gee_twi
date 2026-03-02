# TWI Calculator — Google Earth Engine + Colab

Python-based workflow for computing the **Topographic Wetness Index (TWI)** using Google Earth Engine (GEE) and Google Colab.

The tool combines cloud-based raster processing with locally executed topological algorithms to enable flexible and reproducible TWI computation over user-defined areas.

---

## Features

- Multiple DEM sources (30 m and 90 m; DTM and DSM variants)
- Hydrological conditioning (depression filling, flat resolution)
- Flow routing methods: **D8** and **MFD** (Quinn, 1991)
- Cloud mode (server-side processing in GEE)
- Local mode (NumPy-based routing and accumulation)
- GeoTIFF export (Google Drive or local download)


https://colab.research.google.com/github/barysvla/gee_twi/blob/main/notebooks/gee_twi_workflow.ipynb
