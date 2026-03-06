# Data Layer

This module handles terrain data access and preprocessing.

Key responsibilities:

• Loading DEM tiles and crater catalogues  
• Extracting regions of interest (ROI)  
• Managing caching and derived datasets  

Modules:

- `data_handler.py`
  Central interface for loading terrain data and storing derived products.

- `crater_raster.py`
  Converts crater catalogue data into raster masks aligned with DEM grids.