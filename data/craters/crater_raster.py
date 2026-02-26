# References / origin notes:
# - Uses GDAL/OSR to avoid hand-deriving polar stereographic math:
#   lon/lat (EPSG:4326) -> dataset projection -> pixel (geoTransform).
# - Circle rasterisation approach is standard: bounding-box loop + distance check.
#
# Requirements:
#   pip install numpy pandas
#   and GDAL available in environment (often preinstalled in Colab via apt).
#
# Visual intuition:
#   crater circle fill: (r-r0)^2 + (c-c0)^2 <= radius_px^2

