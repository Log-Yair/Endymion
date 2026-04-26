
# Endymion Data Module

## Overview

The `src/data/` module handles the data ingestion and raster preparation stage of Endymion.

Its main responsibility is to make sure lunar terrain and crater data are loaded, cached, cropped, aligned, and saved in a consistent format before the rest of the pipeline uses them.

---

## Files

```text
src/data/
│
├── data_handler.py
└── crater_raster.py
````

---

## Role in the Pipeline

```text
DEM / GeoTIFF
      ↓
DataHandler
      ↓
ROI DEM patch
      ↓
FeatureExtractor
```

For crater data:

```text
Crater catalogue CSV
      ↓
crater_raster.py
      ↓
ROI-aligned crater rasters
      ↓
CraterPredictor / HazardAssessor
```

---

## `data_handler.py`

## Purpose

`data_handler.py` provides the main data loading and storage interface for Endymion.

It supports:

* GeoTIFF-based DEM loading
* legacy LOLA IMG/LBL support
* local and runtime caching
* canonical ROI extraction
* saving and loading derived artefacts

---

## Main Classes

### `DataHandler`

The central class for resolving, loading, cropping, and saving raster data.

Typical responsibilities:

* resolve the selected DEM tile
* download or locate the GeoTIFF
* read raster metadata
* extract a Region of Interest
* standardise elevation data
* save derived outputs
* reload cached outputs

---

### `GeoTiffTileSpec`

Defines a GeoTIFF tile source.

Typical fields:

```text
tile_id
tif_filename
tif_url
```

Example:

```text
tile_id: ldem_80s_20m
tif_filename: LDEM_80S_20MPP_ADJ.TIF
```

---

### `LOLATileSpec`

Legacy support for older IMG/LBL prototype inputs.

This is kept for backwards compatibility with the earlier prototype phase.

---

### `RasterPatch`

Represents an extracted DEM patch.

Contains:

```text
data
meta
roi
```

---

### `RasterMeta`

Stores metadata about a raster patch, such as:

```text
tile_id
shape
dtype
units
nodata
extra
```

---

## Canonical ROI

The final GeoTIFF workflow uses a canonical ROI rather than manually selected prototype coordinates.

The standard ROI size is:

```text
1024 × 1024 pixels
```

For the selected 20 m/pixel DEM, this corresponds to an approximate area of:

```text
20.48 km × 20.48 km
```

The canonical ROI is resolved from the raster grid itself, which keeps the pipeline stable and reproducible.

---

## Derived Artefacts

The `DataHandler` saves processed outputs under a structured directory:

```text
persistent/derived/<tile_id>/roi_<r0>_<r1>_<c0>_<c1>/
```

Typical derived terrain outputs:

```text
dem_m.npy
slope_deg.npy
roughness_rms.npy
```

Typical crater outputs:

```text
crater_mask.npy
crater_mask_distance.npy
crater_mask_density.npy
crater_mask_meta.json
```

---

## Persistent vs Runtime Storage

Endymion separates longer-term cached files from temporary runtime files.

### Persistent directory

Used for reusable data:

```text
C:\Endymion\persistent
```

### Runtime directory

Used for temporary or working copies:

```text
C:\Endymion\runtime
```

This reduces repeated downloads and keeps large data files outside the GitHub repository.

---

## `crater_raster.py`

## Purpose

`crater_raster.py` converts crater catalogue records into raster products aligned with the DEM ROI.

This is necessary because the rest of Endymion operates in raster grid space.

---

## Input

The expected crater input is a catalogue file such as the Robbins lunar crater catalogue.

Required crater fields include:

```text
latitude
longitude
diameter
```

In the Robbins catalogue, these correspond to fields such as:

```text
LAT_CIRC_IMG
LON_CIRC_IMG
DIAM_CIRC_IMG
```

---

## Outputs

The crater raster process generates:

### `crater_mask`

A binary raster:

```text
0 = no crater cell
1 = crater cell
```

### `crater_distance_m`

A distance transform showing distance to the nearest crater cell in metres.

```text
0 = inside crater
larger value = further from crater
```

### `crater_density`

A local crater-density raster based on a moving window.

```text
higher value = more crater influence nearby
lower value = less crater influence nearby
```

---

## Why Crater Rasterisation Is Needed

Crater catalogues are usually stored as tabular or vector data.

However, the hazard model expects raster-aligned inputs.

Therefore:

```text
catalogue crater record
        ↓
project to DEM coordinate space
        ↓
convert to pixel location
        ↓
rasterise into ROI grid
```

This allows crater information to be fused with slope and roughness inside the hazard model.

---

## Data Alignment Rule

All rasters must match the same ROI shape.

For example:

```text
dem_m.shape == slope_deg.shape == roughness_rms.shape == crater_mask.shape
```

If shapes do not match, the pipeline should fail early rather than produce invalid hazard maps.

---

## Design Notes

The data module follows these principles:

### Reproducibility

The same tile and ROI should produce the same saved artefacts.

### Traceability

Metadata is saved alongside generated crater products.

### Scalability

Large source datasets are kept outside GitHub and accessed through cache paths.

### Compatibility

Legacy IMG/LBL support is preserved, but the final workflow prioritises GeoTIFF.

---

## Common Failure Points

### Missing GeoTIFF

If the DEM file is not found and downloading is disabled, the pipeline cannot resolve the tile.

### Missing crater catalogue

Crater-aware mode requires either cached crater rasters or access to the original crater catalogue.

### Shape mismatch

All DEM-derived and crater-derived rasters must match the canonical ROI shape.

### Coordinate transformation issues

Crater catalogue coordinates must be correctly aligned to the DEM coordinate system before rasterisation.

---

## Relationship to Other Modules

```text
data_handler.py
    → provides DEM patch to FeatureExtractor
    → saves derived terrain rasters
    → provides cached artefacts to BenchmarkRunner

crater_raster.py
    → creates crater rasters
    → consumed by CraterPredictor
    → used by HazardAssessor
```

---

## Summary

The `src/data/` module is the foundation of the Endymion pipeline.

It ensures that terrain and crater data are:

* loaded correctly
* spatially aligned
* cached consistently
* saved for reproducible reuse
* ready for feature extraction, hazard modelling, pathfinding, and benchmarking

```
```
