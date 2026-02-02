# GEFS Grib-Index-Kerchunk (GIK) to cGAN Inference Integration Plan

## Overview

This document outlines the integration of **Grib-Index-Kerchunk (GIK)** data streaming with **cGAN precipitation downscaling inference**. The goal is to create a streamlined workflow that:

1. Streams GEFS ensemble forecast data directly from AWS S3 using GIK references
2. Converts streamed data to the NetCDF format required by cGAN
3. Runs cGAN inference to produce high-resolution precipitation forecasts

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEFS GIK → cGAN Inference Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: GIK Reference Creation                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input: Hugging Face Template (gik-fmrc-gefs-20241112.tar.gz)        │   │
│  │         Target Date (e.g., 20250106)                                  │   │
│  │         Ensemble Members (gep01-gep30)                                │   │
│  │                                                                       │   │
│  │  Process: run_gefs_tutorial.py                                        │   │
│  │           - Download GIK templates from Hugging Face                  │   │
│  │           - Scan GRIB structure for target date                       │   │
│  │           - Create mapped index from templates                        │   │
│  │           - Build zarr reference store                                │   │
│  │                                                                       │   │
│  │  Output: Parquet reference files (gep01_20250106_00z.parquet, ...)   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  STAGE 2: Multi-Variable Data Streaming                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input: Parquet reference files                                       │   │
│  │         Variable list: cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp  │   │
│  │                                                                       │   │
│  │  Process: stream_gefs_for_cgan.py (NEW)                               │   │
│  │           - Read parquet references                                   │   │
│  │           - Stream each variable from S3 using gribberish             │   │
│  │           - Subset to ICPAC region (19.15-54.3°E, -13.65-24.7°N)     │   │
│  │           - Store in disk-based zarr for memory efficiency            │   │
│  │                                                                       │   │
│  │  Output: Zarr stores with all 8 variables × 30 ensemble members       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  STAGE 3: Zarr to NetCDF Conversion                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input: Zarr stores with streaming data                               │   │
│  │                                                                       │   │
│  │  Process: zarr_to_raw_netcdf.py (existing/enhanced)                   │   │
│  │           - Convert zarr to NetCDF format                             │   │
│  │           - Organize by: {field}_{year}.nc                            │   │
│  │           - Dimensions: (time, member, step, latitude, longitude)     │   │
│  │                                                                       │   │
│  │  Output: NetCDF files per variable                                    │   │
│  │          cape_2025.nc, pres_2025.nc, pwat_2025.nc, etc.              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  STAGE 4: cGAN Inference                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input: Raw NetCDF files                                              │   │
│  │         Model weights (gen_weights-0345600.h5)                        │   │
│  │         Constants (elev.nc, lsm.nc, FCSTNorm_GEFS_2018.pkl)          │   │
│  │                                                                       │   │
│  │  Process: run_gefs_inference_raw.py (existing)                        │   │
│  │           - Load and normalize forecast fields                        │   │
│  │           - Compute ensemble mean/std for each field                  │   │
│  │           - Run generator model with noise                            │   │
│  │           - Produce 50 precipitation ensemble members                 │   │
│  │                                                                       │   │
│  │  Output: GAN_{YYYYMMDD}.nc with high-res precipitation                │   │
│  │          Dims: (time, member, valid_time, latitude, longitude)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Mapping

| Stage | Script | Source Location | Function |
|-------|--------|-----------------|----------|
| 1 | `run_gefs_tutorial.py` | `grib-index-kerchunk/tutorial/gefs/` | Creates GIK parquet references |
| 2 | `stream_gefs_for_cgan.py` | **NEW** (to be created) | Streams multi-variable GEFS data |
| 3 | `zarr_to_raw_netcdf.py` | `cGAN_tutorial/example_notebooks/` | Converts zarr to NetCDF |
| 4 | `run_gefs_inference_raw.py` | `cGAN_tutorial/example_notebooks/` | Runs cGAN inference |

## Required Variables for cGAN

The cGAN model requires 8 input fields from GEFS:

| cGAN Field | GEFS Variable | GRIB Filter String | Description |
|------------|---------------|-------------------|-------------|
| `cape` | CAPE | `CAPE:surface` | Convective Available Potential Energy |
| `pres` | SP | `PRES:surface` | Surface Pressure |
| `pwat` | PWAT | `PWAT:entire atmosphere` | Precipitable Water |
| `tmp` | T2M | `TMP:2 m above ground` | 2-meter Temperature |
| `ugrd` | U10 | `UGRD:10 m above ground` | 10-meter U Wind |
| `vgrd` | V10 | `VGRD:10 m above ground` | 10-meter V Wind |
| `msl` | MSLET | `MSLET:mean sea level` | Mean Sea Level Pressure |
| `apcp` | TP | `APCP:surface` | Total Precipitation |

## Current Implementation Gap

### Existing `run_gefs_data_streaming.py`

- **Currently supports**: Only precipitation (`APCP:surface`)
- **Limitation**: Single variable streaming

### Required Enhancement

- **Need**: All 8 variables streamed simultaneously
- **Challenge**: Different variables may have different GRIB structures
- **Solution**: Create `stream_gefs_for_cgan.py` that handles multi-variable streaming

## Implementation Steps

### Step 1: Run GIK Tutorial to Create References

```bash
cd /scratch/notebook/grib-index-kerchunk/tutorial/gefs
python run_gefs_tutorial.py
```

**Configuration to modify in `run_gefs_tutorial.py`:**
```python
TARGET_DATE = '20250918'  # Your target forecast date
TARGET_RUN = '00'         # Model run (00, 06, 12, 18)
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # 30 members

# Extended variables for cGAN
FORECAST_VARIABLES = {
    "Surface pressure": "PRES:surface",
    "2 metre temperature": "TMP:2 m above ground",
    "10m U wind": "UGRD:10 m above ground",
    "10m V wind": "VGRD:10 m above ground",
    "Precipitable water": "PWAT:entire atmosphere (considered as a single layer)",
    "CAPE": "CAPE:surface",
    "Mean sea level pressure": "MSLET:mean sea level",
    "Total Precipitation": "APCP:surface",
}
```

### Step 2: Stream Multi-Variable Data

Use the new `stream_gefs_for_cgan.py` script:

```bash
python stream_gefs_for_cgan.py \
    --parquet_dir output_parquet \
    --output_dir cgan_zarr_output \
    --date 20250918 \
    --run 00
```

### Step 3: Convert to NetCDF

```bash
python zarr_to_raw_netcdf.py \
    --input_dir cgan_zarr_output/20250918_00 \
    --output_dir cgan_raw_netcdf/2025 \
    --date 2025-09-18
```

### Step 4: Run cGAN Inference

```bash
micromamba run -n tf215gpu python run_gefs_inference_raw.py
```

## Streamlined End-to-End Script

A single unified script `run_gefs_gik_cgan_pipeline.py` will be created that:

1. Downloads GIK templates (if not present)
2. Creates parquet references for target date
3. Streams all required variables from S3
4. Converts to NetCDF format
5. Runs cGAN inference
6. Outputs high-resolution precipitation forecasts

## Technical Notes

### Memory Efficiency

Both GIK streaming and cGAN inference use disk-based storage to handle large ensemble datasets:

- **GIK Streaming**: Uses `zarr` with `BloscCodec` compression
- **cGAN Inference**: Processes one timestep at a time

### ICPAC Region Subset

Both processes subset data to the ICPAC East Africa region:

- **Latitude**: -13.65°N to 24.7°N (384 points at 0.1° resolution)
- **Longitude**: 19.15°E to 54.3°E (352 points at 0.1° resolution)

### Forecast Hours for cGAN

The cGAN inference typically uses:
- **Start Hour**: 30 (Day 1.25)
- **End Hour**: 54 (Day 2.25)
- **Step**: 6 hours

This requires GEFS data from steps 30-54 at 3-hour intervals.

## Dependencies

### GIK Data Streaming
```
kerchunk>=0.2.0
zarr>=3.0.0
xarray
pandas
numpy
fsspec
s3fs
gribberish  # For fast GRIB decoding
```

### cGAN Inference
```
tensorflow>=2.15
numpy<2.0
xarray
netCDF4
pyyaml
```

## Output Products

| Product | Format | Dimensions | Description |
|---------|--------|------------|-------------|
| GIK References | Parquet | N/A | Byte offset mappings for S3 |
| Streamed Data | Zarr | (member, step, lat, lon) | Raw ensemble data |
| cGAN Input | NetCDF | (time, member, step, lat, lon) | Normalized input fields |
| cGAN Output | NetCDF | (time, member, valid_time, lat, lon) | High-res precipitation |

## Error Handling

1. **Missing GEFS Data**: Check AWS S3 availability for target date
2. **Template Mismatch**: Ensure GIK templates match GEFS structure
3. **Memory Issues**: Use disk-based zarr storage
4. **GPU Availability**: Falls back to CPU for inference

## References

- **GIK Tutorial**: `grib-index-kerchunk/tutorial/gefs/run_gefs_tutorial.py`
- **Data Streaming Demo**: `grib-index-kerchunk/tutorial/gefs/run_gefs_data_streaming.py`
- **Zarr Conversion**: `cGAN_tutorial/example_notebooks/zarr_to_raw_netcdf.py`
- **cGAN Inference**: `cGAN_tutorial/example_notebooks/run_gefs_inference_raw.py`
- **GEFS Inference Guide**: `cGAN_tutorial/docs/GEFS_INFERENCE_GUIDE.md`

---

*Created: February 2026*
*Git Commit Reference: 2d7e5de (added the files for grib-index-kerchunk gefs zarr for cGAN inference)*
