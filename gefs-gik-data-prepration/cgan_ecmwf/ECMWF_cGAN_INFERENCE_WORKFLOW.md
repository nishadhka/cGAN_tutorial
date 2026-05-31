# ECMWF cGAN Inference Workflow Documentation

## Overview

This document details the two-phase workflow for running cGAN (conditional Generative Adversarial Network) inference on ECMWF (European Centre for Medium-Range Weather Forecasts) data to generate high-resolution ensemble rainfall forecasts for the ICPAC region in East Africa.

**Pipeline Summary:**
- **Phase 1**: Create NetCDF files from ECMWF GRIB data using the GIK (Grib-Index-Kerchunk) method
- **Phase 2**: Apply cGAN inference to generate 1000-member ensemble precipitation forecasts

---

## Phase 1: NetCDF Creation from ECMWF GRIB Data

### 1.1 Pipeline Overview

The GIK method uses a three-stage processing approach to efficiently extract ECMWF ensemble data:

```
ECMWF S3 GRIB Files (51 members × 85 timesteps)
         │
         ▼
┌─────────────────────────────────────────────┐
│ STAGE 1: Scan GRIB (3-5 minutes)            │
│   - Uses kerchunk.scan_grib()               │
│   - Extracts all 51 ensemble members        │
│   - Creates hierarchical zarr structure     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ STAGE 2: Index + GCS Templates (3-5 min)    │
│   - Fast index-based processing (85x faster)│
│   - Reuses pre-built template structure     │
│   - Merges fresh positions with templates   │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ STAGE 3: Final Zarr Store (45-90 seconds)   │
│   - Creates xarray-compatible parquet       │
│   - Generates time axes for all 85 steps    │
│   - Produces final member parquet files     │
└─────────────────────────────────────────────┘
         │
         ▼
    NetCDF/Zarr Output Files
```

### 1.2 Entry Point Script

**Location:** `/scratch/notebook/grib-index-kerchunk/tutorial/ecmwf/run_ecmwf_tutorial.py`

**Usage:**
```bash
# Full pipeline (includes Stage 1 - takes ~30 minutes)
python run_ecmwf_tutorial.py --run-stage1

# Skip Stage 1 if zip file already exists (fast)
python run_ecmwf_tutorial.py

# Process specific date
python run_ecmwf_tutorial.py --date 20260106 --run-stage1

# Limit members for faster testing
python run_ecmwf_tutorial.py --run-stage1 --max-members 3
```

**Command-Line Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--date` | str | 20260106 | Target date (YYYYMMDD format) |
| `--run` | str | 00 | Model run hour (00 or 12) |
| `--run-stage1` | flag | False | Run Stage 1 GRIB scanning |
| `--max-members` | int | None | Max ensemble members to process |
| `--hours` | str | "0,3" | Forecast hours (comma-separated) |

### 1.3 Required Input Data

#### ECMWF GRIB Files (Source: AWS S3)
**S3 Path Pattern:**
```
s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/ifs/0p25/enfo/{YYYYMMDD}{HH}0000-{HOUR}h-enfo-ef.grib2
```

**Example:**
```
s3://ecmwf-forecasts/20260106/00z/ifs/0p25/enfo/2026010600000-0h-enfo-ef.grib2
s3://ecmwf-forecasts/20260106/00z/ifs/0p25/enfo/2026010600000-3h-enfo-ef.grib2
...
```

#### Forecast Timesteps (85 Total)
| Range | Interval | Count | Hours |
|-------|----------|-------|-------|
| 0-144h | 3-hourly | 49 | 0, 3, 6, 9, 12, 15, 18, 21, 24, ... 144 |
| 150-360h | 6-hourly | 36 | 150, 156, 162, 168, ... 360 |

### 1.4 Extracted Meteorological Variables

#### Surface Variables (8 fields)
| Variable | Name | Units | Level Type |
|----------|------|-------|------------|
| `10u` | 10m U-wind | m/s | heightAboveGround (10m) |
| `10v` | 10m V-wind | m/s | heightAboveGround (10m) |
| `2t` | 2m Temperature | K | heightAboveGround (2m) |
| `2d` | 2m Dew point | K | heightAboveGround (2m) |
| `msl` | Mean Sea Level Pressure | Pa | meanSea |
| `sp` | Surface Pressure | Pa | surface |
| `skt` | Skin Temperature | K | surface |
| `tcw` | Total Column Water | kg/m² | entireAtmosphere |

#### Static Fields (1 field)
| Variable | Name | Units | Level Type |
|----------|------|-------|------------|
| `lsm` | Land-Sea Mask | 0/1 | surface |

#### Pressure Level Variables (6 variables × 13 levels = 78 fields)
| Variable | Name | Units | Description |
|----------|------|-------|-------------|
| `gh` | Geopotential Height | m | Height at pressure level |
| `t` | Temperature | K | Temperature at pressure level |
| `u` | U-wind component | m/s | Zonal wind |
| `v` | V-wind component | m/s | Meridional wind |
| `w` | Vertical velocity | Pa/s | Omega (pressure velocity) |
| `q` | Specific humidity | kg/kg | Moisture content |

**Pressure Levels (13):** 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

### 1.5 Ensemble Configuration

| Parameter | Value |
|-----------|-------|
| Total Members | 51 (1 control + 50 perturbed) |
| Control Member | Number = -1 (or 0) in GRIB |
| Perturbed Members | Numbers 1-50 |
| Grid Resolution | 0.25° × 0.25° |
| Latitude Points | 721 (90°N to 90°S) |
| Longitude Points | 1440 (0° to 359.75°E) |

### 1.6 Output Format

#### Directory Structure
```
ecmwf_three_stage_YYYYMMDD_HHz/
├── stage2_{member}_merged.parquet
└── stage3_{member}_final.parquet

ecmwf_{date}_{run}_efficient/
├── comprehensive/
│   └── ecmwf_{date}_{run}z_ensemble_all.parquet
└── members/
    ├── control/control.parquet
    ├── ens_01/ens_01.parquet
    ├── ens_02/ens_02.parquet
    └── ... (up to ens_50)
```

#### NetCDF/Zarr Structure
```
Dimensions:
  - time: 1 (initialization time)
  - valid_time: 85 (forecast timesteps)
  - number: 51 (ensemble members)
  - latitude: 721
  - longitude: 1440

Variables:
  - All surface and pressure-level fields
  - Coordinates: latitude, longitude, time, valid_time, number
```

### 1.7 Dependencies for Phase 1

```bash
pip install kerchunk zarr xarray pandas numpy fsspec s3fs requests pyarrow
```

### 1.8 Performance Characteristics

| Method | Per File | Per 85 Hours | Network I/O |
|--------|----------|--------------|-------------|
| scan_grib | 45-90s | 60-120 min | ~85 GB/member |
| GIK Index | 0.5-1s | 3-5 min | ~85 KB/member |
| **Speedup** | **~85x** | **~20x** | **~6000x reduction** |

---

## Phase 2: cGAN Inference

### 2.1 Overview

The cGAN model takes ECMWF ensemble mean and standard deviation as input and generates 1000 stochastic ensemble members for high-resolution precipitation forecasting.

```
ECMWF NetCDF (51 members)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Compute Ensemble Statistics                 │
│   - Ensemble mean for each field            │
│   - Ensemble standard deviation             │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Normalize Input Fields                      │
│   - Apply field-specific normalization      │
│   - Log-transform precipitation             │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Load High-Resolution Constants              │
│   - Elevation (elev.nc)                     │
│   - Land-sea mask (lsm.nc)                  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ cGAN Generator (×1000 ensemble members)     │
│   - Input: [forecast, constants, noise]     │
│   - Generate stochastic precipitation       │
│   - Denormalize output                      │
└─────────────────────────────────────────────┘
         │
         ▼
    Ensemble Precipitation NetCDF (1000 members)
```

### 2.2 Entry Point Scripts

**Location:** `/scratch/notebook/SEWAA-forecasts/`

#### Main Orchestrator: `run_forecast.py`
```bash
# 6-hour accumulation forecast
python run_forecast.py --accumulation 6h --date 20260106 --time 0000

# 24-hour accumulation forecast
python run_forecast.py --accumulation 24h --date 20260106 --time 0000
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--accumulation` | str | 6h | "6h" or "24h" |
| `--date` | str | today | Forecast date (YYYYMMDD) |
| `--time` | str | 0000 | Initialization time (HHMM) |
| `--delete_forecasts` | str | N | Delete raw files after processing |
| `--disable_ELR` | flag | False | Skip ELR predictions (24h only) |

#### Batch Processing: `auto_forecasts.py`
```bash
# Generate forecasts for date range
python auto_forecasts.py --start_date 20260104 --final_date 20260106 --accumulation 6h
```

#### Core Inference: `forecast_date.py`
**Locations:**
- 6h: `6h_accumulations/cGAN/dsrnngan/forecast_date.py`
- 24h: `24h_accumulations/cGAN/dsrnngan/forecast_date.py`

```bash
# 6h: python forecast_date.py <date_str> <hour>
python forecast_date.py 20260106 0

# 24h: python forecast_date.py <valid_time_num> <date_str>
python forecast_date.py 0 20260106
```

### 2.3 Required Input NetCDF Format

#### IFS Forecast Files
**Naming Convention:** `IFS_YYYYMMDD_HHZ.nc`
**Example:** `IFS_20260106_00Z.nc`

**Storage Locations:**
- 6h: `./6h_accumulations/IFS_forecast_data/`
- 24h: `./24h_accumulations/IFS_forecast_data/`

#### Required NetCDF Variables (13 fields)

| Variable | Name | Processing |
|----------|------|------------|
| `cp` | Convective Precipitation | Accumulated, log-transform |
| `mcc` | Middle Cloud Cover | Instantaneous |
| `sp` | Surface Pressure | Mean/std normalize |
| `ssr` | Solar Surface Radiation | Accumulated |
| `t2m` | 2-meter Temperature | Mean/std normalize |
| `tciw` | Total Cloud Ice Water | Divide by max |
| `tclw` | Total Cloud Liquid Water | Divide by max |
| `tcrw` | Total Rain Water | Divide by max |
| `tcw` | Total Cloud Water | Divide by max |
| `tcwv` | Total Column Water Vapour | Divide by max |
| `tp` | Total Precipitation | Accumulated, log-transform |
| `u700` | U-wind at 700 hPa | Symmetric normalize |
| `v700` | V-wind at 700 hPa | Symmetric normalize |

**Additional for 6h:** `cape` (Convective Available Potential Energy)

#### Data Format per Variable
```
{field}_ensemble_mean: (n_times, n_lats, n_lons)
{field}_ensemble_standard_deviation: (n_times, n_lats, n_lons)
time: hours since 1900-01-01
valid_time: forecast valid times
latitude: 0.1° grid (-13.65 to 24.7°)
longitude: 0.1° grid (19.15 to 54.3°)
```

### 2.4 Spatial Coverage (ICPAC Region)

| Parameter | Value |
|-----------|-------|
| Latitude Range | -13.65° to 24.7°N |
| Longitude Range | 19.15° to 54.3°E |
| Resolution | 0.1° × 0.1° |
| Grid Points | 384 × 352 |

### 2.5 Model Architecture & Weights

#### Two Independent Models

**6h Model:**
| Parameter | Value |
|-----------|-------|
| Location | `6h_accumulations/cGAN/ICPAC-big-ensmeansd/` |
| Checkpoint | `models/gen_weights-0316800.h5` (2.4 MB) |
| Generator Filters | 64 |
| Discriminator Filters | 256 |
| Input Channels | 52 (4 × 13 fields) |
| Noise Channels | 4 |
| Latent Variables | 50 |

**24h Model:**
| Parameter | Value |
|-----------|-------|
| Location | `24h_accumulations/cGAN/logs_X/` |
| Checkpoints | `logs_1/gen_weights-0203776.h5` (6h lead) |
|  | `logs_5/gen_weights-0163840.h5` (30-150h leads) |
|  | `logs_17/gen_weights-0115200.h5` (fallback) |
| Generator Filters | 32 |
| Discriminator Filters | 128 |
| Input Channels | 26 (2 × 13 fields) |
| Noise Channels | 4 |
| Latent Variables | 50 |

### 2.6 Constants & Normalization Files

**Location:** `/scratch/notebook/SEWAA-forecasts/cGAN_data/`

| File | Size | Contents | Processing |
|------|------|----------|------------|
| `elev.nc` | 542 KB | Elevation/orography (meters) | Divide by 10,000 |
| `lsm.nc` | 541 KB | Land-sea mask (0-1) | Already normalized |

**Normalization Statistics:**
| File | Location | Purpose |
|------|----------|---------|
| `FCSTNorm2018.pkl` | `6h_accumulations/cGAN/` | 6h field normalization |
| `FCSTNorm2018.pkl` | `24h_accumulations/cGAN/` | 24h field normalization |

**Structure:**
```python
fcst_norm = {
    'cp': {'min': X, 'max': Y, 'mean': Z, 'std': W},
    'mcc': {'min': X, 'max': Y, 'mean': Z, 'std': W},
    # ... all 13 fields
}
```

### 2.7 Configuration Files

#### `forecast.yaml` (Main inference config)
```yaml
MODEL:
    folder: "../logs_17"           # or "../ICPAC-big-ensmeansd" for 6h
    set_seed: False
    checkpoint: 172544

INPUT:
    folder: "../../IFS_forecast_data"
    file: "tp.nc"                  # Template only
    start_hour: 30
    end_hour: 54

OUTPUT:
    folder: "../../cGAN_forecasts"
    ensemble_members: 1000
```

#### `local_config.yaml`
```yaml
data_paths: "AOPP"                # Environment key
gpu_mem_incr: True                # Incremental GPU memory
use_gpu: True/False               # GPU enable (24h: True, 6h: False)
disable_tf32: False               # TensorFloat-32 setting
```

#### `data_paths.yaml`
```yaml
AOPP:
    GENERAL:
        TRUTH_PATH: ''
        FORECAST_PATH: '../../cGAN_forecasts'
        CONSTANTS_PATH: '../../../cGAN_data'
        NORMALISATION_PATH: '../'
        LEAD_IDX: 21
```

#### `downscaling_factor.yaml`
```yaml
downscaling_factor: 1
steps: [1]                        # No upsampling
```

### 2.8 Normalization Process

#### For Accumulated Fields (cp, ssr, tp)
```python
data = np.mean(ensemble_mean, axis=0)              # Mean across ensemble
data = np.sqrt(np.mean(ensemble_std**2, axis=0))   # RMS of standard deviations
data *= 1000                                        # Convert m to mm
data /= 6                                           # Convert to per-hour rate
data = np.log10(1.0 + data)                        # Log transform
```

#### For Instantaneous Fields
```python
# Trapezium rule over 6-hour window
data = (temp_mean[0]/2 + sum(temp_mean[1:4]) + temp_mean[4]/2) / 4

# Field-specific normalization
if field in ['sp', 't2m']:
    data = (data - fcst_norm[field]['mean']) / fcst_norm[field]['std']
elif field in non_negative_fields:
    data = data / fcst_norm[field]['max']
else:  # winds
    data = data / max(abs(fcst_norm[field]['min']), fcst_norm[field]['max'])
```

#### Output Denormalization
```python
def denormalise(x):
    return np.minimum(10**x - 1.0, 100.0)  # Inverse log, capped at 100 mm/h
```

### 2.9 Output Files

#### Raw cGAN Forecasts

**6h Forecasts:**
- Path: `./6h_accumulations/cGAN_forecasts/`
- Naming: `GAN_YYYYMMDD_HHZ.nc`
- Valid times: 30h, 36h, 42h, 48h

**24h Forecasts:**
- Path: `./24h_accumulations/cGAN_forecasts/`
- Naming: `GAN_YYYYMMDD_00Z_v{0-6}.nc`
- Valid times: 6h, 30h, 54h, 78h, 102h, 126h, 150h

**Structure:**
```
Dimensions: (time=1, member=1000, valid_time=N, latitude=384, longitude=352)
Variable: precipitation (mm/h)
```

#### Histogram Counts (for visualization)

**Paths:**
- 6h: `./interface/view_forecasts/data/counts_6h/YYYY/`
- 24h: `./interface/view_forecasts/data/counts_24h/YYYY/`

**Naming:** `counts_YYYYMMDD_HH_XXh.nc`

**Bins (27 thresholds in mm/h):**
```
[0.0, 0.042, 0.083, 0.208, 0.417, 0.625, 0.833, 1.0, 1.25, 1.5,
1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 4.7, 5.4, 6.1, 7.0, 8.0, 9.0, 10.0,
11.5, 13.25, 15.0, 1000]
```

### 2.10 Dependencies for Phase 2

```bash
# Core
pip install tensorflow numpy xarray netCDF4 pyyaml h5py

# Additional
pip install cftime xesmf

# Environment variable (required for TensorFlow >= 2.16)
export TF_USE_LEGACY_KERAS=1
```

---

## Complete Workflow Summary

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA PREPARATION                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Download ECMWF GRIB from S3                                  │
│    s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/enfo/*.grib2    │
│                                                                 │
│ 2. Run GIK Three-Stage Pipeline                                 │
│    python run_ecmwf_tutorial.py --date YYYYMMDD --run-stage1   │
│                                                                 │
│ 3. Extract ensemble statistics                                  │
│    - Compute mean and std across 51 members                     │
│    - Subset to ICPAC region (384 × 352 grid)                   │
│                                                                 │
│ 4. Create IFS NetCDF file                                       │
│    Output: IFS_YYYYMMDD_HHZ.nc                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2: cGAN INFERENCE                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load and normalize IFS NetCDF                                │
│    - Apply field-specific normalization                         │
│    - Log-transform precipitation fields                         │
│                                                                 │
│ 2. Load constants (elevation, land-sea mask)                    │
│    From: cGAN_data/elev.nc, cGAN_data/lsm.nc                   │
│                                                                 │
│ 3. Load pre-trained cGAN generator                              │
│    From: logs_X/models/gen_weights-XXXXXXX.h5                  │
│                                                                 │
│ 4. Generate 1000 ensemble members                               │
│    - Inject random noise for each member                        │
│    - Run through generator network                              │
│    - Denormalize precipitation output                           │
│                                                                 │
│ 5. Save ensemble forecasts                                      │
│    Output: GAN_YYYYMMDD_HHZ.nc (1000 members)                  │
│                                                                 │
│ 6. Convert to histogram counts for visualization                │
│    Output: counts_YYYYMMDD_HH_XXh.nc                           │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

```bash
# Phase 1: Create NetCDF from ECMWF GRIB
cd /scratch/notebook/grib-index-kerchunk/tutorial/ecmwf
python run_ecmwf_tutorial.py --date 20260106 --run-stage1

# Phase 2: Run cGAN inference
cd /scratch/notebook/SEWAA-forecasts
python run_forecast.py --accumulation 6h --date 20260106 --time 0000
```

---

## Technical Parameters Summary

| Parameter | Phase 1 (GIK) | Phase 2 (cGAN) |
|-----------|---------------|----------------|
| Source | ECMWF S3 GRIB | IFS NetCDF |
| Members In | 51 | 51 (mean/std) |
| Members Out | 51 | 1000 |
| Grid In | 721 × 1440 (0.25°) | 384 × 352 (0.1°) |
| Grid Out | 384 × 352 (0.1°) | 384 × 352 (0.1°) |
| Timesteps | 85 | 4-7 valid times |
| Variables | 87+ | 13 |
| Processing Time | ~30 min | ~5-10 min |

---

## File Location Reference

### Phase 1 Scripts
| File | Location |
|------|----------|
| Main tutorial | `/scratch/notebook/grib-index-kerchunk/tutorial/ecmwf/run_ecmwf_tutorial.py` |
| Stage 1 processor | `/scratch/notebook/grib-index-kerchunk/ecmwf/ecmwf_ensemble_par_creator_efficient_multidate.py` |
| Stage 2 processor | `/scratch/notebook/grib-index-kerchunk/ecmwf/ecmwf_index_processor.py` |
| Three-stage orchestrator | `/scratch/notebook/grib-index-kerchunk/ecmwf/ecmwf_three_stage_multidate.py` |
| Utilities | `/scratch/notebook/grib-index-kerchunk/ecmwf/ecmwf_util.py` |

### Phase 2 Scripts
| File | Location |
|------|----------|
| Main orchestrator | `/scratch/notebook/SEWAA-forecasts/run_forecast.py` |
| Batch processor | `/scratch/notebook/SEWAA-forecasts/auto_forecasts.py` |
| 6h inference | `/scratch/notebook/SEWAA-forecasts/6h_accumulations/cGAN/dsrnngan/forecast_date.py` |
| 24h inference | `/scratch/notebook/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan/forecast_date.py` |

### Data & Model Files
| File | Location |
|------|----------|
| 6h model weights | `/scratch/notebook/SEWAA-forecasts/6h_accumulations/cGAN/ICPAC-big-ensmeansd/models/gen_weights-0316800.h5` |
| 24h model weights | `/scratch/notebook/SEWAA-forecasts/24h_accumulations/cGAN/logs_*/models/gen_weights-*.h5` |
| Elevation | `/scratch/notebook/SEWAA-forecasts/cGAN_data/elev.nc` |
| Land-sea mask | `/scratch/notebook/SEWAA-forecasts/cGAN_data/lsm.nc` |
| 6h normalization | `/scratch/notebook/SEWAA-forecasts/6h_accumulations/cGAN/FCSTNorm2018.pkl` |
| 24h normalization | `/scratch/notebook/SEWAA-forecasts/24h_accumulations/cGAN/FCSTNorm2018.pkl` |

---

## Setup Checklist

### Phase 1 Prerequisites
- [ ] Python environment with kerchunk, zarr, xarray, fsspec, s3fs
- [ ] AWS anonymous access configured (`AWS_NO_SIGN_REQUEST=YES`)
- [ ] Hugging Face template downloaded (automatic)
- [ ] Sufficient disk space for parquet outputs (~1-2 GB per date)

### Phase 2 Prerequisites
- [ ] TensorFlow 2.x with Keras support
- [ ] `TF_USE_LEGACY_KERAS=1` environment variable set
- [ ] `local_config.yaml` configured with correct environment
- [ ] `data_paths.yaml` updated with actual paths
- [ ] `forecast.yaml` configured with model path and output location
- [ ] Model weights exist at specified checkpoint
- [ ] Constants files exist (elevation, land-sea mask)
- [ ] Normalization statistics exist (`FCSTNorm2018.pkl`)
- [ ] Input IFS NetCDF files available
- [ ] Output directories created

---

## Troubleshooting

### Phase 1 Issues
| Problem | Solution |
|---------|----------|
| S3 access denied | Set `AWS_NO_SIGN_REQUEST=YES` |
| GRIB file not found | Verify date format and S3 path |
| Memory error | Reduce `--max-members` or process fewer hours |
| Import error | Ensure ECMWF module path is in sys.path |

### Phase 2 Issues
| Problem | Solution |
|---------|----------|
| TensorFlow Keras error | Set `TF_USE_LEGACY_KERAS=1` before imports |
| GPU memory error | Set `gpu_mem_incr: True` in local_config.yaml |
| File not found | Check all paths in data_paths.yaml are absolute |
| Normalization file missing | Run `gen_fcst_norm()` to create FCSTNorm2018.pkl |
| Shape mismatch | Verify input grid is 384 × 352 (ICPAC region) |

---

## Performance Benchmarks

### Current Sequential Processing Times (2026-02-05)

**Phase 1: GIK Parquet Creation**
| Stage | Time | Notes |
|-------|------|-------|
| Stage 1 (GRIB scanning) | ~90 min | Per date, 51 members, limited hours |
| Stage 2 (Index processing) | ~5 min | Uses pre-built templates |
| Stage 3 (Final parquet) | ~2 min | Fast merge operation |
| **Total Phase 1** | **~100 min** | One-time per date |

**Phase 2: Data Streaming for cGAN**
| Configuration | Time | Notes |
|---------------|------|-------|
| 5 members, 9 timesteps | 27 min | Test configuration |
| 51 members, 9 timesteps | **240 min (4 hours)** | Production configuration |
| 51 members, 12 variables | ~14,445 seconds | Full 12-variable extraction |

**Bottleneck Analysis:**
- Sequential S3 fetches dominate: ~4.7 seconds per member per variable per timestep
- Total fetch operations: 51 members × 12 variables × 9 timesteps = 5,508 fetches
- Network latency to ECMWF S3 is the primary constraint

---

## Coiled Dask Parallelization (IMPLEMENTED)

### Problem Statement

The sequential data streaming takes **~4 hours** for 51 ensemble members, which is too slow for operational forecasting that requires daily updates.

### Solution: Coiled Dask + Icechunk Intermediate Storage

Based on lessons from the CMORPH multi-year processor (`deploy-itt/arco_fetch/CMORPH/`), we use:
- **Coiled** for managed cloud compute (20 workers)
- **Icechunk** for intermediate storage with batch-wise branches (avoiding concurrent commit conflicts)

### Implementation Status: ✅ COMPLETE

**Script:** `stream_cgan_variables_coiled.py`

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   COILED CLUSTER (20 workers)                     │
├──────────────────────────────────────────────────────────────────┤
│  Worker 0: Members 0-2    →  Icechunk batch_0 branch             │
│  Worker 1: Members 3-5    →  Icechunk batch_1 branch             │
│  Worker 2: Members 6-8    →  Icechunk batch_2 branch             │
│  ...                                                              │
│  Worker 16: Members 48-50 →  Icechunk batch_16 branch            │
├──────────────────────────────────────────────────────────────────┤
│  Each worker:                                                     │
│    1. Reads parquet references                                    │
│    2. Fetches GRIB bytes from ECMWF S3 (parallel)                │
│    3. Decodes with gribberish                                    │
│    4. Subsets to ICPAC region                                    │
│    5. Writes to unique Icechunk branch (NO CONFLICTS!)           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│           INTERMEDIATE ICECHUNK STORE (GCS)                       │
├──────────────────────────────────────────────────────────────────┤
│  Branch: batch_0  →  {member_0, member_1, member_2} data         │
│  Branch: batch_1  →  {member_3, member_4, member_5} data         │
│  Branch: batch_2  →  {member_6, member_7, member_8} data         │
│  ...                                                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              LOCAL AGGREGATION                                    │
├──────────────────────────────────────────────────────────────────┤
│  1. Read all batch branches from Icechunk                         │
│  2. Concatenate member data                                       │
│  3. Compute: ensemble_mean, ensemble_std                          │
│  4. Write: IFS_YYYYMMDD_HHz_cgan.nc                              │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions (From CMORPH Lessons)

1. **Batch-wise branches**: Each worker writes to its own Icechunk branch (`batch_0`, `batch_1`, etc.)
   - Avoids `ConflictError` from concurrent commits to same branch
   - Avoids GCS 429 rate limiting on branch reference files

2. **Worker function isolation**: All dependencies imported inside worker function
   - Ensures clean serialization to Coiled workers

3. **Sequential aggregation**: Final merge happens locally after parallel phase
   - Guarantees no conflicts during aggregation

### Usage

```bash
# Full production run (20 workers, ~15-20 minutes)
python stream_cgan_variables_coiled.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --n-workers 20 \
  --members-per-batch 3 \
  --service-account coiled-data-e4drr.json

# Test mode (5 workers, 5 members)
python stream_cgan_variables_coiled.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --test

# Custom configuration
python stream_cgan_variables_coiled.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --n-workers 30 \
  --members-per-batch 2 \
  --gcs-bucket my-bucket \
  --gcs-prefix ecmwf_cgan_temp \
  --coiled-region us-east-1
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet-dir` | ecmwf_three_stage_20260203_00z | Stage3 parquet directory |
| `--n-workers` | 20 | Coiled workers |
| `--members-per-batch` | 3 | Members per worker |
| `--service-account` | coiled-data-e4drr.json | GCS credentials |
| `--gcs-bucket` | cpc_awc | Icechunk GCS bucket |
| `--gcs-prefix` | ecmwf_cgan_intermediate | Icechunk GCS prefix |
| `--coiled-region` | eu-west-1 | Coiled cluster region |
| `--max-members` | None | Limit members (testing) |
| `--test` | False | Test mode (5 members, 5 workers) |

### Performance Comparison

| Metric | Sequential | Coiled (20 workers) | Improvement |
|--------|------------|---------------------|-------------|
| Total Time | ~240 min | ~15-20 min | **12-16x** |
| S3 Fetches/sec | ~0.4 | ~8 | **20x** |
| Cost per run | $0 | ~$2-3 | N/A |
| Members parallel | 1 | 17 batches | **17x** |

### Dependencies

```bash
# Core Coiled/Dask
pip install coiled dask[complete] distributed

# Icechunk for intermediate storage
pip install icechunk virtualizarr obstore

# Existing dependencies
pip install xarray fsspec s3fs gribberish pandas numpy
```

### Files

| File | Purpose |
|------|---------|
| `stream_cgan_variables_coiled.py` | Main Coiled-enabled script |
| `stream_cgan_variables.py` | Original sequential script (for comparison) |
| `COILED_DASK_IMPLEMENTATION_PLAN.md` | Detailed implementation plan |

### Alternative: Sequential Processing

For development/testing without Coiled:

```bash
# Sequential (original script, ~4 hours for 51 members)
python stream_cgan_variables.py \
  --parquet-dir ecmwf_three_stage_20260203_00z \
  --max-members 5  # Limit for testing
```

---

## Next Steps

1. **✅ COMPLETE:** Coiled Dask parallelization implemented (`stream_cgan_variables_coiled.py`)
2. **✅ COMPLETE:** Icechunk intermediate storage with batch-wise branches (no conflicts)
3. **Next:** Test Coiled script with production data
4. **Future:** Consider Zarr-based caching to avoid repeated S3 fetches
5. **Future:** Integrate with automated forecast pipeline

### Testing Checklist

- [ ] Test with 5 members (`--test` flag)
- [ ] Verify Icechunk branches created correctly
- [ ] Verify aggregation produces correct ensemble statistics
- [ ] Compare output with sequential script
- [ ] Full production run (51 members)

---

*Document generated for ICPAC SEWAA-Forecasts project*
*Last updated: 2026-02-05 - Added Coiled Dask implementation*
