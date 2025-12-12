# GEFS cGAN Inference Guide

## Overview

This document describes the GEFS (Global Ensemble Forecast System) precipitation downscaling inference routine using a conditional Generative Adversarial Network (cGAN). The inference script generates ensemble precipitation forecasts by combining GEFS forecast fields with trained model weights.

**Key Script:** `example_notebooks/run_gefs_inference.py`

---

## Table of Contents

1. [Required Input Files](#required-input-files)
2. [Directory Structure](#directory-structure)
3. [Inference Pipeline](#inference-pipeline)
4. [Major Changes from Original Implementation](#major-changes-from-original-implementation)
5. [Comparison with SEWAA-Forecasts Method](#comparison-with-sewaa-forecasts-method)
6. [Configuration](#configuration)
7. [Running the Inference](#running-the-inference)
8. [Output Format](#output-format)
9. [Troubleshooting](#troubleshooting)

---

## Required Input Files

### 1. Model Files

| File | Location | Description |
|------|----------|-------------|
| `setup_params.yaml` | `{model_folder}/` | Model architecture configuration (arch, filters, noise channels, etc.) |
| `gen_weights-XXXXXXX.h5` | `{model_folder}/models/` | Pre-trained generator weights checkpoint |

**Example:** `/home/roller/cgan_gefs_forecast/logfile_gefs_v3/models/gen_weights-0345600.h5`

### 2. Input Forecast Data (NetCDF)

Located in `{input_folder}/{YYYY}/`:

| Field | File Name | Description | Units |
|-------|-----------|-------------|-------|
| `cape` | `cape_{YYYY}.nc` | Convective Available Potential Energy | J/kg |
| `pres` | `pres_{YYYY}.nc` | Surface Pressure | Pa |
| `pwat` | `pwat_{YYYY}.nc` | Precipitable Water | kg/m² |
| `tmp` | `tmp_{YYYY}.nc` | Temperature | K |
| `ugrd` | `ugrd_{YYYY}.nc` | U-component of wind | m/s |
| `vgrd` | `vgrd_{YYYY}.nc` | V-component of wind | m/s |
| `msl` | `msl_{YYYY}.nc` | Mean Sea Level Pressure | Pa |
| `apcp` | `apcp_{YYYY}.nc` | Accumulated Precipitation | mm |

**Example:** `/home/roller/cgan_gefs_forecast/netcdf/2024/cape_2024.nc`

### 3. Constants Files

Located in `{constants_path}/`:

| File | Description |
|------|-------------|
| `elev.nc` | Elevation/orography data (normalized by 10,000) |
| `lsm.nc` | Land-sea mask (0-1) |
| `FCSTNorm2018.pkl` | Normalization statistics for forecast fields |

**Example:** `/home/roller/cgan_gefs_forecast/CONSTANTS/`

---

## Directory Structure

```
cgan_gefs_forecast/
├── logfile_gefs_v3/           # Trained model directory
│   ├── setup_params.yaml      # Model configuration
│   └── models/
│       └── gen_weights-0345600.h5  # Generator weights
├── netcdf/                    # Input forecast data
│   └── 2024/
│       ├── apcp_2024.nc
│       ├── cape_2024.nc
│       ├── msl_2024.nc
│       ├── pres_2024.nc
│       ├── pwat_2024.nc
│       ├── tmp_2024.nc
│       ├── ugrd_2024.nc
│       └── vgrd_2024.nc
├── CONSTANTS/                 # Static data
│   ├── elev.nc
│   ├── lsm.nc
│   └── FCSTNorm2018.pkl
└── predictions/               # Output directory
    └── test/
        └── 2024/
            └── GAN_20240420.nc
```

---

## Inference Pipeline

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INITIALIZATION                                              │
│     ├── Load normalization statistics (FCSTNorm2018.pkl)        │
│     ├── Load constants (elevation, land-sea mask)               │
│     └── Read model setup parameters (setup_params.yaml)         │
│                                                                 │
│  2. MODEL BUILDING                                              │
│     ├── Build generator architecture from parameters            │
│     │   ├── Input layers (forecast, constants, noise)           │
│     │   ├── Residual blocks with custom padding                 │
│     │   ├── Upsampling layers                                   │
│     │   └── Output layer (softplus activation)                  │
│     └── Load pre-trained weights from checkpoint                │
│                                                                 │
│  3. DATA LOADING (per date, per valid time)                     │
│     ├── Load each field from NetCDF files                       │
│     ├── Select time slice and forecast steps                    │
│     ├── Resize to model input size (384 x 352)                  │
│     ├── Apply non-negativity constraints                        │
│     └── Normalize fields (field-specific methods)               │
│                                                                 │
│  4. ENSEMBLE GENERATION                                         │
│     ├── For each ensemble member (1 to N):                      │
│     │   ├── Generate random noise                               │
│     │   ├── Run generator: [forecast, constants, noise] → pred  │
│     │   └── Denormalize output (10^x - 1, capped at 100 mm/h)   │
│     └── Store all members in output array                       │
│                                                                 │
│  5. OUTPUT                                                      │
│     └── Save to NetCDF with dimensions:                         │
│         (time, member, valid_time, latitude, longitude)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Normalization Methods by Field Type

| Field Type | Fields | Normalization Method |
|------------|--------|---------------------|
| Precipitation | `apcp` | `log10(1 + x)`, then compute mean/std across ensemble |
| Pressure/Temperature | `msl`, `pres`, `tmp` | `(x - mean) / std`, then compute mean/std across ensemble |
| Non-negative bounded | `cape`, `pwat` | `x / max`, then compute mean/std across ensemble |
| Wind components | `ugrd`, `vgrd` | `x / max(abs(min), max)`, then compute mean/std across ensemble |

Each field produces 4 channels: `[mean_t1, std_t1, mean_t2, std_t2]` for two consecutive time steps.

---

## Major Changes from Original Implementation

### 1. Keras Import Resolution (Critical Change)

**Problem:** The original `cGAN_tutorial` code uses standalone Keras imports:
```python
# Original (cGAN_tutorial) - BREAKS with TensorFlow 2.15+
from keras.models import Model
from keras.layers import Conv2D, Input, ...
from keras.utils import generic_utils
```

**Solution:** The new script uses `tensorflow.keras` imports:
```python
# New implementation - WORKS with TensorFlow 2.15+
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, ...
from tensorflow.keras.utils import Progbar
```

### 2. Self-Contained Model Definition

**Original:** Imports model from `model/models.py`, `model/blocks.py`, `model/layers.py`
- These files use legacy Keras imports
- Requires `TF_USE_LEGACY_KERAS=1` environment variable
- Dependent on relative path imports (`sys.path.insert`)

**New:** All model components defined within the script:
- Custom layers: `ReflectionPadding2D`, `SymmetricPadding2D`, `Conv2DPadding`
- Building blocks: `residual_block`, `const_upscale_block`
- Full generator architecture: `generator()`

### 3. Field Name Mapping (IFS → GEFS)

**Problem:** Normalization files contain IFS field names, but GEFS uses different names.

**Solution:** Automatic mapping in `load_fcst_norm()`:

| IFS Field | GEFS Field | Description |
|-----------|------------|-------------|
| `sp` | `pres` | Surface pressure |
| `t2m` | `tmp` | Temperature |
| `u700` | `ugrd` | U-wind component |
| `v700` | `vgrd` | V-wind component |
| `tcwv` | `pwat` | Precipitable water |
| `cape` | `cape` | CAPE (same) |
| `tp` | `apcp` | Precipitation |
| (derived) | `msl` | Mean sea level pressure |

### 4. Overflow Protection in Denormalization

**Original:**
```python
def denormalise(x):
    return np.power(10.0, x) - 1.0  # Can overflow!
```

**New:**
```python
def denormalise(data):
    data_capped = np.minimum(data, 10.0)  # Cap log-space values
    result = np.power(10.0, data_capped) - 1.0
    return np.minimum(np.maximum(result, 0.0), 100.0)  # Cap at 100 mm/h
```

### 5. Removed External Dependencies

| Removed | Reason |
|---------|--------|
| `sys.path.insert()` manipulation | Self-contained script |
| `xesmf` (regridding) | Not needed for inference |
| `keras.layers._Merge` | Private class removed in new Keras |
| `keras.utils.generic_utils` | Deprecated module |

---

## Comparison with SEWAA-Forecasts Method

### Architecture Comparison

| Aspect | SEWAA-Forecasts | This Implementation |
|--------|-----------------|---------------------|
| **Location** | `SEWAA-forecasts/6h_accumulations/cGAN/dsrnngan/` | `cGAN_tutorial/example_notebooks/run_gefs_inference.py` |
| **Keras Imports** | `from tensorflow.keras...` | `from tensorflow.keras...` |
| **Model Definition** | Separate files (`models.py`, `blocks.py`, `layers.py`) | Self-contained in single script |
| **Config System** | Uses `forecast_date.py` with YAML configs | Hardcoded CONFIG dict in script |
| **Field Processing** | IFS-specific fields | GEFS-specific fields with IFS mapping |

### Key Similarity: Keras Import Solution

Both implementations solve the Keras import issue the **same fundamental way**:

```python
# Both use tensorflow.keras instead of standalone keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Conv2D, Dense, ...
from tensorflow.keras.utils import Progbar
```

### Key Differences

1. **Self-Containment:**
   - SEWAA: Modular with separate model files
   - This: Single script with all components embedded

2. **Field Handling:**
   - SEWAA: Designed for IFS forecast data
   - This: Designed for GEFS with IFS normalization mapping

3. **Configuration:**
   - SEWAA: External YAML configuration files
   - This: Inline CONFIG dictionary (easily modifiable)

### Is It Fundamentally Different?

**No, it is NOT fundamentally different.** Both approaches:

1. Use the same Keras import pattern (`tensorflow.keras`)
2. Implement the same generator architecture (residual blocks, upsampling)
3. Use the same custom layers (ReflectionPadding2D, etc.)
4. Follow the same inference workflow

The main difference is **packaging**: this implementation consolidates everything into a single portable script for GEFS data, while SEWAA maintains a modular structure for IFS data.

---

## Configuration

Edit the `CONFIG` dictionary in `run_gefs_inference.py`:

```python
CONFIG = {
    "model_folder": "/home/roller/cgan_gefs_forecast/logfile_gefs_v3/",
    "checkpoint": 345600,           # Checkpoint number to load
    "input_folder": "/home/roller/cgan_gefs_forecast/netcdf/",
    "constants_path": "/home/roller/cgan_gefs_forecast/CONSTANTS/",
    "output_folder": "/home/roller/cgan_gefs_forecast/predictions/",
    "dates": ["2024-04-20"],        # List of forecast dates
    "start_hour": 30,               # First forecast hour
    "end_hour": 54,                 # Last forecast hour
    "ensemble_members": 50,         # Number of ensemble members

    # Normalization options (see below)
    "normalization_mode": "gefs",
    "gefs_norm_file": "/home/roller/Downloads/v2FCSTNorm2018.pkl",
}
```

### Normalization Mode Options

The script supports three normalization modes for flexibility with different normalization files:

| Mode | Description | Use Case |
|------|-------------|----------|
| `"gefs"` | Use native GEFS normalization file directly | **Recommended** - Use with `v2FCSTNorm2018.pkl` |
| `"auto"` | Auto-detect field names and map if needed | Fallback mode |
| `"ifs_mapped"` | Use IFS file with field name mapping | Legacy IFS normalization files |

#### Native GEFS Normalization (`v2FCSTNorm2018.pkl`)

The preferred normalization file contains native GEFS field names:

```python
# v2FCSTNorm2018.pkl contents:
{
    'cape': {'min': 0.0, 'max': 6497.0, 'mean': 364.27, 'std': 599.32},
    'hgt':  {'min': -55.34, 'max': 2990.58, 'mean': 562.20, 'std': 507.95},
    'pres': {'min': 70930.13, 'max': 102719.36, 'mean': 94907.64, 'std': 5422.82},
    'pwat': {'min': 0.40, 'max': 83.90, 'mean': 29.94, 'std': 12.98},
    'tmp':  {'min': 266.81, 'max': 325.50, 'mean': 298.84, 'std': 6.09},
    'ugrd': {'min': -43.09, 'max': 43.72, 'mean': -1.01, 'std': 3.25},
    'vgrd': {'min': -41.20, 'max': 43.77, 'mean': 0.54, 'std': 3.85},
    'msl':  {'min': 95088.59, 'max': 103164.02, 'mean': 101139.35, 'std': 437.77}
}
```

**Configuration for native GEFS:**
```python
"normalization_mode": "gefs",
"gefs_norm_file": "/path/to/v2FCSTNorm2018.pkl",
```

#### IFS-Mapped Normalization (Legacy)

For legacy IFS normalization files with field names like `sp`, `t2m`, `u700`, `v700`:

**Configuration for IFS mapping:**
```python
"normalization_mode": "ifs_mapped",
# Uses FCSTNorm2018.pkl from constants_path
```

The mapping converts IFS → GEFS field names:
- `sp` → `pres`
- `t2m` → `tmp`
- `u700` → `ugrd`
- `v700` → `vgrd`
- `tcwv` → `pwat`

### Available Checkpoints

Checkpoints are saved every 19,200 training steps. Available range:
- Minimum: `gen_weights-0019200.h5`
- Maximum: `gen_weights-0480000.h5`
- Recommended: `gen_weights-0345600.h5` (well-trained)

---

## Running the Inference

### Prerequisites

1. **Python Environment:** `tf215gpu` (micromamba/conda)
2. **NumPy Version:** Must be < 2.0 for TensorFlow 2.15 compatibility

```bash
# Ensure NumPy compatibility
micromamba run -n tf215gpu pip install "numpy<2.0"
```

### Execution

```bash
# Run inference
micromamba run -n tf215gpu python /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/example_notebooks/run_gefs_inference.py
```

### Expected Output

```
============================================================
cGAN GEFS Inference
============================================================
Model: /home/roller/cgan_gefs_forecast/logfile_gefs_v3/
Checkpoint: 345600
Dates: ['2024-04-20']
Hours: 30 to 54
Ensemble members: 50
============================================================
No GPU found, using CPU
Loaded normalization with IFS->GEFS field mapping

Building model...
Loading weights: .../gen_weights-0345600.h5
Model loaded successfully!

Processing: 2024-04-20

  Valid time: +30h
50/50 [==============================] - 44s 872ms/step

  Valid time: +36h
50/50 [==============================] - 48s 963ms/step
...

  Output saved: .../predictions/test/2024/GAN_20240420.nc

============================================================
Inference complete!
============================================================
```

---

## Output Format

### NetCDF Structure

```
Dimensions:
    time:       1 (forecast initialization date)
    member:     50 (ensemble members)
    valid_time: 4 (forecast valid times)
    latitude:   384 (ICPAC region)
    longitude:  352 (ICPAC region)

Variables:
    precipitation: (time, member, valid_time, latitude, longitude)
        units: mm h**-1
        long_name: Precipitation

    fcst_valid_time: (time, valid_time)
        units: hours since 1900-01-01

    latitude: (latitude,)
        units: degrees_north
        range: -13.65 to 24.65

    longitude: (longitude,)
        units: degrees_east
        range: 19.15 to 54.25

Attributes:
    description: GAN 6-hour rainfall ensemble members in the ICPAC region.
```

### Output File Naming

```
{output_folder}/test/{YYYY}/GAN_{YYYYMMDD}.nc
```

Example: `/home/roller/cgan_gefs_forecast/predictions/test/2024/GAN_20240420.nc`

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `AttributeError: _ARRAY_API not found` | NumPy 2.x incompatibility | `pip install "numpy<2.0"` |
| `ImportError: cannot import name 'generic_utils'` | Legacy Keras import | Use this script (uses tensorflow.keras) |
| `KeyError: 'pres'` | Missing field in normalization | Script auto-maps IFS→GEFS fields |
| `overflow encountered in power` | Large model outputs | Script caps at 100 mm/h |
| `No GPU found` | Missing CUDA drivers | CPU inference works but slower |

### Verifying Output

```python
import xarray as xr
ds = xr.open_dataset('GAN_20240420.nc')
print(f"Shape: {ds.precipitation.shape}")
print(f"Range: {ds.precipitation.min().values:.2f} to {ds.precipitation.max().values:.2f} mm/h")
print(f"Mean: {ds.precipitation.mean().values:.2f} mm/h")
```

---

## References

- Original cGAN Tutorial: `cGAN_tutorial/` repository
- SEWAA-Forecasts: `SEWAA-forecasts/6h_accumulations/cGAN/dsrnngan/`
- Technical Debt Documentation: `CGAN_INFERENCE_SETUP.md`

---

*Last updated: December 2024*
