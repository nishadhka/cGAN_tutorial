# cGAN GEFS Forecast Workflow Documentation

## Overview

This document describes the data pipeline for running cGAN (conditional Generative Adversarial Network) precipitation downscaling inference using GEFS (Global Ensemble Forecast System) ensemble forecasts, as implemented in `example_notebooks/make_forecast_gefs.ipynb`.

---

## 1. Required GEFS Variables

The cGAN model requires **8 meteorological variables** from GEFS forecasts, defined in `data/data_gefs.py:25`:

| cGAN Variable | GEFS Variable | Description | Units |
|---------------|---------------|-------------|-------|
| `cape` | `cape` | Convective Available Potential Energy | J/kg |
| `pres` | `sp` | Surface Pressure | Pa |
| `msl` | `mslet` | Mean Sea Level Pressure | Pa |
| `pwat` | `pwat` | Precipitable Water | kg/m² |
| `tmp` | `t2m` | 2-meter Temperature | K |
| `ugrd` | `u10` | 10-meter U-wind Component | m/s |
| `vgrd` | `v10` | 10-meter V-wind Component | m/s |
| `apcp` | `tp` | Accumulated Precipitation (6-hourly) | mm |

These are stored in `all_fcst_fields` at `data/data_gefs.py:25`:
```python
all_fcst_fields = ['cape', 'pres', 'pwat', 'tmp', 'ugrd', 'vgrd', 'msl', 'apcp']
```

### Variable Classification

**Non-negative fields** (`data/data_gefs.py:26`):
```python
nonnegative_fields = ['cape', 'msl', 'pres', 'pwat', 'tmp']
```

---

## 2. Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE OVERVIEW                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  GEFS GRIB2      │────▶│ process_ensemble_    │────▶│ ensemble_{var}.nc     │
│  Ensemble Files  │     │ by_variable.py       │     │ (31 members)          │
└──────────────────┘     └──────────────────────┘     └───────────────────────┘
                                                               │
                                                               ▼
                         ┌──────────────────────┐     ┌───────────────────────┐
                         │ restructure_for_     │◀────│ Variable Mapping      │
                         │ cgan.py              │     │ (GEFS→cGAN names)     │
                         └──────────────────────┘     └───────────────────────┘
                                   │
                                   ▼
                         ┌───────────────────────┐
                         │ {cgan_var}_{year}.nc  │
                         │ Per-variable NetCDF   │
                         │ (cGAN format)         │
                         └───────────────────────┘
                                   │
                                   ▼
┌──────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ cGAN Generator   │◀────│ forecast_gfs.py       │◀────│ Constants (elev, lsm) │
│ Model (.h5)      │     │ make_fcst()           │     │ + Normalization       │
└──────────────────┘     └───────────────────────┘     └───────────────────────┘
                                   │
                                   ▼
                         ┌───────────────────────┐
                         │ GAN_{YYYYMMDD}.nc     │
                         │ Downscaled Precip     │
                         │ (50 ensemble members) │
                         └───────────────────────┘
```

---

## 3. Input File Requirements

### 3.1 Forecast Input Files

**Location**: Specified in `config/forecast_gfs.yaml` → `INPUT.folder`
```yaml
INPUT:
    folder: "/home/nkalladath_icpac_net/data/netcdf/"
```

**File naming convention**: `{variable}_{year}.nc`
- Example: `cape_2024.nc`, `apcp_2024.nc`, `tmp_2024.nc`

**Expected dimensions** (from `scripts/forecast_gfs.py:217-219`):
```
(time, step, member, latitude, longitude)
```

The script selects 2 consecutive time steps for each forecast window:
```python
nc_file = nc_file.sel({"time": day}).isel({"step": [in_time_idx-5, in_time_idx-4]})
```

### 3.2 Constant Files

**Location**: `CONSTANTS_PATH_GEFS` from `config/data_paths.yaml`

Two constant files are required (`data/data_gefs.py:202-219`):

| File | Variable | Description | Normalization |
|------|----------|-------------|---------------|
| `elev.nc` | `elevation` | Digital elevation model | Divided by 10,000 |
| `lsm.nc` | `lsm` | Land-sea mask | Already 0-1 scaled |

**Output shape**: `(batch_size, 384, 352, 2)`

### 3.3 Normalization Constants

**File**: `FCSTNorm{year}.pkl` (e.g., `FCSTNorm2018.pkl`)
**Location**: `CONSTANTS_PATH_GEFS` directory

Contains per-field statistics:
```python
fcst_norm[field] = {
    "min": ...,
    "max": ...,
    "mean": ...,
    "std": ...
}
```

---

## 4. cGAN Model Files

### 4.1 Generator Weights

**Location**: Specified in `config/forecast_gfs.yaml` → `MODEL.folder`
```yaml
MODEL:
    folder: "/home/nkalladath_icpac_net/data/logfile_gefs_v3/"
    checkpoint: 345600
```

**Weight file path** (constructed in `scripts/forecast_gfs.py:91`):
```python
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
# Example: logfile_gefs_v3/models/gen_weights-0345600.h5
```

### 4.2 Model Configuration

**File**: `setup_params.yaml` (inside `MODEL.folder`)

Required parameters (`scripts/forecast_gfs.py:79-86`):
```yaml
GENERAL:
    mode: "GAN"
MODEL:
    architecture: "forceconv"
    padding: "same"
GENERATOR:
    filters_gen: 64
    noise_channels: 8
    latent_variables: null
DISCRIMINATOR:
    filters_disc: 64
```

---

## 5. Variable Normalization Strategy

The `forecast_gfs.py` script applies different normalization based on variable type (`scripts/forecast_gfs.py:225-253`):

### 5.1 Precipitation (`apcp`)
```python
data = np.log10(1 + data)  # Log transform
# Then compute mean/std across ensemble
```

### 5.2 Pressure/Temperature fields (`msl`, `pres`, `tmp`)
```python
data -= fcst_norm[field]["mean"]
data /= fcst_norm[field]["std"]
# Then compute mean/std across ensemble
```

### 5.3 Non-negative fields (`cape`, `pwat`)
```python
data = np.maximum(data, 0.0)
data /= fcst_norm[field]["max"]
# Then compute mean/std across ensemble
```

### 5.4 Wind components (`ugrd`, `vgrd`)
```python
data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
# Then compute mean/std across ensemble
```

### 5.5 Final Channel Structure

For each variable, 4 channels are created:
```
[mean_t1, std_t1, mean_t2, std_t2]
```

Total input channels: `4 * 8 variables = 32 channels`

---

## 6. How `process_ensemble_by_variable.py` Supplies Variables

The script at `grib-index-kerchunk/gefs/process_ensemble_by_variable.py` processes GEFS ensemble data:

### 6.1 Input Discovery (`line 91-122`)
```python
def discover_ensemble_files(zarr_dir, logger):
    # Finds all .zarr files in directory
    # Extracts ensemble member names (e.g., gep01, gep02, ...)
    # Returns sorted list of (members, files)
```

### 6.2 Variable Processing (`line 154-274`)
```python
def process_variable_ensemble(variable_name, ensemble_members, zarr_files, output_dir, logger):
    # For each ensemble member:
    #   - Open zarr file
    #   - Extract single variable
    #   - Add 'member' dimension
    # Concatenate all members along member dimension
    # Save as: ensemble_{variable_name}.nc
```

### 6.3 Statistics Computation (`line 276-378`)
```python
def compute_statistics_for_variable(variable_name, variable_file, output_dir, logger):
    # Compute ensemble mean and standard deviation
    # Save as:
    #   - ensemble_mean_{variable_name}.nc
    #   - ensemble_std_{variable_name}.nc
```

### 6.4 Output Files

| Output File | Contents |
|-------------|----------|
| `ensemble_{var}.nc` | Full ensemble (31 members) |
| `ensemble_mean_{var}.nc` | Ensemble mean |
| `ensemble_std_{var}.nc` | Ensemble standard deviation |

---

## 7. Data Restructuring for cGAN

The `example_notebooks/restructure_for_cgan.py` script converts processed GEFS data to cGAN format:

### 7.1 Variable Mapping (`line 21-30`)
```python
VARIABLE_MAPPING = {
    'cape': 'cape',
    'sp': 'pres',
    'mslet': 'msl',
    'pwat': 'pwat',
    't2m': 'tmp',
    'u10': 'ugrd',
    'v10': 'vgrd',
    'tp': 'apcp',
}
```

### 7.2 Dimension Transformation
```
Input:  (member, valid_times, latitude, longitude)
Output: (time, member, valid_time, latitude, longitude)
```

### 7.3 Forecast Hour Filtering
Default: Hours 30-54 (every 6 hours)
```python
target_hours = np.arange(start_hour, end_hour + 1, hour_interval)
# Results in: [30, 36, 42, 48, 54]
```

---

## 8. Inference Workflow in `make_forecast_gefs.ipynb`

### 8.1 Initialization
```python
import forecast_gfs
# Loads:
# - Model weights from MODEL.folder
# - Constants (elevation, land-sea mask)
# - Normalization parameters
```

### 8.2 Forecast Generation
```python
forecast_gfs.make_fcst()
```

**Processing per date:**
1. Open variable NetCDF files for the year
2. Select time step and forecast lead times
3. Resize data to model grid (384 x 352)
4. Normalize each variable
5. Concatenate all variables → `(1, 384, 352, 32)`
6. Generate 50 ensemble members using noise injection
7. Denormalize precipitation output
8. Save to NetCDF

### 8.3 Output Format
```
Dimensions: (time, member, valid_time, latitude, longitude)
            (1, 50, 4, 384, 352)

Variables:
- precipitation: mm/h (6-hour average rainfall rate)
- latitude: -13.65 to 24.7 (0.1° resolution)
- longitude: 19.15 to 54.3 (0.1° resolution)
```

---

## 9. Configuration Files Summary

| File | Purpose | Key Parameters |
|------|---------|----------------|
| `config/forecast_gfs.yaml` | Forecast runtime config | Model path, checkpoint, dates, hours |
| `config/local_config.yaml` | Environment settings | GPU mode, data paths selector |
| `config/data_paths.yaml` | Data directory paths | Truth, forecast, constants paths |
| `config/downscaling_factor.yaml` | Model architecture | Upsampling steps |
| `{MODEL}/setup_params.yaml` | Model hyperparameters | Architecture, filters, noise channels |

---

## 10. Directory Structure

```
cGAN_tutorial/
├── config/
│   ├── forecast_gfs.yaml          # Forecast configuration
│   ├── local_config.yaml          # Environment settings
│   ├── data_paths.yaml            # Data path definitions
│   └── downscaling_factor.yaml    # Model scaling config
├── data/
│   └── data_gefs.py               # GEFS data loading functions
├── scripts/
│   └── forecast_gfs.py            # Main forecast script
├── model/
│   ├── gan.py                     # GAN architecture
│   ├── noise.py                   # Noise generator
│   └── models.py                  # Model definitions
├── example_notebooks/
│   ├── make_forecast_gefs.ipynb   # Inference notebook
│   └── restructure_for_cgan.py    # Data restructuring script
└── setupmodel.py                  # Model setup utilities

External Dependencies:
├── {MODEL_FOLDER}/
│   ├── setup_params.yaml          # Model training config
│   └── models/
│       └── gen_weights-*.h5       # Generator weights
├── {CONSTANTS_PATH}/
│   ├── elev.nc                    # Elevation data
│   ├── lsm.nc                     # Land-sea mask
│   └── FCSTNorm{year}.pkl         # Normalization constants
└── {INPUT_FOLDER}/{year}/
    ├── cape_{year}.nc
    ├── pres_{year}.nc
    ├── pwat_{year}.nc
    ├── tmp_{year}.nc
    ├── ugrd_{year}.nc
    ├── vgrd_{year}.nc
    ├── msl_{year}.nc
    └── apcp_{year}.nc
```

---

## 11. Key Code References

| Functionality | File | Line |
|--------------|------|------|
| Variable list | `data/data_gefs.py` | 25 |
| Constants loading | `data/data_gefs.py` | 202-219 |
| Normalization params | `data/data_gefs.py` | 453-462 |
| Model setup | `setupmodel.py` | 10-116 |
| Forecast generation | `scripts/forecast_gfs.py` | 164-273 |
| Variable normalization | `scripts/forecast_gfs.py` | 225-253 |
| GEFS→cGAN mapping | `example_notebooks/restructure_for_cgan.py` | 21-30 |
| Ensemble processing | `process_ensemble_by_variable.py` | 154-274 |
