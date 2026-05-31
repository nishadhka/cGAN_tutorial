# cGAN Ensemble Generation Workflow: From 5+1 GEFS Members to 50 Ensemble Members

## Overview

This document provides an in-depth explanation of how the cGAN inference system transforms input NetCDF files containing 5+1 GEFS (Global Ensemble Forecast System) ensemble members into 50 high-resolution ensemble members through the generative AI model.

## Input Data Structure

### NetCDF File Format
**Example file:** `/home/roller/cgan_gefs_forecast/netcdf/2024/apcp_2024.nc`

**Dimensions:**
```
dimensions:
    latitude = 155        # Spatial grid points (north-south)
    longitude = 141       # Spatial grid points (east-west)
    member = 5           # GEFS ensemble members (5+1 = control + 5 perturbed)
    step = 5             # Forecast time steps
    time = 1             # Forecast initialization time
```

**Key Variables:**
- `tp(time, member, step, latitude, longitude)` - Total precipitation field
- `valid_time(step)` - Valid forecast times for each step

**Data Shape:** The precipitation data has shape `(1, 5, 5, 155, 141)`:
- 1 forecast initialization time
- 5 ensemble members (GEFS control run + perturbed members)
- 5 time steps (forecast lead times)
- 155 × 141 spatial grid

### Required Input Fields

The system requires 8 meteorological fields (defined in `data_gefs.py:25`):

| Field | Variable Name | Description |
|-------|--------------|-------------|
| `cape` | Convective Available Potential Energy | Atmospheric instability measure |
| `pres` | Pressure | Surface pressure |
| `pwat` | Precipitable Water | Total atmospheric water vapor |
| `tmp` | Temperature | Surface temperature |
| `ugrd` | U-component of wind | East-west wind velocity |
| `vgrd` | V-component of wind | North-south wind velocity |
| `msl` | Mean Sea Level Pressure | Sea level pressure |
| `apcp` | Accumulated Precipitation | 6-hour accumulated rainfall |

Each field is stored in a separate NetCDF file: `{field}_{year}.nc`

## Data Processing Pipeline

### Stage 1: NetCDF File Opening and Reading

**Location:** `forecast_gfs.py:213-218`

```python
# Open NetCDF file for specific field and year
input_file = f"{field}_{d.year}.nc"
nc_in_path = os.path.join(input_folder_year, input_file)
nc_file = xr.open_dataset(nc_in_path)

# Select specific forecast date and time steps
nc_file = nc_file.sel({"time": day}).isel({"step": [in_time_idx-5, in_time_idx-4]})
```

**Key Points:**
- Uses `xarray` to open NetCDF files with lazy loading
- Selects specific forecast initialization date
- Extracts two consecutive time steps (e.g., steps at indices -5 and -4 relative to target time)
- The time indexing allows the model to use temporal context from previous forecast hours

### Stage 2: Ensemble Statistics Calculation

**Location:** `forecast_gfs.py:220-222`

```python
short_name = [var for var in nc_file.data_vars][0]
data = np.moveaxis(np.squeeze(nc_file[short_name].values), 0, -1)
data = tf.constant(data)
data = tf.image.resize(data, [384, 352]).numpy()
```

**Process:**
1. Extract the primary data variable from NetCDF
2. Convert xarray DataArray to NumPy array
3. Reshape dimensions: move time dimension to the end
4. Resize spatial grid to model's expected resolution (384×352) using TensorFlow's image resize
5. Original data shape: `(155, 141, 5_members, 2_timesteps)` → Resized: `(384, 352, 5_members, 2_timesteps)`

**Critical Transformation - From Ensemble Members to Ensemble Statistics:**

The 5+1 GEFS ensemble members are converted into **mean and standard deviation** statistics:

```python
# Calculate ensemble mean and std across members
data_mean = np.moveaxis(np.nanmean(data, axis=-1), 0, -1)  # Average across members
data_std = np.moveaxis(np.nanstd(data, axis=-1), 0, -1)    # Std dev across members

# Concatenate: [mean_t1, std_t1, mean_t2, std_t2]
data = np.concatenate([data_mean[...,[0]], data_std[...,[0]],
                      data_mean[...,[1]], data_std[...,[1]]], axis=-1)
```

**Output:** Each field produces 4 channels:
- Channel 1: Ensemble mean at timestep 1
- Channel 2: Ensemble standard deviation at timestep 1
- Channel 3: Ensemble mean at timestep 2
- Channel 4: Ensemble standard deviation at timestep 2

### Stage 3: Field-Specific Normalization

**Location:** `forecast_gfs.py:224-252`

Different meteorological fields require different normalization approaches:

#### Precipitation Fields (`apcp`):
```python
if field in ["apcp"]:
    data = np.log10(1 + data)  # Log transform to handle extreme values
    # Then calculate mean/std and create 4 channels
```

#### Pressure/Temperature Fields (`msl`, `pres`, `tmp`):
```python
if field in ["msl", "pres", "tmp"]:
    data -= fcst_norm[field]["mean"]  # Center around zero
    data /= fcst_norm[field]["std"]    # Scale to unit variance
    # Then calculate mean/std and create 4 channels
```

#### Non-negative Fields (`cape`, `pwat`):
```python
elif field in nonnegative_fields:
    data = np.maximum(data, 0.0)      # Ensure non-negative
    data /= fcst_norm[field]["max"]    # Normalize by maximum value
    # Then calculate mean/std and create 4 channels
```

#### Wind Fields (`ugrd`, `vgrd`):
```python
elif field in ["ugrd", "vgrd"]:
    data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])  # Symmetric scaling
    # Then calculate mean/std and create 4 channels
```

**Normalization Statistics Source:** Pre-computed from `FCSTNorm2018.pkl` (generated via `generate_fcst_norm.py`)

### Stage 4: Multi-Field Input Concatenation

**Location:** `forecast_gfs.py:255-256`

```python
# Concatenate all 8 fields, each contributing 4 channels
network_fcst_input = np.concatenate(field_arrays, axis=-1)
# Shape: (384, 352, 32)  [32 = 8 fields × 4 channels]

# Add batch dimension
network_fcst_input = np.expand_dims(network_fcst_input, axis=0)
# Final shape: (1, 384, 352, 32)
```

**Input Array Structure:**
- **Shape:** `(batch=1, height=384, width=352, channels=32)`
- **Channels:** 8 fields × 4 statistical channels = 32 total input channels
- **Content:** Normalized ensemble statistics from GEFS forecast

## cGAN Inference: Generating 50 Ensemble Members

### Stage 5: Model Architecture and Inputs

**Location:** `forecast_gfs.py:93-106`

The cGAN generator requires **three inputs**:

1. **Forecast Input** (`network_fcst_input`): Shape `(1, 384, 352, 32)`
   - Low-resolution forecast fields (ensemble statistics)

2. **Constants Input** (`network_const_input`): Shape `(1, 384, 352, 2)`
   - Orography (elevation / 10000): Terrain height normalized
   - Land-sea mask (0-1): Land/ocean indicator

3. **Noise Input**: Shape `(1, 384, 352, noise_channels)`
   - Random Gaussian noise for stochastic generation
   - **This is the key to ensemble diversity!**

```python
gen = model.gen  # Generator network
gen.load_weights(weights_fn)  # Load pre-trained weights

network_const_input = load_hires_constants(batch_size=1)  # Load terrain/LSM
```

### Stage 6: Noise Generation for Ensemble Diversity

**Location:** `forecast_gfs.py:257-258` and `model/noise.py`

```python
# Define noise shape matching spatial dimensions
noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
noise_gen = NoiseGenerator(noise_shape, batch_size=1)
```

**NoiseGenerator Class** (`model/noise.py:4-21`):
```python
class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __call__(self, mean=0.0, std=1.0):
        shape = (self.batch_size,) + self.noise_shapes
        n = self.prng.randn(*shape).astype(np.float32)
        if std != 1.0:
            n *= std
        if mean != 0.0:
            n += mean
        return n
```

**Noise Properties:**
- Samples from Gaussian distribution N(0,1)
- Same spatial dimensions as input (384×352)
- Different noise channels (configurable via `noise_channels` parameter)
- **Each ensemble member uses different random noise**

### Stage 7: Ensemble Member Generation Loop

**Location:** `forecast_gfs.py:261-266`

```python
progbar = Progbar(ensemble_members)  # Progress bar for 50 members

for ii in range(ensemble_members):  # Loop 50 times
    # Create fresh random noise for this member
    gan_inputs = [network_fcst_input, network_const_input, noise_gen()]

    # Generate high-resolution rainfall prediction
    gan_prediction = gen.predict(gan_inputs, verbose=False)
    # Shape: (1, 384, 352, 1)

    # Denormalize and write to output NetCDF
    netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(gan_prediction[0, :, :, 0])

    progbar.add(1)
```

**Critical Process:**
1. **Loop 50 times** - one iteration per ensemble member
2. **Each iteration:**
   - Generate NEW random noise → `noise_gen()` produces different noise each call
   - Combine: forecast + constants + noise
   - Run through generator network → produces rainfall field
   - Denormalize output: `10^x - 1.0` (inverse of log10(1+x) applied earlier)
   - Write to output NetCDF file

**Why Different Noise Creates Different Predictions:**
- Same forecast input + same constants + **different noise** = different rainfall patterns
- The GAN was trained to map noise variations to physically plausible forecast uncertainty
- Noise effectively samples from the learned probability distribution of possible weather outcomes

### Stage 8: Denormalization and Output

**Location:** `data_gefs.py:61-65`

```python
def denormalise(x):
    """
    Undo log-transform of rainfall. Also cap at 100 mm/h.
    """
    return np.minimum(10**x - 1.0, 100.0)
```

**Process:**
- Inverse log transformation: `10^x - 1.0`
- Cap extreme values at 100 mm/h (physically realistic limit)
- Output units: mm/h (millimeters per hour)

## Output NetCDF File Structure

### File Creation

**Location:** `forecast_gfs.py:109-160`

```python
def create_output_file(nc_out_path):
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")

    # Dimensions
    rootgrp.createDimension("latitude", len(latitude))    # 384
    rootgrp.createDimension("longitude", len(longitude))  # 352
    rootgrp.createDimension("member", ensemble_members)   # 50
    rootgrp.createDimension("time", None)                 # Unlimited
    rootgrp.createDimension("valid_time", None)           # Unlimited

    # Main variable: precipitation
    precipitation = rootgrp.createVariable(
        "precipitation", "f4",
        ("time", "member", "valid_time", "latitude", "longitude"),
        compression="zlib",
        chunksizes=(1, 1, 1, len(latitude), len(longitude))
    )
    precipitation.units = "mm h**-1"
    precipitation.long_name = "Precipitation"
```

**Output Dimensions:**
- `time`: Forecast initialization time(s)
- `member`: 50 ensemble members (numbered 1-50)
- `valid_time`: Forecast valid times (from start_hour to end_hour)
- `latitude`: 384 grid points (high-res)
- `longitude`: 352 grid points (high-res)

**Output Location:**
```
{OUTPUT.folder}/test/{YYYY}/GAN_{YYYYMMDD}.nc
```

## Complete Workflow Summary

### Input → Processing → Output

```
INPUT NetCDF (5+1 ensemble members)
    ↓
[Stage 1] Open & Read NetCDF files
    ├─ 8 fields × (5 members × 2 timesteps)
    ↓
[Stage 2] Calculate Ensemble Statistics
    ├─ For each field: compute mean & std across 5 members
    ├─ Result: 8 fields × 4 channels (mean_t1, std_t1, mean_t2, std_t2)
    ↓
[Stage 3] Field-Specific Normalization
    ├─ Log transform for precipitation
    ├─ Standardization for pressure/temperature
    ├─ Max normalization for non-negative fields
    ├─ Symmetric scaling for winds
    ↓
[Stage 4] Concatenate Multi-Field Input
    ├─ Stack all 32 channels: (384, 352, 32)
    ├─ Add constants: elevation + land-sea mask
    ↓
[Stage 5] Load Pre-trained cGAN Model
    ├─ Generator network with learned weights
    ├─ Architecture: conditional GAN for downscaling
    ↓
[Stage 6] Generate Noise for Ensemble
    ├─ Gaussian noise: N(0,1)
    ├─ Shape: (384, 352, noise_channels)
    ↓
[Stage 7] Loop: Generate 50 Members
    ├─ For each member:
    │   ├─ Sample new random noise
    │   ├─ GAN prediction = Gen(forecast + constants + noise)
    │   ├─ Denormalize output
    │   └─ Write to NetCDF
    ↓
[Stage 8] Output NetCDF (50 ensemble members)
    └─ Shape: (time, 50_members, valid_time, 384, 352)
```

## Key Insights: How 5 Members Become 50

### 1. **Ensemble Statistics as Conditional Input**
- The 5+1 GEFS members are **not** directly used as individual inputs
- Instead, they are collapsed into **ensemble mean and standard deviation**
- These statistics capture the uncertainty information from the original ensemble
- The cGAN uses this uncertainty info as conditioning context

### 2. **Noise-Driven Stochasticity**
- The cGAN was trained to translate random noise into realistic forecast variability
- By generating 50 different noise samples, we get 50 different realizations
- Each realization is consistent with the input ensemble statistics
- The noise essentially samples from the learned conditional probability distribution

### 3. **Spatial Downscaling**
- Input: 155×141 grid → Upsampled to 384×352 during preprocessing
- Output: 384×352 high-resolution grid (maintained through GAN)
- The GAN adds realistic fine-scale details that are physically consistent

### 4. **Physical Conditioning**
- Orography (terrain) and land-sea mask ensure predictions respect geography
- Temperature, wind, moisture fields provide atmospheric context
- Multiple time steps capture temporal evolution

### 5. **No Direct Member-to-Member Mapping**
- The 50 output members are **not** simple transformations of the 5 input members
- Instead, the cGAN learns to generate plausible high-resolution fields that:
  - Match the statistics of the input ensemble
  - Respect physical constraints (terrain, land/sea)
  - Capture realistic spatial patterns learned during training

## Technical Configuration

### Configuration Files

**`forecast_gfs.yaml`:**
```yaml
MODEL:
    folder: "/path/to/trained/model/"
    checkpoint: 345600

INPUT:
    folder: "/path/to/input/netcdf/"
    dates: ["2024-04-20"]
    start_hour: 30
    end_hour: 54

OUTPUT:
    folder: "/path/to/output/"
    ensemble_members: 50  # ← Number of ensemble members to generate
```

**Key Parameters:**
- `ensemble_members`: Controls how many times the generation loop runs (default: 50)
- `start_hour`, `end_hour`: Forecast lead time range (in hours, must be divisible by 6)
- `checkpoint`: Which model checkpoint to load for inference

### Model Architecture

**From `setup_params.yaml`:**
- `mode`: "GAN" (not VAE-GAN or deterministic)
- `input_channels`: 32 (4 channels × 8 fields)
- `constant_fields`: 2 (elevation + land-sea mask)
- `noise_channels`: Configurable (e.g., 8 or 16)
- `filters_gen`: Number of filters in generator layers
- `padding`: "SAME" or "VALID" padding for convolutions

## Limitations and Considerations

### 1. **Hard-coded Geographic Region**
- Latitude: -13.65° to 24.7° (ICPAC region)
- Longitude: 19.15° to 54.3°
- Not generalizable to other regions without modification

### 2. **Input File Requirements**
- Expects specific NetCDF structure with 5 ensemble members
- Requires all 8 meteorological fields
- Time steps must align with 6-hour intervals

### 3. **Computational Cost**
- Generating 50 members requires 50 forward passes through the GAN
- GPU recommended for reasonable inference time
- Memory usage scales with ensemble size and spatial resolution

### 4. **Statistical Relationship**
- Output ensemble may not preserve exact input ensemble statistics
- The 50 members capture learned forecast uncertainty patterns
- Ensemble spread depends on model training and noise configuration

## References

**Key Files:**
- `scripts/forecast_gfs.py` - Main inference script
- `data/data_gefs.py` - Data loading and normalization utilities
- `model/noise.py` - Noise generation for ensemble diversity
- `model/gan.py` - GAN architecture and training
- `config/forecast_gfs.yaml` - Inference configuration

**Related Documentation:**
- `CGAN_INFERENCE_SETUP.md` - Setup and technical debt documentation
- `setupmodel.py` - Model initialization utilities
