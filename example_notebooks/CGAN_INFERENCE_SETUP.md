# cGAN Inference Setup Guide

## Overview

This document covers the end-to-end workflow for running cGAN precipitation
downscaling on GEFS data using the GIK (Grib-Index-Kerchunk) pipeline. All
scripts use **PEP 723 inline metadata** so they can be executed directly with
[`uv run`](https://docs.astral.sh/uv/) — no manual environment setup required.

## Quick Start

```bash
# Full pipeline: GIK fetch -> Zarr -> NetCDF -> cGAN inference
uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --output_dir gik_cgan_output

# Run inference only (if NetCDF inputs already exist)
uv run run_gefs_inference_raw.py

# Plot comparison of input vs output
uv run plot_cgan_comparison.py

# Run unit tests
uv run --python 3.11 --with pytest --with "tensorflow==2.15" \
    --with "numpy<2.0" --with xarray --with netcdf4 --with pyyaml \
    --with cftime pytest test_cgan_inference.py -v
```

## Why `uv run`?

The cGAN model requires **TensorFlow 2.15** which only supports Python 3.11.
`uv` handles this automatically:

- Reads PEP 723 `# /// script` metadata from each script
- Downloads CPython 3.11 if not present
- Resolves and installs all dependencies into an isolated environment
- Caches the environment for fast subsequent runs

No conda, micromamba, or virtualenv management needed.

## Pipeline Stages

The unified script `run_gefs_gik_cgan_pipeline.py` runs 5 stages sequentially:

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Download GIK template from Hugging Face | `gik-fmrc-gefs-20241112.tar.gz` |
| 2 | Create parquet references for target date | `gik_cgan_output/parquet_refs/` |
| 3 | Stream GEFS variables from AWS S3 | `gik_cgan_output/zarr_YYYYMMDD_00z/` |
| 4 | Convert Zarr to per-field NetCDF | `gik_cgan_output/netcdf/YYYY/` |
| 5 | Run cGAN inference | `gik_cgan_output/cgan_output/YYYY/GAN_YYYYMMDD.nc` |

### Running specific stages

```bash
# Only GIK reference creation (stages 1-2)
uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --stages 1,2

# Only streaming + conversion (stages 3-4)
uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --stages 3,4

# Only inference (stage 5), if NetCDF data already exists
uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --stages 5 \
    --netcdf_dir gik_cgan_output/netcdf/2025

# Fewer ensemble members for faster testing
uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --max_members 5
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--date` | (required) | Forecast date `YYYYMMDD` |
| `--run` | `00` | Model run hour |
| `--output_dir` | `gik_cgan_pipeline_output` | Base output directory |
| `--stages` | `1,2,3,4,5` | Comma-separated stages to run |
| `--max_members` | `30` | Number of GEFS ensemble members to stream |
| `--model_folder` | `cgan_compact_20260202/logfile_gefs_v3/` | Trained model path |
| `--constants_path` | `cgan_compact_20260202/CONSTANTS/` | Constants directory |
| `--checkpoint` | `345600` | Model checkpoint number |

## Required Files

### Pre-trained Model

Extract from `cgan_compact_20260202.zip`:

```
cgan_compact_20260202/
├── logfile_gefs_v3/
│   ├── setup_params.yaml            # Architecture: forceconv, 128 filters, 4 noise channels
│   └── models/
│       └── gen_weights-0345600.h5   # Generator weights
└── CONSTANTS/
    ├── elev.nc                      # Elevation (metres, normalised /10000 at load time)
    ├── lsm.nc                       # Land-sea mask (0-1)
    ├── FCSTNorm2018.pkl             # IFS normalization stats
    └── FCSTNorm_GEFS_2018.pkl       # Native GEFS normalization stats
```

### Input Forecast Data

Produced by pipeline stages 1-4, or provide manually:

```
{input_folder}/YYYY/
├── cape_YYYY.nc    # (time, member, step, latitude, longitude)
├── pres_YYYY.nc
├── pwat_YYYY.nc
├── tmp_YYYY.nc
├── ugrd_YYYY.nc
├── vgrd_YYYY.nc
├── msl_YYYY.nc
└── apcp_YYYY.nc
```

Fields must be in this exact order for concatenation:
**cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp** (8 fields x 4 channels = 32 input channels).

## Normalization Reference

The following normalization must match the training code (`data/data_gefs.py`)
exactly. Mismatches cause incorrect output (e.g., precipitation over ocean).

### Field normalization

| Field type | Fields | Method | Formula |
|-----------|--------|--------|---------|
| Precipitation | `apcp` | Log transform | `log10(1 + x)` |
| Pressure/Temperature | `msl`, `pres`, `tmp` | Z-score | `(x - mean) / std` |
| Non-negative bounded | `cape`, `pwat` | Max scaling | `x / max` |
| Wind components | `ugrd`, `vgrd` | Symmetric max | `x / max(abs(min), abs(max))` |

After normalization, ensemble mean and std are computed across members,
producing 4 channels per field: `[mean_t1, std_t1, mean_t2, std_t2]`.

### Non-negative fields

```python
nonnegative_fields = ['cape', 'msl', 'pres', 'pwat', 'tmp']
```

These have `np.maximum(data, 0.0)` applied before normalization. Note that
`msl`, `pres`, `tmp` are caught by the z-score branch (earlier in the
if/elif chain) and `cape`, `pwat` fall through to the `/max` branch.

**`apcp` is NOT in this list** — it uses `log10(1+x)` directly.

### Constant fields

| Field | Normalization | Notes |
|-------|--------------|-------|
| Elevation (`elev.nc`) | `/ 10000.0` | Raw metres to O(1) scale |
| Land-sea mask (`lsm.nc`) | None (already 0-1) | Binary land/ocean |

### Output denormalization

```python
def denormalise(data):
    return np.minimum(np.power(10.0, data) - 1.0, 100.0)
```

Inverse of `log10(1 + x)`, capped at 100 mm/h.

## Corrections Applied (Feb 2026)

Three bugs in `run_gefs_inference_raw.py` caused the model to produce
precipitation over ocean instead of over land:

### 1. Elevation not normalised (CRITICAL)

```python
# BEFORE (bug): raw elevation in metres (0-5000)
elev_data = elev[elev_var].values

# AFTER (fix): normalised to O(0.1) matching training code
elev_data = elev[elev_var].values
elev_data = elev_data / 10000.0
```

The model was trained with elevation ~O(0.1), but received ~O(5000) during
inference. This ~50,000x discrepancy in the topographic input was the primary
cause of incorrect spatial patterns.

### 2. Wrong `nonnegative_fields` definition

```python
# BEFORE (bug):
nonnegative_fields = ["cape", "pwat", "apcp"]

# AFTER (fix): matches data/data_gefs.py line 26
nonnegative_fields = ["cape", "msl", "pres", "pwat", "tmp"]
```

### 3. Extra capping in `denormalise()`

```python
# BEFORE: extra log-space cap not in training code
data_capped = np.minimum(data, 10.0)
result = np.power(10.0, data_capped) - 1.0
return np.minimum(np.maximum(result, 0.0), 100.0)

# AFTER: matches training code exactly
return np.minimum(np.power(10.0, data) - 1.0, 100.0)
```

## Unit Tests

The file `test_cgan_inference.py` contains 34 pytest tests that validate
all corrections and guard against regressions. Tests are organised into
7 test classes:

| Class | Tests | What it validates |
|-------|-------|-------------------|
| `TestFieldDefinitions` | 5 | Field order, nonnegative_fields matches training |
| `TestNormalization` | 4 | Normalization branches, input channel count |
| `TestElevationNormalization` | 4 | Elevation /10000 range, LSM [0,1], shape, dtype |
| `TestDenormalise` | 7 | Inverse log transform, 100 mm/h cap, vectorisation |
| `TestNormalizationStats` | 4 | GEFS norm pickle: fields present, plausible ranges |
| `TestModelConfig` | 5 | setup_params.yaml: GAN mode, architecture, filters |
| `TestGridCoordinates` | 5 | ICPAC grid: 384x352, 0.1° resolution, bounds |

### Running the tests

```bash
uv run --python 3.11 --with pytest --with "tensorflow==2.15" \
    --with "numpy<2.0" --with xarray --with netcdf4 --with pyyaml \
    --with cftime pytest test_cgan_inference.py -v
```

### Key tests for the corrections

- `test_nonnegative_fields_match_training` — fails if `nonnegative_fields`
  doesn't match `['cape','msl','pres','pwat','tmp']`
- `test_apcp_not_in_nonnegative_fields` — fails if `apcp` is in the list
- `test_elevation_range` — fails if elevation max > 1.0 (meaning /10000 is missing)
- `test_cap_at_100` / `test_zero_input` / `test_one_input` — validate
  `denormalise()` matches training: `min(10^x - 1, 100)`

### Adding new tests

Tests that only need NumPy (field definitions, denormalise) run fast.
Tests that load constants files or the norm pickle require the model data
directory and will `pytest.skip()` if files are not present.

## Output Format

```
GAN_YYYYMMDD.nc
├── dimensions: (time, member, valid_time, latitude, longitude)
├── precipitation: float32, units "mm h**-1"
├── latitude: 384 points, -13.65° to 24.65° (0.1° resolution)
├── longitude: 352 points, 19.15° to 54.25° (0.1° resolution)
├── member: 1 to 50 (ensemble members)
└── valid_time: 4 steps (+30h, +36h, +42h, +48h)
```

## Troubleshooting

**`uv run` fails with Python ABI mismatch:**
TensorFlow 2.15 requires Python 3.11. The scripts pin `requires-python = "==3.11.*"`.
If uv can't download Python 3.11, install it manually and retry.

**Stage 2 import errors (gefs_util.py):**
Ensure `gefs_util.py` and `stream_gefs_for_cgan.py` are in the same directory
as `run_gefs_gik_cgan_pipeline.py`.

**Stage 3 slow streaming:**
Each member streams ~200 MB from AWS S3. With 30 members, expect ~2 hours.
Use `--max_members 5` for testing.

**Stage 5 slow on CPU:**
Each ensemble member takes ~11s on CPU. With 50 members x 4 valid times
= 200 predictions, expect ~37 minutes. Use a GPU for production runs.

**No GPU found:**
TensorFlow 2.15 GPU requires CUDA 12. The CPU fallback works but is slower.
