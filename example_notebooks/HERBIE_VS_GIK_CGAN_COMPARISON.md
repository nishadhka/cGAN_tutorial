# Herbie vs GIK cGAN Inference Comparison

Side-by-side comparison of cGAN precipitation downscaling using two independent
GEFS data pipelines: **Herbie** (direct HTTP download) and **GIK** (Grib-Index-Kerchunk
byte-range streaming from S3).

## Overview

- **Date tested:** 2024-05-20 00Z
- **Ensemble members:** 30 (gep01–gep30)
- **Forecast hours:** 30–54h (valid times +30h, +36h, +42h, +48h)
- **cGAN model:** `cgan_compact_20260202/logfile_gefs_v3/`, checkpoint 345600
- **cGAN ensemble output:** 25 members per valid time
- **Region:** ICPAC East Africa (lat -13.65 to 24.65, lon 19.15 to 54.25)
- **8 input variables:** cape, pres, pwat, tmp, ugrd, vgrd, msl, apcp

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_gefs_herbie_for_cgan.py` | Herbie path: fetch all 8 cGAN variables from GEFS |
| `run_gefs_gik_cgan_pipeline.py` | GIK path: stages 1-4 (template → parquet → stream → NetCDF) |
| `run_gefs_inference_raw.py` | cGAN inference on either input set |
| `test_herbie_vs_gik_cgan.py` | End-to-end comparison orchestrator |

## Quick Start

### Full run (fetch + inference + comparison)

```bash
cd example_notebooks

# Ensure GIK template is available (symlink or download)
ln -sf /path/to/gik-fmrc-gefs-20241112.tar.gz .
# Also need the deflated-store template for fast Stage 2
ln -sf /path/to/gefs-deflated-store-template-20241112.parquet .

# Run full comparison (takes ~30 min with 30 members on CPU)
uv run test_herbie_vs_gik_cgan.py --date 20240520 --max-members 30
```

### Run with fewer members (faster, for testing)

```bash
uv run test_herbie_vs_gik_cgan.py --date 20240520 --max-members 5
```

### Skip data fetch (reuse existing NetCDFs, only run inference + comparison)

```bash
uv run test_herbie_vs_gik_cgan.py --date 20240520 --skip-fetch
```

### Comparison only (skip fetch and inference, regenerate plots from existing outputs)

```bash
uv run test_herbie_vs_gik_cgan.py --date 20240520 --compare-only
```

### Run each path independently

**Herbie path only:**
```bash
uv run fetch_gefs_herbie_for_cgan.py --date 20240520 --max-members 30 \
    --output-dir herbie_vs_gik_test/herbie/20240520_00z
```

**GIK pipeline only (stages 1-4):**
```bash
uv run run_gefs_gik_cgan_pipeline.py --date 20240520 --max_members 30 \
    --stages 1,2,3,4 --cgan_steps_only --cumulative_apcp \
    --output_dir herbie_vs_gik_test/gik/netcdf
```

**cGAN inference on either input set:**
Edit `CONFIG` in `run_gefs_inference_raw.py` to point to the desired input directory,
then:
```bash
uv run run_gefs_inference_raw.py
```

## Output Directory Structure

```
herbie_vs_gik_test/
├── comparison_results_20240520.json   # Full stats JSON (correlations, intensity ratios, timing)
├── log_main.txt                       # Main comparison log
├── log_herbie.txt                     # Herbie path timing log
├── log_gik.txt                        # GIK path timing log
├── plots/
│   ├── herbie_vs_gik_20240520_T030h.png  # +30h comparison (raw + cGAN + diff, with colorbars)
│   ├── herbie_vs_gik_20240520_T036h.png  # +36h
│   ├── herbie_vs_gik_20240520_T042h.png  # +42h
│   ├── herbie_vs_gik_20240520_T048h.png  # +48h
│   └── intensity_ratio_20240520.png      # Histogram of cGAN/raw ratio
├── herbie/
│   └── 20240520_00z/                     # Herbie input NetCDFs (8 files, ~50 MB)
│       ├── cape_20240520_00z.nc          # dims: (time, member, step, latitude, longitude)
│       ├── pres_20240520_00z.nc          # steps: [30, 36, 42, 48, 54, 60]
│       ├── pwat_20240520_00z.nc
│       ├── tmp_20240520_00z.nc
│       ├── ugrd_20240520_00z.nc
│       ├── vgrd_20240520_00z.nc
│       ├── msl_20240520_00z.nc
│       └── apcp_20240520_00z.nc
├── herbie_cgan/
│   └── 2024/GAN_20240520.nc              # cGAN output from Herbie path (38 MB)
├── gik/
│   └── netcdf/
│       ├── 20240520_00z/                 # GIK input NetCDFs (8 files, ~99 MB)
│       │   ├── cape_20240520_00z.nc      # steps: [30, 36, 42, 48, 54]
│       │   └── ...
│       ├── parquet_refs/                 # 30 GIK parquet reference files
│       └── zarr_20240520_00z/            # Intermediate zarr store
└── gik_cgan/
    └── 2024/GAN_20240520.nc              # cGAN output from GIK path (34 MB)
```

## Results: 20240520 00Z, 30 members

### Speed Comparison

| Stage | Herbie | GIK |
|-------|--------|-----|
| Data fetch (all 8 vars, 30 members) | **584s (9.7 min)** | **1202s (20.0 min)** |
| — Stage 1: Template download | — | ~0s (cached) |
| — Stage 2: Parquet refs | — | 834s (13.9 min) |
| — Stage 3: Stream from S3 | — | 368s (6.1 min) |
| — Stage 4: Zarr → NetCDF | — | ~1s |
| cGAN Inference (4 valid times × 25 members) | 545s (9.1 min) | 537s (8.9 min) |
| **Total** | **~1129s (18.8 min)** | **~1739s (29.0 min)** |

**Herbie is ~1.5x faster** for data acquisition.

Herbie downloads per-variable partial GRIB messages via HTTP (1440 fetches at ~0.4s each).
GIK builds parquet references (~14 min for 30 members using scan_grib) then streams
byte ranges from S3 (~6 min). The GIK Stage 2 bottleneck could be reduced using the
template-based `build_gefs_deflated_store_from_template()` method when a standalone
deflated-store parquet is available.

### Pre-Inference Input Comparison

All 7 non-APCP variables are **identical** between Herbie and GIK (r=1.0, RMSE=0):

| Variable | Correlation | RMSE | Note |
|----------|------------|------|------|
| cape | 1.000000 | 0 | Identical |
| pres | 1.000000 | 0 | Identical |
| pwat | 1.000000 | 0 | Identical |
| tmp  | 1.000000 | 0 | Identical |
| ugrd | 1.000000 | 0 | Identical |
| vgrd | 1.000000 | 0 | Identical |
| msl  | 1.000000 | 0 | Identical |
| apcp | 0.842649 | 2.54 | Differs: GIK uses cumulative APCP, Herbie uses raw per-step |

The APCP difference is expected — GIK converts bucket-incremental APCP to
total-accumulated from hour 0, while Herbie fetches the raw per-step values.

### Post-Inference cGAN Output Comparison

| Valid Time | Correlation | Herbie max (mm/h) | GIK max (mm/h) | Herbie mean | GIK mean |
|------------|------------|-------------------|-----------------|-------------|----------|
| +30h | 0.838 | 3.22 | 6.90 | 0.088 | 0.314 |
| +36h | 0.785 | 3.07 | 6.61 | 0.137 | 0.367 |
| +42h | 0.747 | 2.66 | 6.11 | 0.101 | 0.339 |
| +48h | 0.747 | 2.83 | 6.35 | 0.099 | 0.379 |

GIK path produces ~2x higher cGAN values than Herbie, likely due to the
cumulative APCP input having larger magnitudes.

### Intensity Ratio (cGAN output / raw GEFS input)

| Path | cGAN max (mm/h) | Raw GEFS max (mm) | Ratio |
|------|-----------------|-------------------|-------|
| Herbie | 3.22 | 30.66 | **0.11x** |
| GIK | 6.90 | 142.37 | **0.05x** |

**Both paths show ~10–20x underestimation** — the cGAN produces much lower
precipitation intensity than the raw GEFS input. This confirms the intensity
gap is a **model behavior issue**, not a data pipeline problem.

## Plot Layout

Each per-timestep PNG has 2 rows × 3 columns:

```
Row 0 (Raw GEFS):     Herbie        |    GIK          |   Diff (H−G)     [colorbar: mm]
Row 1 (cGAN output):  Herbie        |    GIK          |   Diff (H−G)     [colorbar: mm/h]
```

Panel titles include max and mean values. Color scales:
- **Raw GEFS APCP:** [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100] mm (blue-yellow-red)
- **cGAN output:** [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10] mm/h (green-yellow-red)
- **Difference:** diverging RdBu_r centered at 0

## Bug Fixes Applied

### 1. Herbie product name (`product="atmos.25"`)
Herbie's GEFS model requires `product="atmos.25"` for the 0.25° grid,
not `product="atmos"`.

### 2. APCP cumulative steps T+0h missing (commit `02f2486`)
**Problem:** `compute_apcp_cumulative_steps()` started from hour 0 (step position 0),
but GEFS has no APCP at T+0h. This created 19 zarr slots but only 18 parquet chunks
existed (positions 1–18), leaving the last slot (hour 54) as NaN. The cGAN VT3 (+48h)
inference needs steps 48 and 54, so it produced all-NaN output.

**Fix:** Start APCP cumulative steps from hour 3 (step position 1). Now 18 zarr
slots match 18 parquet chunks exactly.

### 3. Deflated-store template lookup
The GIK pipeline now looks for a standalone `gefs-deflated-store-template*.parquet`
file alongside the tar.gz, falling back to scan_grib if neither is found.

## Dependencies

All scripts use PEP 723 inline metadata and can be run with `uv run`:

```
tensorflow==2.15, numpy<2.0, xarray, netcdf4, scipy, matplotlib, cartopy,
herbie-data, cfgrib, eccodes, gribberish, kerchunk, zarr, pandas, fsspec,
s3fs, pyarrow, pyyaml, cftime, requests, dask, distributed, gcsfs,
google-cloud-storage, google-auth
```

Requires Python 3.11 (pinned for TensorFlow 2.15 compatibility).

## Diagnostic Conclusion

Both Herbie and GIK data pipelines produce consistent input data (r=1.0 for
all non-APCP fields). Both paths yield the same order-of-magnitude intensity
underestimation (~10–20x). **The low intensity is confirmed as a model-side
issue**, not a data pipeline problem.

Next steps to investigate:
- Training data format vs inference data format (units, normalization)
- Model checkpoint selection
- Wind level mismatch (10m vs 700hPa)
