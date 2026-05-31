# cGAN GEFS Precipitation Downscaling — Investigation Methods & Steps Summary

**Date:** 2026-03-02
**Project:** GIK-cGAN GEFS Precipitation Downscaling (ICPAC East Africa)
**Test Case:** 20240520 00Z (+30h to +54h valid times)
**Reference:** `correct-run-20240520.jpeg` (~60 mm max near Tanzania/Kenya coast)

---

## 1. Problem Statement

The cGAN inference pipeline produces correct **spatial patterns** (precipitation
concentrated near the Tanzania/Kenya coast) but output **magnitudes are ~25x too low**
(~2.4 mm ensemble mean max vs ~60 mm in the operational reference image).

This document details every step taken across multiple debugging sessions
(2026-02-26 through 2026-03-02) to identify and fix root causes.

---

## 2. Chronological Investigation Steps

### Session 1: 2026-02-26 — Initial Bug Fixes (3 issues)

#### Step 1.1: nonnegative_fields Correction

**Method:** Compared inference code against training code (`data/data_gefs.py`).

- **Finding:** Inference had `nonnegative_fields = ['cape', 'pwat', 'apcp']`
- **Training code had:** `['cape', 'msl', 'pres', 'pwat', 'tmp']`
- **Impact:** `msl`, `pres`, `tmp` were not clamped to >= 0 before normalization;
  `apcp` was incorrectly treated as nonnegative (it IS nonnegative but handled
  separately via `log10(1+x)`)
- **Fix:** Updated `nonnegative_fields` in `run_gefs_inference_raw.py` to match training

#### Step 1.2: Elevation Normalization

**Method:** Traced constants loading path from training code to inference code.

- **Finding:** Training code divides elevation by 10000 (`z / 10000.0`); inference
  was using raw elevation values (~0-3000 m)
- **Impact:** Elevation input was orders of magnitude too large
- **Fix:** Added `z = elev_data / 10000.0` in constants loading

#### Step 1.3: denormalise Function Fix

**Method:** Traced output postprocessing in training evaluation scripts.

- **Finding:** The correct denormalization for precipitation is:
  `min(10^x - 1, 100)` (inverse of `log10(1+x)`, capped at 100 mm)
- **Impact:** Output values were not properly converted from log-space to mm
- **Fix:** Updated `denormalise()` to `np.minimum(10**x - 1.0, 100.0)`

---

### Session 2: 2026-02-27 — Plotting and Configuration Fixes

#### Step 2.1: Plotting Routine Corrections

**Method:** Reviewed cartopy/matplotlib plotting code against expected output format.

- Fixed coordinate system alignment in `plot_cgan_comparison.py`
- Corrected color scale to match ICPAC operational plots

---

### Session 3: 2026-02-28 — Critical Bug Discovery (Latitude Flip + Step Indices)

#### Step 3.1: Latitude Flip Detection

**Method:** Systematic comparison of coordinate systems across all data sources.

1. Extracted latitude arrays from:
   - GEFS source data (via AWS S3): **descending** (24.5 → -13.5, N→S)
   - Constants `elev.nc` / `lsm.nc`: **ascending** (-13.65 → 24.65, S→N)
   - Training zarr files: **ascending** (matched constants)

2. Created `test_lat_flip.py` — a diagnostic script that runs the model with
   both latitude orientations and compares spatial patterns:

   ```
   WITH flip (ascending):     max=4.25mm at row 59  ≈ -7.75°S  (Tanzania coast) ✓
   WITHOUT flip (descending): max=7.27mm at row 314 ≈ 17.75°N  (Sudan/Sahara)   ✗
   ZEROS input:               max=0.00mm                        (correct baseline)
   ```

- **Impact:** Model was seeing upside-down meteorological data; spatial patterns
  were geographically inverted
- **Fix:** Added latitude flip in `run_gefs_inference_raw.py:600-602`:
  ```python
  lat_vals = nc_file.latitude.values
  if lat_vals[0] > lat_vals[-1]:
      data = data[:, :, ::-1, :]
  ```

#### Step 3.2: GEFS Step Index Calculation Bug

**Method:** Printed actual vs expected forecast hours from downloaded data.

1. GEFS has 81 steps at 3-hour intervals: positions 0,1,2,... = hours 0,3,6,...,240
2. `compute_cgan_step_indices()` was using forecast hours directly as positional
   indices: positions [29,35,41,47,53] = hours [87,105,123,141,159]
3. cGAN expects hours [30,36,42,48,54] = positions [10,12,14,16,18]

- **Finding:** Pipeline was downloading 3-4 day forecasts instead of 1-2 day
- **Impact:** All input data was from wrong forecast lead times
- **Fix:** Changed to `position = hour // 3`:
  ```python
  step_positions = [h // step_interval for h in step_hours]
  ```

---

### Session 4: 2026-03-01 — Normalization Deep Dive + Cumulative APCP Hypothesis

#### Step 4.1: Normalization File (FCSTNorm_GEFS_2018.pkl) Evaluation

**Method:** Loaded and printed all normalization statistics; cross-referenced each
field's normalization branch against the training code.

**pkl file contents:**
```
Field  Min         Max          Mean         Std
cape   0.0         6497.0       364.27       599.32
hgt    -55.34      2990.58      562.20       507.95
msl    95088.59    103164.02    101139.35    437.77
pres   70930.13    102719.36    94907.64     5422.82
pwat   0.40        83.90        29.94        12.98
tmp    266.81      325.50       298.84       6.09
ugrd   -43.09      43.72        -1.01        3.25
vgrd   -41.20      43.77        0.54         3.85
```

**Key finding:** No `apcp` entry — apcp uses `log10(1+x)` only (no reference stats).

**Normalization branch validation:**

| Branch | Fields | Method | Matches Training? |
|--------|--------|--------|-------------------|
| 1 | apcp | `log10(1 + x)` | Yes |
| 2 | msl, pres, tmp | z-score `(x - mean) / std` | Yes |
| 3 | cape, pwat | divide by max | Yes |
| 4 | ugrd, vgrd | divide by `max(abs(min), max)` | Yes |

#### Step 4.2: Wind Level Mismatch Identification

**Method:** Compared normalization stats ranges with our streaming data ranges.

- **pkl stats:** ugrd min=-43.09, max=43.72 → suggests 700 hPa wind (~44 m/s)
- **Our data:** u10/v10 max ~24 m/s → 10-meter wind
- **Normalized signal:** 24/43.72 ≈ 0.55 (only 55% of expected range)
- **Confirmed by IFS→GEFS mapping:** `u700 → ugrd`, `v700 → vgrd`
- **Streaming code fetches:** `u10/instant/heightAboveGround/u10` (wrong level!)

**Impact:** Secondary contributor — wind provides context but doesn't directly
drive precipitation magnitude.

**Status:** Identified but **paused** — the 55% signal reduction likely has less
influence than the precipitation accumulation issue.

#### Step 4.3: APCP Accumulation Type Analysis

**Method:** Analyzed GEFS bucket structure and compared raw data values.

1. **GEFS bucket structure:** 6-hour accumulation periods that reset at boundaries
   (hours 0, 6, 12, 18, 24, 30, 36, ...)
2. **Raw data check:** 19 of 30 ensemble members showed `apcp(step=36) < apcp(step=30)`,
   confirming bucket-incremental storage
3. **Training code analysis** (`data/data_gefs.py:256-266`): reads raw zarr values
   with NO accumulation processing
4. **Hypothesis:** Training zarr may have stored total accumulated precipitation
   (from hour 0), making bucket-incremental values ~10-20x too small

**Quantification:**
```
                    Bucket-incremental    Total-accumulated (hypothesized)
Typical value       5-10 mm               80-120 mm
log10(1 + x)        0.78-1.04             1.91-2.08
Gap in log-space    ~1.1 units
Linear impact       10^1.1 ≈ 12.6x
```

---

### Session 5: 2026-03-01/02 — Cumulative APCP Implementation & Testing

#### Step 5.1: Pipeline Modifications for Cumulative APCP

**Method:** Modified the streaming and pipeline code to support cumulative APCP
as a test of the accumulation hypothesis.

**Files modified:**

1. **`stream_gefs_for_cgan.py`:**
   - Added `n_timesteps_overrides` parameter to `create_cgan_zarr_store()` for
     per-variable step count overrides
   - Added `step_filter_overrides` parameter to `stream_all_variables_for_member()`
     for per-variable step filtering
   - This allows apcp to download 19 steps (hours 0-54) while other variables
     download only 5 steps

2. **`run_gefs_gik_cgan_pipeline.py`:**
   - Added `--cumulative_apcp` CLI flag
   - Added `compute_apcp_cumulative_steps(end_hour=54)` — computes all 19 step
     positions needed to reconstruct total accumulated precipitation
   - Added `bucket_to_total_accumulated(data, step_hours, bucket_period=6)`:
     ```python
     def bucket_to_total_accumulated(data, step_hours, bucket_period=6):
         cumulative = np.zeros_like(data)
         for i, hour in enumerate(step_hours):
             if hour == 0:
                 cumulative[:, i, :, :] = 0
                 continue
             current_bucket_start = (hour // bucket_period) * bucket_period
             total = np.zeros_like(data[:, 0, :, :])
             for boundary in range(bucket_period, current_bucket_start + 1, bucket_period):
                 if boundary in hour_to_idx:
                     total += data[:, hour_to_idx[boundary], :, :]
             if hour % bucket_period != 0:
                 total += data[:, i, :, :]
             cumulative[:, i, :, :] = total
         return cumulative
     ```
   - Modified `convert_zarr_to_netcdf()` to apply cumulative conversion and
     extract only the cGAN-needed hours from the 19-step apcp data

#### Step 5.2: Pipeline Execution

**Method:** Ran full pipeline with `--cumulative_apcp` for 20240520.

```
uv run run_gefs_gik_cgan_pipeline.py --date 20240520 --stages 3 4 5 --cumulative_apcp
```

- **Stage 3 (stream):** Successfully streamed all 30 GEFS members with 19 apcp
  steps each (~35s per member)
- **Stage 4 (zarr→netcdf):** Successfully converted, applied cumulative accumulation,
  produced 8 NetCDF files in `gik_cgan_output/netcdf/20240520_00z/`
- **Stage 5 (inference):** OOM killed (exit code 137) at +36h valid time,
  member 35/50 — the 7.9GB VM ran out of memory

**Output:** `GAN_20240520.nc` — truncated at 14KB (corrupted from OOM crash)

#### Step 5.3: Quick Cumulative APCP Test

**Method:** Created `test_cumulative_apcp.py` — a lightweight 5-member test to
evaluate cumulative APCP impact without the full 50-member inference.

**Design:** Uses `h5py` for file reading (avoids netCDF4 HDF5 library errors),
loads fields from pipeline NetCDF output, runs 5 ensemble members with fixed seeds.

**Results:**
```
TEST 1: CUMULATIVE APCP (total accumulated from hour 0)
  apcp raw: step1 mean=24.08 max=51.89, step2 mean=28.72 max=56.58
  apcp log10: step1 mean=1.1440 max=1.7234
  Raw model output (member 1): min=-0.0138 max=1.0916 mean=0.0285
  5-member ens mean: max=6.91mm mean=0.0408mm

Previous bucket-incremental reference:
  apcp raw: step1 mean=1.56 max=8.90
  Raw model output max=1.0134 (log-space)
  5-member ens mean: max=4.25mm
```

**Key finding:** Cumulative APCP produced only **1.6x improvement** (6.91 mm vs
4.25 mm) despite **22x larger apcp input values** (mean 24.08 vs 1.56 mm).

This means the model is relatively insensitive to apcp magnitude — the
precipitation field provides spatial signal but the model doesn't scale output
proportionally to input magnitude.

#### Step 5.4: Comparison Plot

**Method:** Generated `comparison_cumulative_apcp_20240520.png` — 3-panel plot
showing raw GEFS apcp (max 51.89 mm), cGAN output (max 6.91 mm), and enhanced
contrast view.

#### Step 5.5: Inference Config Update

**Method:** Updated `run_gefs_inference_raw.py` CONFIG to point to cumulative
APCP data for standalone inference run:

```python
CONFIG = {
    "input_folder": "gik_cgan_output/netcdf/",
    "output_folder": "gik_cgan_output/cgan_output/",
    "dates": ["2024-05-20"],
    "run": "00",
    "ensemble_members": 25,  # reduced from 50 to avoid OOM
    ...
}
```

**Status:** Config updated but inference not yet executed.

---

## 3. Summary of All Issues Found

| # | Issue | Impact | Status | Session |
|---|-------|--------|--------|---------|
| 1 | nonnegative_fields wrong | Wrong normalization for 5 fields | Fixed | 2026-02-26 |
| 2 | Elevation not divided by 10000 | Constants input too large | Fixed | 2026-02-26 |
| 3 | denormalise() wrong formula | Output not properly converted | Fixed | 2026-02-26 |
| 4 | Latitude flip missing | Model saw upside-down data | Fixed | 2026-02-28 |
| 5 | Step index calculation wrong | Wrong forecast lead times | Fixed | 2026-02-28 |
| 6 | Wind level (10m vs 700hPa) | 55% of expected signal range | Identified, paused | 2026-03-01 |
| 7 | APCP accumulation type | Bucket-incremental vs total | Tested, 1.6x gain only | 2026-03-02 |

---

## 4. Current State After All Fixes

### What's working:
- Spatial patterns are correct (precipitation at Tanzania/Kenya coast)
- Latitude orientation matches training data and constants
- Step positions correspond to correct forecast hours (30-54h)
- Normalization branches match training code exactly
- Cumulative APCP pipeline is functional (stages 3-4 complete)

### What's not yet resolved:
- **Output magnitude still ~10x too low** (6.91 mm vs ~60 mm reference)
- Cumulative APCP alone doesn't explain the gap (only 1.6x improvement)
- Wind level fix (10m → 700hPa) identified but not yet implemented
- Combined effect of wind + apcp fix not yet tested
- Full 50-member inference OOMs on 8GB VM (25 members configured but not run)

### Remaining hypotheses:
1. **Training zarr format differs from GEFS S3 format** — the training zarr may
   have pre-processed apcp differently than raw GEFS bucket accumulation
2. **Wind level fix may have compound effect** — 700hPa winds carry different
   moisture transport signals that could amplify precipitation output
3. **Model checkpoint sensitivity** — checkpoint 345600 may need different
   input characteristics than what we're providing
4. **Ensemble statistics computation** — slight differences in how member
   mean/std are computed could affect the 32-channel input tensor

---

## 5. Files Created/Modified During Investigation

### New files:
| File | Purpose |
|------|---------|
| `test_cumulative_apcp.py` | Quick 5-member cumulative APCP test |
| `test_lat_flip.py` | Latitude flip diagnostic (5 tests) |
| `test_cgan_inference.py` | Unit tests for normalization |
| `plot_cgan_comparison.py` | GEFS vs cGAN comparison plots |
| `plot_exceedance_comparison.py` | Exceedance probability plots |
| `comparison_cumulative_apcp_20240520.png` | Cumulative test result plot |
| `2026-03-01-cgan-normalization-investigation.md` | Detailed normalization investigation |
| `2026-03-02-cgan-investigation-methods-summary.md` | This document |

### Modified files:
| File | Changes |
|------|---------|
| `run_gefs_inference_raw.py` | Lat flip, nonneg fields, denormalise, elevation norm, CONFIG update |
| `run_gefs_gik_cgan_pipeline.py` | Cumulative APCP flag, bucket-to-total conversion, step calculation fix |
| `stream_gefs_for_cgan.py` | Per-variable step overrides for cumulative APCP download |

### Pipeline output:
| Path | Contents |
|------|----------|
| `gik_cgan_output/netcdf/20240520_00z/` | 8 NetCDF files with cumulative APCP (30 members, 5 steps each) |
| `gik_cgan_output/cgan_output/2024/GAN_20240520.nc` | 14KB truncated file (OOM crash, needs regeneration) |

---

## 6. Recommended Next Steps

1. **Run standalone inference** (`uv run run_gefs_inference_raw.py`) with 25 members
   on the cumulative APCP data to get a proper GAN_20240520.nc output file

2. **Fix wind level** in `stream_gefs_for_cgan.py` — change from 10m (u10/v10)
   to 700hPa wind and re-stream

3. **Test combined fix** — cumulative APCP + 700hPa wind together

4. **Investigate training zarr contents** — if possible, directly inspect the
   ICPAC training zarr files (`/home/nshruti_icpac_net/zarr/2018/`) to determine
   exact apcp format (bucket vs cumulative vs raw instantaneous)

5. **Cross-validate** with known-good dates (20240523, 20240524, 20240525) that
   already have GAN output for comparison
