#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "tensorflow==2.15",
#     "numpy<2.0",
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "netcdf4",
#     "scipy",
#     "herbie-data",
#     "cfgrib",
#     "eccodes",
#     "gribberish",
#     "kerchunk",
#     "zarr",
#     "pandas",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "pyyaml",
#     "cftime",
#     "requests",
#     "dask",
#     "distributed",
#     "gcsfs",
#     "google-cloud-storage",
#     "google-auth",
# ]
# ///
"""
Herbie vs GIK cGAN Side-by-Side Comparison Test
=================================================

End-to-end test that for a given date:
  Path A (Herbie): Fetch all 8 vars via Herbie → save NetCDFs → run cGAN inference
  Path B (GIK):    Run GIK pipeline stages 1-4 → run cGAN inference

Produces separate log files for each path with per-stage timing so you can
evaluate which method (Herbie vs GIK) is faster.

After subsetting to East Africa, Herbie GRIB cache is cleaned up.

Usage:
    uv run test_herbie_vs_gik_cgan.py --date 20240520 --max-members 30
    uv run test_herbie_vs_gik_cgan.py --date 20240520 --skip-fetch
    uv run test_herbie_vs_gik_cgan.py --date 20240520 --compare-only
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_OUTPUT = _SCRIPT_DIR / "herbie_vs_gik_test"

ALL_FIELDS = ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl", "apcp"]

HOURS = 6
START_HOUR = 30
END_HOUR = 54
STEP_HOURS = list(range(START_HOUR, END_HOUR + HOURS + 1, HOURS))  # [30,36,42,48,54,60]

LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25


# ── Logging helpers ───────────────────────────────────────────────────

def create_logger(name: str, log_file: Path) -> logging.Logger:
    """Create a logger that writes to both stdout and a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class TimingLog:
    """Collects per-stage timing entries and writes a summary."""

    def __init__(self, label: str):
        self.label = label
        self.entries = []
        self.start = time.time()

    def record(self, stage: str, elapsed: float):
        self.entries.append({"stage": stage, "seconds": elapsed})

    def total(self) -> float:
        return time.time() - self.start

    def summary_lines(self) -> list:
        lines = [
            f"{'Stage':<35} {'Time (s)':<12} {'Time (min)':<12}",
            "-" * 59,
        ]
        for e in self.entries:
            lines.append(f"{e['stage']:<35} {e['seconds']:<12.1f} {e['seconds']/60:<12.2f}")
        total = self.total()
        lines.append("-" * 59)
        lines.append(f"{'TOTAL':<35} {total:<12.1f} {total/60:<12.2f}")
        return lines

    def write(self, log: logging.Logger):
        log.info(f"\n{'='*59}")
        log.info(f"TIMING SUMMARY — {self.label}")
        log.info(f"{'='*59}")
        for line in self.summary_lines():
            log.info(line)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "stages": self.entries,
            "total_seconds": self.total(),
        }


# ==============================================================================
# PATH A: Herbie-based fetch
# ==============================================================================

def run_herbie_path(date_str: str, run: str, max_members: int,
                    output_dir: Path, log: logging.Logger,
                    timing: TimingLog) -> bool:
    """Fetch all 8 cGAN variables via Herbie and save as NetCDFs."""
    log.info("=" * 70)
    log.info("PATH A: Herbie-based GEFS Fetch")
    log.info("=" * 70)

    sys.path.insert(0, str(_SCRIPT_DIR))
    from fetch_gefs_herbie_for_cgan import (
        fetch_all_variables_for_date, cleanup_herbie_cache,
    )

    t0 = time.time()
    saved, var_timing = fetch_all_variables_for_date(
        date_str, run=run, max_members=max_members,
        logger=log, output_dir=output_dir,
    )
    fetch_elapsed = time.time() - t0
    timing.record("Herbie fetch (all 8 vars)", fetch_elapsed)

    # Per-variable timing
    for var, secs in var_timing.items():
        timing.record(f"  {var}", secs)

    # Cleanup Herbie GRIB cache
    t1 = time.time()
    cleanup_herbie_cache(log)
    timing.record("Herbie GRIB cache cleanup", time.time() - t1)

    # Report NetCDF sizes
    total_kb = sum(p.stat().st_size for p in saved.values()) / 1024
    log.info(f"\n  Variables saved: {len(saved)}/{len(ALL_FIELDS)}")
    log.info(f"  Total NetCDF size: {total_kb:.1f} KB (East Africa only)")
    log.info(f"  Fetch time: {fetch_elapsed:.1f}s ({fetch_elapsed/60:.1f} min)")

    return len(saved) == len(ALL_FIELDS)


# ==============================================================================
# PATH B: GIK pipeline (stages 1-4)
# ==============================================================================

def run_gik_path(date_str: str, run: str, max_members: int,
                 output_dir: Path, template_path: str,
                 log: logging.Logger, timing: TimingLog) -> bool:
    """Run GIK pipeline stages 1-4 to produce NetCDF files."""
    log.info("=" * 70)
    log.info("PATH B: GIK Pipeline (Stages 1-4)")
    log.info("=" * 70)

    sys.path.insert(0, str(_SCRIPT_DIR))
    from run_gefs_gik_cgan_pipeline import (
        download_template,
        create_gik_references,
        stream_gefs_data,
        convert_zarr_to_netcdf,
        compute_cgan_step_indices,
        TEMPLATE_URL,
        FORECAST_HOURS,
        CGAN_HOURS,
    )

    ensemble_members = [f'gep{i:02d}' for i in range(1, max_members + 1)]
    gik_base = output_dir.parent
    parquet_dir = gik_base / "parquet_refs"
    netcdf_dir = output_dir

    step_positions, step_hours = compute_cgan_step_indices(
        FORECAST_HOURS[0], FORECAST_HOURS[1], CGAN_HOURS
    )
    step_filter = set(step_positions)

    # NOTE: Do NOT use cumulative_apcp — the cGAN model was trained on
    # raw bucket-incremental APCP (6h accumulation per step), not total
    # accumulated from hour 0.  Using cumulative inflates APCP by ~5-9x.

    # Stage 1: Download template
    log.info("\n  Stage 1: Download template")
    t0 = time.time()
    if not download_template(TEMPLATE_URL, template_path):
        log.error("  Stage 1 failed")
        return False
    timing.record("GIK Stage 1: download template", time.time() - t0)

    # Stage 2: Create parquet references
    log.info("\n  Stage 2: Create parquet references")
    t0 = time.time()
    if not create_gik_references(template_path, date_str, run, parquet_dir, ensemble_members):
        log.error("  Stage 2 failed")
        return False
    timing.record("GIK Stage 2: parquet refs", time.time() - t0)

    # Stage 3: Stream data (raw bucket-incremental, cGAN steps only)
    log.info("\n  Stage 3: Stream GEFS data")
    t0 = time.time()
    if not stream_gefs_data(
        parquet_dir, gik_base, date_str, run, max_members,
        step_filter=step_filter, step_hours=step_hours,
    ):
        log.error("  Stage 3 failed")
        return False
    timing.record("GIK Stage 3: stream GEFS data", time.time() - t0)

    # Stage 4: Convert to NetCDF
    log.info("\n  Stage 4: Convert zarr to NetCDF")
    t0 = time.time()
    zarr_path = gik_base / f"zarr_{date_str}_{run}z"
    if not convert_zarr_to_netcdf(zarr_path, netcdf_dir, date_str, run):
        log.error("  Stage 4 failed")
        return False
    timing.record("GIK Stage 4: zarr→NetCDF", time.time() - t0)

    # Report NetCDF sizes
    nc_files = list(netcdf_dir.glob("*.nc"))
    total_kb = sum(f.stat().st_size for f in nc_files) / 1024
    log.info(f"\n  NetCDF files: {len(nc_files)}")
    log.info(f"  Total size: {total_kb:.1f} KB")

    return True


# ==============================================================================
# COMPARISON: Pre-inference input data
# ==============================================================================

def compare_inputs(herbie_dir: Path, gik_dir: Path, date_str: str,
                   run: str, log: logging.Logger) -> Dict:
    """Compare raw input data from both paths for all 8 variables."""
    from scipy import stats as sp_stats

    log.info("\n" + "=" * 70)
    log.info("PRE-INFERENCE COMPARISON: Herbie vs GIK Inputs")
    log.info("=" * 70)

    results = {}

    for field in ALL_FIELDS:
        herbie_file = herbie_dir / f"{field}_{date_str}_{run}z.nc"
        gik_file = gik_dir / f"{field}_{date_str}_{run}z.nc"

        if not herbie_file.exists():
            results[field] = {"error": f"Herbie file missing"}
            log.warning(f"  {field}: Herbie file missing: {herbie_file}")
            continue
        if not gik_file.exists():
            results[field] = {"error": f"GIK file missing"}
            log.warning(f"  {field}: GIK file missing: {gik_file}")
            continue

        ds_h = xr.open_dataset(herbie_file)
        ds_g = xr.open_dataset(gik_file)

        var_h = list(ds_h.data_vars)[0]
        var_g = list(ds_g.data_vars)[0]

        h_data = ds_h[var_h].isel(time=0)
        g_data = ds_g[var_g].isel(time=0)

        h_steps = set(int(s) for s in ds_h.step.values)
        g_steps = set(int(s) for s in ds_g.step.values)
        common_steps = sorted(h_steps & g_steps)

        if not common_steps:
            results[field] = {"error": f"No common steps: H={h_steps}, G={g_steps}"}
            log.warning(f"  {field}: No common steps")
            ds_h.close(); ds_g.close()
            continue

        step = common_steps[0]
        h_arr = h_data.sel(step=step).mean(dim="member").values
        g_arr = g_data.sel(step=step).mean(dim="member").values

        if h_arr.shape != g_arr.shape:
            results[field] = {"error": f"Shape mismatch: {h_arr.shape} vs {g_arr.shape}"}
            log.warning(f"  {field}: Shape mismatch {h_arr.shape} vs {g_arr.shape}")
            ds_h.close(); ds_g.close()
            continue

        diff = h_arr - g_arr
        valid = ~(np.isnan(h_arr) | np.isnan(g_arr))
        h_v, g_v = h_arr[valid], g_arr[valid]

        if len(h_v) == 0:
            results[field] = {"error": "No valid pixels"}
            continue

        if np.std(h_v) > 0 and np.std(g_v) > 0:
            r, p = sp_stats.pearsonr(h_v.flatten(), g_v.flatten())
        else:
            r, p = float('nan'), float('nan')

        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        mae = float(np.nanmean(np.abs(diff)))
        max_abs = float(np.nanmax(np.abs(diff)))

        results[field] = {
            "corr": float(r), "corr_p": float(p),
            "rmse": rmse, "mae": mae, "max_abs_diff": max_abs,
            "herbie_range": [float(np.nanmin(h_arr)), float(np.nanmax(h_arr))],
            "gik_range": [float(np.nanmin(g_arr)), float(np.nanmax(g_arr))],
            "step": step, "n_common_steps": len(common_steps),
        }

        log.info(f"  {field:6s}: r={r:.6f}  RMSE={rmse:.4e}  MAE={mae:.4e}  "
                 f"MaxDiff={max_abs:.4e}  "
                 f"H=[{np.nanmin(h_arr):.2f},{np.nanmax(h_arr):.2f}]  "
                 f"G=[{np.nanmin(g_arr):.2f},{np.nanmax(g_arr):.2f}]")

        ds_h.close(); ds_g.close()

    return results


# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference_on_path(
    netcdf_dir: Path, output_dir: Path, date_str: str, run: str,
    label: str, log: logging.Logger, timing: TimingLog,
) -> bool:
    """Run cGAN inference using the given input NetCDF directory."""
    log.info(f"\n  Running cGAN inference ({label})...")
    log.info(f"    Input:  {netcdf_dir}")
    log.info(f"    Output: {output_dir}")

    sys.path.insert(0, str(_SCRIPT_DIR))

    t0 = time.time()
    try:
        from run_gefs_inference_raw import run_inference, CONFIG as inference_config

        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        model_folder = os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/logfile_gefs_v3/")
        constants_path = os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/CONSTANTS/")

        inference_config["model_folder"] = model_folder
        inference_config["checkpoint"] = 345600
        inference_config["input_folder"] = str(netcdf_dir.parent)
        inference_config["constants_path"] = constants_path
        inference_config["output_folder"] = str(output_dir)
        inference_config["dates"] = [date_formatted]
        inference_config["run"] = run
        inference_config["start_hour"] = START_HOUR
        inference_config["end_hour"] = END_HOUR
        inference_config["ensemble_members"] = 25
        inference_config["normalization_mode"] = "gefs"
        inference_config["gefs_norm_file"] = os.path.join(
            constants_path, "FCSTNorm_GEFS_2018.pkl"
        )

        run_inference()
        elapsed = time.time() - t0
        timing.record(f"cGAN inference ({label})", elapsed)
        log.info(f"    Inference ({label}): {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        timing.record(f"cGAN inference ({label}) FAILED", elapsed)
        log.error(f"    Inference failed after {elapsed:.1f}s: {e}")
        return False


# ==============================================================================
# COMPARISON: Post-inference cGAN outputs
# ==============================================================================

def compare_outputs(herbie_output_dir: Path, gik_output_dir: Path,
                    date_str: str) -> Dict:
    """Compare cGAN outputs from both paths."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("POST-INFERENCE COMPARISON: Herbie vs GIK cGAN Outputs")
    print("=" * 70)

    year = date_str[:4]
    gan_file = f"GAN_{date_str}.nc"

    herbie_file = herbie_output_dir / year / gan_file
    gik_file = gik_output_dir / year / gan_file

    results = {}

    if not herbie_file.exists():
        print(f"  Herbie cGAN output not found: {herbie_file}")
        results["error"] = f"Herbie output missing: {herbie_file}"
        return results
    if not gik_file.exists():
        print(f"  GIK cGAN output not found: {gik_file}")
        results["error"] = f"GIK output missing: {gik_file}"
        return results

    ds_h = xr.open_dataset(herbie_file)
    ds_g = xr.open_dataset(gik_file)

    h_precip = ds_h["precipitation"].isel(time=0).mean(dim="member")
    g_precip = ds_g["precipitation"].isel(time=0).mean(dim="member")

    n_valid_times = min(len(h_precip.valid_time), len(g_precip.valid_time))

    for vt_idx in range(n_valid_times):
        h_arr = h_precip.isel(valid_time=vt_idx).values
        g_arr = g_precip.isel(valid_time=vt_idx).values

        valid = ~(np.isnan(h_arr) | np.isnan(g_arr))
        h_v, g_v = h_arr[valid], g_arr[valid]

        if len(h_v) == 0:
            continue

        if np.std(h_v) > 0 and np.std(g_v) > 0:
            r, _ = sp_stats.pearsonr(h_v.flatten(), g_v.flatten())
        else:
            r = float('nan')

        rmse = float(np.sqrt(np.nanmean((h_arr - g_arr) ** 2)))
        key = f"valid_time_{vt_idx}"
        results[key] = {
            "corr": float(r),
            "rmse": rmse,
            "herbie_max": float(np.nanmax(h_arr)),
            "gik_max": float(np.nanmax(g_arr)),
            "herbie_mean": float(np.nanmean(h_arr)),
            "gik_mean": float(np.nanmean(g_arr)),
        }

        print(f"  VT{vt_idx}: r={r:.6f}  RMSE={rmse:.4f}  "
              f"H_max={np.nanmax(h_arr):.2f}  G_max={np.nanmax(g_arr):.2f}  "
              f"H_mean={np.nanmean(h_arr):.4f}  G_mean={np.nanmean(g_arr):.4f}")

    ds_h.close()
    ds_g.close()
    return results


# ==============================================================================
# BENCHMARK: cGAN output vs raw GEFS input
# ==============================================================================

def compute_intensity_ratio(input_dir: Path, output_dir: Path,
                            date_str: str, run: str, label: str,
                            log: logging.Logger) -> Dict:
    """Compare cGAN output intensity against raw GEFS input."""
    log.info(f"\n  Intensity ratio ({label}):")

    year = date_str[:4]
    apcp_file = input_dir / f"apcp_{date_str}_{run}z.nc"
    gan_file = output_dir / year / f"GAN_{date_str}.nc"

    if not apcp_file.exists() or not gan_file.exists():
        missing = []
        if not apcp_file.exists(): missing.append(f"apcp: {apcp_file}")
        if not gan_file.exists(): missing.append(f"GAN: {gan_file}")
        log.warning(f"    Missing: {missing}")
        return {"error": f"Missing files: {missing}"}

    ds_apcp = xr.open_dataset(apcp_file)
    ds_gan = xr.open_dataset(gan_file)

    apcp_var = list(ds_apcp.data_vars)[0]
    raw = ds_apcp[apcp_var].isel(time=0)
    if "member" in raw.dims: raw = raw.mean(dim="member")
    if "step" in raw.dims: raw = raw.isel(step=0)
    raw_arr = raw.values

    gan = ds_gan["precipitation"].isel(time=0).mean(dim="member")
    if "valid_time" in gan.dims: gan = gan.isel(valid_time=0)
    gan_arr = gan.values

    raw_max = float(np.nanmax(raw_arr))
    gan_max = float(np.nanmax(gan_arr))
    raw_mean = float(np.nanmean(raw_arr))
    gan_mean = float(np.nanmean(gan_arr))
    ratio_max = gan_max / max(raw_max, 1e-12)
    ratio_mean = gan_mean / max(raw_mean, 1e-12)

    result = {
        "raw_max": raw_max, "gan_max": gan_max,
        "raw_mean": raw_mean, "gan_mean": gan_mean,
        "ratio_max": ratio_max, "ratio_mean": ratio_mean,
    }

    log.info(f"    Raw GEFS: max={raw_max:.2f} mm, mean={raw_mean:.4f} mm")
    log.info(f"    cGAN out: max={gan_max:.2f} mm, mean={gan_mean:.4f} mm")
    log.info(f"    Ratio (cGAN/raw): max={ratio_max:.2f}x, mean={ratio_mean:.2f}x")

    ds_apcp.close(); ds_gan.close()
    return result


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_comparison(herbie_input_dir: Path, gik_input_dir: Path,
                    herbie_output_dir: Path, gik_output_dir: Path,
                    date_str: str, run: str, output_dir: Path,
                    log: logging.Logger):
    """Generate multi-panel comparison plots with colorbars for every timestep.

    Layout:
      - One figure per valid time (+30h, +36h, +42h, +48h)
        Row 0: Raw GEFS APCP — Herbie | GIK | Difference
        Row 1: cGAN output   — Herbie | GIK | Difference
        Each row has a shared colorbar (mm/h for cGAN, mm for raw APCP)
      - One summary figure with intensity ratio histograms
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    log.info("\n" + "=" * 70)
    log.info("GENERATING COMPARISON PLOTS")
    log.info("=" * 70)

    year = date_str[:4]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Precipitation color scale (mm/h for cGAN, mm for raw APCP per 6h step)
    precip_levels = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    precip_colors = [
        '#ffffff', '#c6dbef', '#9ecae1', '#6baed6',
        '#3182bd', '#08519c', '#ffff00', '#ff7f00', '#ff0000'
    ]
    cmap_precip = mcolors.ListedColormap(precip_colors)
    norm_precip = mcolors.BoundaryNorm(precip_levels, cmap_precip.N)

    # cGAN output scale (mm/h — typically 0–10 range)
    cgan_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    cgan_colors = [
        '#ffffff', '#edf8e9', '#c7e9c0', '#a1d99b',
        '#74c476', '#31a354', '#006d2c', '#ffff00', '#ff0000'
    ]
    cmap_cgan = mcolors.ListedColormap(cgan_colors)
    norm_cgan = mcolors.BoundaryNorm(cgan_levels, cmap_cgan.N)

    def add_map(ax):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.add_feature(cfeature.LAKES, alpha=0.3)

    # Load all timesteps from raw APCP inputs
    def load_apcp_all(path):
        if not path.exists():
            return None, None, None
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        d = ds[var].isel(time=0)
        if "member" in d.dims:
            d = d.mean(dim="member")
        lats, lons = ds.latitude.values, ds.longitude.values
        # d has shape (step, lat, lon) or (lat, lon)
        steps = ds.step.values if "step" in ds.dims else None
        vals = d.values
        ds.close()
        return vals, lats, lons, steps

    # Load all valid times from cGAN output
    def load_gan_all(path):
        if not path.exists():
            return None, None, None
        ds = xr.open_dataset(path)
        d = ds["precipitation"].isel(time=0).mean(dim="member")
        lats, lons = ds.latitude.values, ds.longitude.values
        vt = ds.valid_time.values if "valid_time" in ds.dims else None
        vals = d.values  # (valid_time, lat, lon) or (lat, lon)
        ds.close()
        return vals, lats, lons, vt

    h_apcp_all, h_lats, h_lons, h_steps = load_apcp_all(
        herbie_input_dir / f"apcp_{date_str}_{run}z.nc")
    g_apcp_all, g_lats, g_lons, g_steps = load_apcp_all(
        gik_input_dir / f"apcp_{date_str}_{run}z.nc")
    h_gan_all, ho_lats, ho_lons, h_vt = load_gan_all(
        herbie_output_dir / year / f"GAN_{date_str}.nc")
    g_gan_all, go_lats, go_lons, g_vt = load_gan_all(
        gik_output_dir / year / f"GAN_{date_str}.nc")

    # Determine valid times from cGAN output
    n_vt = 0
    if h_gan_all is not None and h_gan_all.ndim >= 2:
        n_vt = h_gan_all.shape[0] if h_gan_all.ndim == 3 else 1
    elif g_gan_all is not None and g_gan_all.ndim >= 2:
        n_vt = g_gan_all.shape[0] if g_gan_all.ndim == 3 else 1
    if n_vt == 0:
        n_vt = 1

    valid_hours = [START_HOUR + i * HOURS for i in range(n_vt)]

    # ── Per-timestep plots ──
    for vt_idx in range(n_vt):
        hour = valid_hours[vt_idx]

        # Extract data for this timestep
        def get_step(data_all, steps, idx):
            if data_all is None:
                return None
            if data_all.ndim == 3:
                if idx < data_all.shape[0]:
                    return data_all[idx]
            elif data_all.ndim == 2:
                return data_all if idx == 0 else None
            return None

        h_raw = get_step(h_apcp_all, h_steps, vt_idx)
        g_raw = get_step(g_apcp_all, g_steps, vt_idx)
        h_gan = get_step(h_gan_all, h_vt, vt_idx)
        g_gan = get_step(g_gan_all, g_vt, vt_idx)

        fig = plt.figure(figsize=(24, 14))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.04],
                               hspace=0.22, wspace=0.12)

        # Row 0: Raw GEFS APCP
        row0_data = [
            (h_raw, h_lats, h_lons, "Herbie"),
            (g_raw, g_lats, g_lons, "GIK"),
        ]
        im_raw = None
        for col, (data, lats, lons, src) in enumerate(row0_data):
            ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
            if data is not None and lats is not None:
                mx = float(np.nanmax(data))
                mn = float(np.nanmean(data))
                im_raw = ax.pcolormesh(lons, lats, data, cmap=cmap_precip, norm=norm_precip,
                                       transform=ccrs.PlateCarree(), shading='auto')
                ax.set_title(f"Raw GEFS ({src})\nmax={mx:.2f} mm  mean={mn:.4f} mm",
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"Raw GEFS ({src})\n(no data)", fontsize=11)
            add_map(ax)

        # Row 0, col 2: Difference
        ax = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        if h_raw is not None and g_raw is not None and h_raw.shape == g_raw.shape:
            diff = h_raw - g_raw
            vmax = max(float(np.nanmax(np.abs(diff))), 1e-6)
            ax.pcolormesh(h_lons, h_lats, diff, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')
            ax.set_title(f"Diff (H−G)\nmax|diff|={vmax:.2f} mm",
                         fontsize=11, fontweight='bold')
        else:
            ax.set_title("Diff (H−G)\n(shape mismatch)", fontsize=11)
        add_map(ax)

        # Row 0 colorbar
        if im_raw is not None:
            cbar_ax = fig.add_subplot(gs[0, 3])
            cb = fig.colorbar(im_raw, cax=cbar_ax, extend='max')
            cb.set_label("Precipitation (mm)", fontsize=10)
            cb.ax.tick_params(labelsize=9)

        # Row 1: cGAN output
        row1_data = [
            (h_gan, ho_lats, ho_lons, "Herbie"),
            (g_gan, go_lats, go_lons, "GIK"),
        ]
        im_cgan = None
        for col, (data, lats, lons, src) in enumerate(row1_data):
            ax = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree())
            if data is not None and lats is not None:
                mx = float(np.nanmax(data))
                mn = float(np.nanmean(data))
                im_cgan = ax.pcolormesh(lons, lats, data, cmap=cmap_cgan, norm=norm_cgan,
                                        transform=ccrs.PlateCarree(), shading='auto')
                ax.set_title(f"cGAN ({src})\nmax={mx:.2f} mm/h  mean={mn:.4f} mm/h",
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"cGAN ({src})\n(no data)", fontsize=11)
            add_map(ax)

        # Row 1, col 2: Difference
        ax = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
        if h_gan is not None and g_gan is not None and h_gan.shape == g_gan.shape:
            diff = h_gan - g_gan
            vmax = max(float(np.nanmax(np.abs(diff))), 1e-6)
            ax.pcolormesh(ho_lons, ho_lats, diff, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')
            ax.set_title(f"Diff (H−G)\nmax|diff|={vmax:.2f} mm/h",
                         fontsize=11, fontweight='bold')
        else:
            ax.set_title("Diff (H−G)\n(shape mismatch)", fontsize=11)
        add_map(ax)

        # Row 1 colorbar
        if im_cgan is not None:
            cbar_ax = fig.add_subplot(gs[1, 3])
            cb = fig.colorbar(im_cgan, cax=cbar_ax, extend='max')
            cb.set_label("Precipitation (mm/h)", fontsize=10)
            cb.ax.tick_params(labelsize=9)

        fig.suptitle(
            f"Herbie vs GIK — {date_str} {run}Z — Valid +{hour}h\n"
            f"Top: Raw GEFS APCP (ens. mean)  |  Bottom: cGAN Downscaled (ens. mean, mm/h)",
            fontsize=14, fontweight='bold', y=0.99,
        )

        out_path = output_dir / f"herbie_vs_gik_{date_str}_T{hour:03d}h.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        log.info(f"  Saved: {out_path}")

    # ── Intensity ratio histograms ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (gan_all, raw_all, label) in zip(axes2, [
        (h_gan_all, h_apcp_all, "Herbie"),
        (g_gan_all, g_apcp_all, "GIK"),
    ]):
        if gan_all is not None and raw_all is not None:
            # Use all valid times
            gan_f = gan_all.flatten()
            # For raw, use matching number of steps
            n_match = min(gan_all.shape[0] if gan_all.ndim == 3 else 1,
                          raw_all.shape[0] if raw_all.ndim == 3 else 1)
            raw_slice = raw_all[:n_match] if raw_all.ndim == 3 else raw_all
            raw_f = raw_slice.flatten()
            # Align sizes
            n = min(len(gan_f), len(raw_f))
            gan_f, raw_f = gan_f[:n], raw_f[:n]
            ok = (raw_f > 0.01) & np.isfinite(raw_f) & np.isfinite(gan_f)
            if ok.sum() > 0:
                ratios = gan_f[ok] / raw_f[ok]
                ratios = ratios[ratios < 100]
                ax.hist(ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                med = np.median(ratios)
                ax.axvline(med, color='red', linestyle='--',
                           label=f'Median: {med:.2f}x')
                ax.legend(fontsize=10)
        ax.set_xlabel("cGAN (mm/h) / Raw GEFS (mm) ratio", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{label} path — Intensity Ratio (all valid times)",
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    hist_path = output_dir / f"intensity_ratio_{date_str}.png"
    fig2.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info(f"  Saved: {hist_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Herbie vs GIK cGAN Side-by-Side Comparison Test"
    )
    parser.add_argument("--date", type=str, required=True,
                        help="Target date (YYYYMMDD)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default: 00)")
    parser.add_argument("--max-members", type=int, default=30,
                        help="Max ensemble members (default: 30)")
    parser.add_argument("--output-dir", type=str, default=str(BASE_OUTPUT),
                        help=f"Base output directory (default: {BASE_OUTPUT})")
    parser.add_argument("--template", type=str, default="gik-fmrc-gefs-20241112.tar.gz",
                        help="GIK template tar.gz file")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching (use existing NetCDFs)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip cGAN inference (just compare inputs)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip fetch and inference, only run comparisons")
    args = parser.parse_args()

    date_str = args.date
    run = args.run
    base_dir = Path(args.output_dir)

    # Directory layout
    herbie_input_dir = base_dir / "herbie" / f"{date_str}_{run}z"
    gik_input_dir = base_dir / "gik" / "netcdf" / f"{date_str}_{run}z"
    herbie_output_dir = base_dir / "herbie_cgan"
    gik_output_dir = base_dir / "gik_cgan"
    plots_dir = base_dir / "plots"

    # ── Create separate log files for each path ──
    herbie_log = create_logger("herbie_path", base_dir / "log_herbie.txt")
    gik_log = create_logger("gik_path", base_dir / "log_gik.txt")
    main_log = create_logger("main", base_dir / "log_main.txt")

    herbie_timing = TimingLog("Herbie Path")
    gik_timing = TimingLog("GIK Path")

    main_log.info("=" * 70)
    main_log.info("HERBIE vs GIK cGAN SIDE-BY-SIDE COMPARISON")
    main_log.info("=" * 70)
    main_log.info(f"Date: {date_str} {run}Z")
    main_log.info(f"Max members: {args.max_members}")
    main_log.info(f"Forecast hours: {START_HOUR}–{END_HOUR} (steps: {STEP_HOURS})")
    main_log.info(f"Base dir: {base_dir}")
    main_log.info(f"Herbie inputs → {herbie_input_dir}")
    main_log.info(f"GIK inputs    → {gik_input_dir}")
    main_log.info(f"Skip fetch:     {args.skip_fetch or args.compare_only}")
    main_log.info(f"Skip inference: {args.skip_inference or args.compare_only}")
    main_log.info("=" * 70)

    pipeline_start = time.time()

    # ── Path A: Herbie ──
    if not args.skip_fetch and not args.compare_only:
        herbie_ok = run_herbie_path(
            date_str, run, args.max_members, herbie_input_dir,
            herbie_log, herbie_timing,
        )
        herbie_log.info(f"\n  Result: {'OK' if herbie_ok else 'FAILED'}")
    else:
        main_log.info("  Herbie fetch: SKIPPED")

    # ── Path B: GIK ──
    if not args.skip_fetch and not args.compare_only:
        gik_ok = run_gik_path(
            date_str, run, args.max_members, gik_input_dir,
            args.template, gik_log, gik_timing,
        )
        gik_log.info(f"\n  Result: {'OK' if gik_ok else 'FAILED'}")
    else:
        main_log.info("  GIK fetch: SKIPPED")

    # ── Pre-inference comparison ──
    input_results = compare_inputs(
        herbie_input_dir, gik_input_dir, date_str, run, main_log,
    )

    # ── cGAN Inference ──
    if not args.skip_inference and not args.compare_only:
        main_log.info("\n" + "=" * 70)
        main_log.info("cGAN INFERENCE")
        main_log.info("=" * 70)

        run_inference_on_path(
            herbie_input_dir, herbie_output_dir, date_str, run,
            "Herbie", herbie_log, herbie_timing,
        )
        run_inference_on_path(
            gik_input_dir, gik_output_dir, date_str, run,
            "GIK", gik_log, gik_timing,
        )
    else:
        main_log.info("  cGAN Inference: SKIPPED")

    # ── Post-inference comparison ──
    output_results = compare_outputs(herbie_output_dir, gik_output_dir, date_str)

    # ── Intensity ratios ──
    main_log.info("\n" + "=" * 70)
    main_log.info("INTENSITY RATIO: cGAN vs Raw GEFS")
    main_log.info("=" * 70)
    herbie_ratio = compute_intensity_ratio(
        herbie_input_dir, herbie_output_dir, date_str, run, "Herbie", main_log,
    )
    gik_ratio = compute_intensity_ratio(
        gik_input_dir, gik_output_dir, date_str, run, "GIK", main_log,
    )

    # ── Write timing summaries to log files ──
    herbie_timing.write(herbie_log)
    gik_timing.write(gik_log)

    # ── Plots ──
    try:
        plot_comparison(
            herbie_input_dir, gik_input_dir,
            herbie_output_dir, gik_output_dir,
            date_str, run, plots_dir, main_log,
        )
    except Exception as e:
        main_log.error(f"  Plot generation failed: {e}")

    # ── Save all results as JSON ──
    all_results = {
        "date": date_str, "run": run,
        "max_members": args.max_members,
        "input_comparison": input_results,
        "output_comparison": output_results,
        "intensity_ratio_herbie": herbie_ratio,
        "intensity_ratio_gik": gik_ratio,
        "timing_herbie": herbie_timing.to_dict(),
        "timing_gik": gik_timing.to_dict(),
    }

    results_file = base_dir / f"comparison_results_{date_str}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Final summary ──
    total_time = time.time() - pipeline_start

    main_log.info("\n" + "=" * 70)
    main_log.info("FINAL SUMMARY")
    main_log.info("=" * 70)

    main_log.info("\nPre-inference input correlation (should be ~1.0):")
    for field in ALL_FIELDS:
        r = input_results.get(field, {})
        if "error" in r:
            main_log.info(f"  {field:6s}: ERROR — {r['error']}")
        else:
            main_log.info(f"  {field:6s}: r={r.get('corr', 'N/A'):.6f}")

    if output_results and "error" not in output_results:
        main_log.info("\nPost-inference output correlation:")
        for key, val in output_results.items():
            if isinstance(val, dict):
                main_log.info(f"  {key}: r={val.get('corr', 'N/A'):.6f}  "
                              f"H_max={val.get('herbie_max', 0):.2f}  "
                              f"G_max={val.get('gik_max', 0):.2f}")

    main_log.info("\nIntensity ratios (cGAN_max / raw_GEFS_max):")
    for label, ratio in [("Herbie", herbie_ratio), ("GIK", gik_ratio)]:
        if "error" in ratio:
            main_log.info(f"  {label}: N/A")
        else:
            main_log.info(f"  {label}: {ratio.get('ratio_max', 0):.2f}x "
                          f"(cGAN {ratio.get('gan_max', 0):.2f} mm vs "
                          f"raw {ratio.get('raw_max', 0):.2f} mm)")

    # ── Speed comparison ──
    main_log.info("\n" + "=" * 70)
    main_log.info("SPEED COMPARISON: Herbie vs GIK")
    main_log.info("=" * 70)
    herbie_timing.write(main_log)
    gik_timing.write(main_log)

    h_total = herbie_timing.total()
    g_total = gik_timing.total()
    if h_total > 0 and g_total > 0:
        faster = "GIK" if g_total < h_total else "Herbie"
        speedup = max(h_total, g_total) / min(h_total, g_total)
        main_log.info(f"\n  {faster} is {speedup:.1f}x FASTER")
        main_log.info(f"  Herbie total: {h_total:.1f}s ({h_total/60:.1f} min)")
        main_log.info(f"  GIK total:    {g_total:.1f}s ({g_total/60:.1f} min)")

    main_log.info(f"\nTotal pipeline time: {total_time:.0f}s ({total_time/60:.1f} min)")
    main_log.info(f"\nLog files:")
    main_log.info(f"  Main:   {base_dir / 'log_main.txt'}")
    main_log.info(f"  Herbie: {base_dir / 'log_herbie.txt'}")
    main_log.info(f"  GIK:    {base_dir / 'log_gik.txt'}")
    main_log.info(f"  Results JSON: {results_file}")
    main_log.info("=" * 70)


if __name__ == "__main__":
    main()
