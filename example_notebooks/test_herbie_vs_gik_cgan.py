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
#     "kerchunk",
#     "zarr",
#     "pandas",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "pyyaml",
#     "cftime",
#     "requests",
# ]
# ///
"""
Herbie vs GIK cGAN Side-by-Side Comparison Test
=================================================

End-to-end test that for a given date:
  Path A (Herbie): Fetch all 8 vars via Herbie → save NetCDFs → run cGAN inference
  Path B (GIK):    Run GIK pipeline stages 1-4 → run cGAN inference

Comparisons:
  1. Pre-inference: raw input data from both paths (should match, r≈1.0)
  2. Post-inference: cGAN outputs from both paths
  3. Benchmark: both outputs vs raw GEFS precipitation (intensity ratio)

Generates multi-panel comparison plots.

Usage:
    uv run test_herbie_vs_gik_cgan.py --date 20250106 --max-members 5
    uv run test_herbie_vs_gik_cgan.py --date 20250106 --skip-fetch  # if data already exists
    uv run test_herbie_vs_gik_cgan.py --date 20250106 --compare-only  # skip inference too
"""

import argparse
import json
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

# Output directories
BASE_OUTPUT = _SCRIPT_DIR / "herbie_vs_gik_test"

# cGAN fields (ordering matters — must match run_gefs_inference_raw.py)
ALL_FIELDS = ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl", "apcp"]

# Forecast hours for cGAN (matching pipeline)
HOURS = 6
START_HOUR = 30
END_HOUR = 54
STEP_HOURS = list(range(START_HOUR, END_HOUR + HOURS + 1, HOURS))  # [30,36,42,48,54,60]

# ICPAC region
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25


# ==============================================================================
# PATH A: Herbie-based fetch
# ==============================================================================

def run_herbie_path(date_str: str, run: str, max_members: int,
                    output_dir: Path) -> bool:
    """Fetch all 8 cGAN variables via Herbie and save as NetCDFs."""
    print("\n" + "=" * 70)
    print("PATH A: Herbie-based GEFS Fetch")
    print("=" * 70)

    sys.path.insert(0, str(_SCRIPT_DIR))
    from fetch_gefs_herbie_for_cgan import fetch_all_variables_for_date

    import logging
    logger = logging.getLogger("herbie_cgan")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        logger.addHandler(ch)

    saved = fetch_all_variables_for_date(
        date_str, run=run, max_members=max_members,
        logger=logger, output_dir=output_dir,
    )

    print(f"\n  Herbie path: {len(saved)}/{len(ALL_FIELDS)} variables saved to {output_dir}")
    return len(saved) == len(ALL_FIELDS)


# ==============================================================================
# PATH B: GIK pipeline (stages 1-4)
# ==============================================================================

def run_gik_path(date_str: str, run: str, max_members: int,
                 output_dir: Path, template_path: str) -> bool:
    """Run GIK pipeline stages 1-4 to produce NetCDF files."""
    print("\n" + "=" * 70)
    print("PATH B: GIK Pipeline (Stages 1-4)")
    print("=" * 70)

    sys.path.insert(0, str(_SCRIPT_DIR))
    from run_gefs_gik_cgan_pipeline import (
        download_template,
        create_gik_references,
        stream_gefs_data,
        convert_zarr_to_netcdf,
        compute_cgan_step_indices,
        compute_apcp_cumulative_steps,
        TEMPLATE_URL,
        FORECAST_HOURS,
        CGAN_HOURS,
    )

    ensemble_members = [f'gep{i:02d}' for i in range(1, max_members + 1)]
    gik_base = output_dir.parent  # e.g., herbie_vs_gik_test/gik/
    parquet_dir = gik_base / "parquet_refs"
    netcdf_dir = output_dir

    # Step filter for cGAN steps only
    step_positions, step_hours = compute_cgan_step_indices(
        FORECAST_HOURS[0], FORECAST_HOURS[1], CGAN_HOURS
    )
    step_filter = set(step_positions)

    # Cumulative APCP steps
    max_cgan_hour = max(step_hours)
    apcp_positions, apcp_step_hours = compute_apcp_cumulative_steps(end_hour=max_cgan_hour)
    apcp_step_filter = set(apcp_positions)

    # Stage 1: Download template
    print("\n  Stage 1: Download template")
    if not download_template(TEMPLATE_URL, template_path):
        print("  Stage 1 failed")
        return False

    # Stage 2: Create parquet references
    print("\n  Stage 2: Create parquet references")
    if not create_gik_references(template_path, date_str, run, parquet_dir, ensemble_members):
        print("  Stage 2 failed")
        return False

    # Stage 3: Stream data
    print("\n  Stage 3: Stream GEFS data")
    if not stream_gefs_data(
        parquet_dir, gik_base, date_str, run, max_members,
        step_filter=step_filter, step_hours=step_hours,
        cumulative_apcp=True,
        apcp_step_filter=apcp_step_filter,
        apcp_step_hours=apcp_step_hours,
    ):
        print("  Stage 3 failed")
        return False

    # Stage 4: Convert to NetCDF
    print("\n  Stage 4: Convert zarr to NetCDF")
    zarr_path = gik_base / f"zarr_{date_str}_{run}z"
    if not convert_zarr_to_netcdf(zarr_path, netcdf_dir, date_str, run):
        print("  Stage 4 failed")
        return False

    print(f"\n  GIK path: NetCDF files saved to {netcdf_dir}")
    return True


# ==============================================================================
# COMPARISON: Pre-inference input data
# ==============================================================================

def compare_inputs(herbie_dir: Path, gik_dir: Path, date_str: str,
                   run: str) -> Dict:
    """Compare raw input data from both paths for all 8 variables."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("PRE-INFERENCE COMPARISON: Herbie vs GIK Inputs")
    print("=" * 70)

    results = {}

    for field in ALL_FIELDS:
        herbie_file = herbie_dir / f"{field}_{date_str}_{run}z.nc"
        gik_file = gik_dir / f"{field}_{date_str}_{run}z.nc"

        if not herbie_file.exists():
            results[field] = {"error": f"Herbie file missing: {herbie_file}"}
            print(f"  {field}: Herbie file missing")
            continue
        if not gik_file.exists():
            results[field] = {"error": f"GIK file missing: {gik_file}"}
            print(f"  {field}: GIK file missing")
            continue

        ds_h = xr.open_dataset(herbie_file)
        ds_g = xr.open_dataset(gik_file)

        var_h = list(ds_h.data_vars)[0]
        var_g = list(ds_g.data_vars)[0]

        # Get ensemble mean across members for a single step
        h_data = ds_h[var_h].isel(time=0)
        g_data = ds_g[var_g].isel(time=0)

        # Use the first common step
        h_steps = set(int(s) for s in ds_h.step.values)
        g_steps = set(int(s) for s in ds_g.step.values)
        common_steps = sorted(h_steps & g_steps)

        if not common_steps:
            results[field] = {"error": f"No common steps: H={h_steps}, G={g_steps}"}
            print(f"  {field}: No common steps")
            ds_h.close()
            ds_g.close()
            continue

        step = common_steps[0]
        h_arr = h_data.sel(step=step).mean(dim="member").values
        g_arr = g_data.sel(step=step).mean(dim="member").values

        # Handle shape mismatch
        if h_arr.shape != g_arr.shape:
            results[field] = {
                "error": f"Shape mismatch: Herbie {h_arr.shape} vs GIK {g_arr.shape}"
            }
            print(f"  {field}: Shape mismatch {h_arr.shape} vs {g_arr.shape}")
            ds_h.close()
            ds_g.close()
            continue

        # Compute stats
        diff = h_arr - g_arr
        valid = ~(np.isnan(h_arr) | np.isnan(g_arr))
        h_v, g_v = h_arr[valid], g_arr[valid]

        if len(h_v) == 0:
            results[field] = {"error": "No valid overlapping pixels"}
            continue

        if np.std(h_v) > 0 and np.std(g_v) > 0:
            r, p = sp_stats.pearsonr(h_v.flatten(), g_v.flatten())
        else:
            r, p = float('nan'), float('nan')

        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        mae = float(np.nanmean(np.abs(diff)))
        max_abs = float(np.nanmax(np.abs(diff)))

        results[field] = {
            "corr": float(r),
            "corr_p": float(p),
            "rmse": rmse,
            "mae": mae,
            "max_abs_diff": max_abs,
            "herbie_range": [float(np.nanmin(h_arr)), float(np.nanmax(h_arr))],
            "gik_range": [float(np.nanmin(g_arr)), float(np.nanmax(g_arr))],
            "step": step,
            "n_common_steps": len(common_steps),
        }

        print(f"  {field:6s}: r={r:.6f}  RMSE={rmse:.4e}  MAE={mae:.4e}  "
              f"MaxDiff={max_abs:.4e}  "
              f"H=[{np.nanmin(h_arr):.2f},{np.nanmax(h_arr):.2f}]  "
              f"G=[{np.nanmin(g_arr):.2f},{np.nanmax(g_arr):.2f}]")

        ds_h.close()
        ds_g.close()

    return results


# ==============================================================================
# INFERENCE: Run cGAN on both input sets
# ==============================================================================

def run_inference_on_path(
    netcdf_dir: Path,
    output_dir: Path,
    date_str: str,
    run: str,
    label: str,
) -> bool:
    """Run cGAN inference using the given input NetCDF directory."""
    print(f"\n  Running cGAN inference ({label})...")
    print(f"    Input:  {netcdf_dir}")
    print(f"    Output: {output_dir}")

    sys.path.insert(0, str(_SCRIPT_DIR))

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
        return True

    except Exception as e:
        print(f"    Inference failed: {e}")
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

    # Ensemble mean precipitation
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
                            date_str: str, run: str, label: str) -> Dict:
    """Compare cGAN output intensity against raw GEFS input."""
    print(f"\n  Intensity ratio ({label}):")

    year = date_str[:4]
    apcp_file = input_dir / f"apcp_{date_str}_{run}z.nc"
    gan_file = output_dir / year / f"GAN_{date_str}.nc"

    if not apcp_file.exists() or not gan_file.exists():
        missing = []
        if not apcp_file.exists():
            missing.append(f"apcp: {apcp_file}")
        if not gan_file.exists():
            missing.append(f"GAN: {gan_file}")
        print(f"    Missing: {missing}")
        return {"error": f"Missing files: {missing}"}

    ds_apcp = xr.open_dataset(apcp_file)
    ds_gan = xr.open_dataset(gan_file)

    # Raw GEFS: ensemble mean at first cGAN step
    apcp_var = list(ds_apcp.data_vars)[0]
    raw = ds_apcp[apcp_var].isel(time=0)
    if "member" in raw.dims:
        raw = raw.mean(dim="member")
    if "step" in raw.dims:
        raw = raw.isel(step=0)
    raw_arr = raw.values

    # cGAN: ensemble mean at first valid time
    gan = ds_gan["precipitation"].isel(time=0).mean(dim="member")
    if "valid_time" in gan.dims:
        gan = gan.isel(valid_time=0)
    gan_arr = gan.values

    raw_max = float(np.nanmax(raw_arr))
    gan_max = float(np.nanmax(gan_arr))
    raw_mean = float(np.nanmean(raw_arr))
    gan_mean = float(np.nanmean(gan_arr))

    ratio_max = gan_max / max(raw_max, 1e-12)
    ratio_mean = gan_mean / max(raw_mean, 1e-12)

    result = {
        "raw_max": raw_max,
        "gan_max": gan_max,
        "raw_mean": raw_mean,
        "gan_mean": gan_mean,
        "ratio_max": ratio_max,
        "ratio_mean": ratio_mean,
    }

    print(f"    Raw GEFS: max={raw_max:.2f} mm, mean={raw_mean:.4f} mm")
    print(f"    cGAN out: max={gan_max:.2f} mm, mean={gan_mean:.4f} mm")
    print(f"    Ratio (cGAN/raw): max={ratio_max:.2f}x, mean={ratio_mean:.2f}x")

    ds_apcp.close()
    ds_gan.close()
    return result


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_comparison(herbie_input_dir: Path, gik_input_dir: Path,
                    herbie_output_dir: Path, gik_output_dir: Path,
                    date_str: str, run: str, output_dir: Path):
    """Generate multi-panel comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    year = date_str[:4]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color scale
    precip_levels = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    precip_colors = [
        '#ffffff', '#c6dbef', '#9ecae1', '#6baed6',
        '#3182bd', '#08519c', '#ffff00', '#ff7f00', '#ff0000'
    ]
    cmap = mcolors.ListedColormap(precip_colors)
    norm = mcolors.BoundaryNorm(precip_levels, cmap.N)

    def add_map(ax):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.add_feature(cfeature.LAKES, alpha=0.3)

    # ── Row 1: Raw GEFS input (apcp) ──
    h_apcp_file = herbie_input_dir / f"apcp_{date_str}_{run}z.nc"
    g_apcp_file = gik_input_dir / f"apcp_{date_str}_{run}z.nc"

    h_input = g_input = None
    h_lats = h_lons = g_lats = g_lons = None

    if h_apcp_file.exists():
        ds = xr.open_dataset(h_apcp_file)
        var = list(ds.data_vars)[0]
        d = ds[var].isel(time=0)
        if "member" in d.dims:
            d = d.mean(dim="member")
        if "step" in d.dims:
            d = d.isel(step=0)
        h_input = d.values
        h_lats = ds.latitude.values
        h_lons = ds.longitude.values
        ds.close()

    if g_apcp_file.exists():
        ds = xr.open_dataset(g_apcp_file)
        var = list(ds.data_vars)[0]
        d = ds[var].isel(time=0)
        if "member" in d.dims:
            d = d.mean(dim="member")
        if "step" in d.dims:
            d = d.isel(step=0)
        g_input = d.values
        g_lats = ds.latitude.values
        g_lons = ds.longitude.values
        ds.close()

    # ── Row 2: cGAN outputs ──
    h_gan_file = herbie_output_dir / year / f"GAN_{date_str}.nc"
    g_gan_file = gik_output_dir / year / f"GAN_{date_str}.nc"

    h_output = g_output = None
    ho_lats = ho_lons = go_lats = go_lons = None

    if h_gan_file.exists():
        ds = xr.open_dataset(h_gan_file)
        d = ds["precipitation"].isel(time=0).mean(dim="member")
        if "valid_time" in d.dims:
            d = d.isel(valid_time=0)
        h_output = d.values
        ho_lats = ds.latitude.values
        ho_lons = ds.longitude.values
        ds.close()

    if g_gan_file.exists():
        ds = xr.open_dataset(g_gan_file)
        d = ds["precipitation"].isel(time=0).mean(dim="member")
        if "valid_time" in d.dims:
            d = d.isel(valid_time=0)
        g_output = d.values
        go_lats = ds.latitude.values
        go_lons = ds.longitude.values
        ds.close()

    # ── Create figure ──
    fig, axes = plt.subplots(
        2, 3, figsize=(22, 14),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Row 1: Raw inputs
    panels_r1 = [
        (h_input, h_lats, h_lons, "Raw GEFS (Herbie)"),
        (g_input, g_lats, g_lons, "Raw GEFS (GIK)"),
    ]
    for col, (data, lats, lons, title) in enumerate(panels_r1):
        ax = axes[0, col]
        if data is not None and lats is not None:
            ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree(), shading='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        add_map(ax)

    # Row 1, col 3: Difference
    ax = axes[0, 2]
    if h_input is not None and g_input is not None and h_input.shape == g_input.shape:
        diff = h_input - g_input
        vmax = max(np.nanmax(np.abs(diff)), 1e-6)
        ax.pcolormesh(h_lons, h_lats, diff, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax,
                      transform=ccrs.PlateCarree(), shading='auto')
    ax.set_title("Difference (Herbie - GIK)", fontsize=12, fontweight='bold')
    add_map(ax)

    # Row 2: cGAN outputs
    panels_r2 = [
        (h_output, ho_lats, ho_lons, "cGAN Output (Herbie path)"),
        (g_output, go_lats, go_lons, "cGAN Output (GIK path)"),
    ]
    for col, (data, lats, lons, title) in enumerate(panels_r2):
        ax = axes[1, col]
        if data is not None and lats is not None:
            ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree(), shading='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        add_map(ax)

    # Row 2, col 3: Output difference
    ax = axes[1, 2]
    if h_output is not None and g_output is not None and h_output.shape == g_output.shape:
        diff = h_output - g_output
        vmax = max(np.nanmax(np.abs(diff)), 1e-6)
        ax.pcolormesh(ho_lons, ho_lats, diff, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax,
                      transform=ccrs.PlateCarree(), shading='auto')
    ax.set_title("Difference (Herbie - GIK)", fontsize=12, fontweight='bold')
    add_map(ax)

    fig.suptitle(
        f"Herbie vs GIK cGAN Comparison — {date_str} {run}Z\n"
        f"Top: Raw GEFS APCP (ens. mean, +{STEP_HOURS[0]}h)  |  "
        f"Bottom: cGAN Downscaled (ens. mean, +{START_HOUR}h)",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / f"herbie_vs_gik_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Intensity ratio histogram ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (inp, out, label) in zip(axes2, [
        (h_input, h_output, "Herbie"),
        (g_input, g_output, "GIK"),
    ]):
        if inp is not None and out is not None:
            inp_flat = inp.flatten()
            out_flat = out.flatten()
            valid = (inp_flat > 0.01) & np.isfinite(inp_flat) & np.isfinite(out_flat)
            if valid.sum() > 0:
                ratios = out_flat[valid] / inp_flat[valid]
                ratios = ratios[ratios < 100]  # cap outliers
                ax.hist(ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(np.median(ratios), color='red', linestyle='--',
                           label=f'Median: {np.median(ratios):.2f}x')
                ax.legend()
        ax.set_xlabel("cGAN / Raw GEFS ratio")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} path — Intensity Ratio", fontweight='bold')

    plt.tight_layout()
    hist_path = output_dir / f"intensity_ratio_{date_str}.png"
    fig2.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {hist_path}")


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
    parser.add_argument("--max-members", type=int, default=5,
                        help="Max ensemble members (default: 5)")
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

    print("=" * 70)
    print("HERBIE vs GIK cGAN SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"Date: {date_str} {run}Z")
    print(f"Max members: {args.max_members}")
    print(f"Base dir: {base_dir}")
    print(f"Herbie inputs: {herbie_input_dir}")
    print(f"GIK inputs:    {gik_input_dir}")
    print(f"Skip fetch:    {args.skip_fetch or args.compare_only}")
    print(f"Skip inference:{args.skip_inference or args.compare_only}")
    print("=" * 70)

    pipeline_start = time.time()

    # ── Path A: Herbie fetch ──
    if not args.skip_fetch and not args.compare_only:
        t0 = time.time()
        herbie_ok = run_herbie_path(date_str, run, args.max_members, herbie_input_dir)
        print(f"\n  Herbie path: {'OK' if herbie_ok else 'FAILED'} in {time.time()-t0:.1f}s")
    else:
        print("\n  Herbie fetch: SKIPPED")

    # ── Path B: GIK pipeline ──
    if not args.skip_fetch and not args.compare_only:
        t0 = time.time()
        gik_ok = run_gik_path(date_str, run, args.max_members,
                              gik_input_dir, args.template)
        print(f"\n  GIK path: {'OK' if gik_ok else 'FAILED'} in {time.time()-t0:.1f}s")
    else:
        print("  GIK fetch: SKIPPED")

    # ── Pre-inference comparison ──
    input_results = compare_inputs(herbie_input_dir, gik_input_dir, date_str, run)

    # ── cGAN Inference ──
    if not args.skip_inference and not args.compare_only:
        print("\n" + "=" * 70)
        print("cGAN INFERENCE")
        print("=" * 70)

        # The inference expects input in {input_folder}/{YYYYMMDD}_{RUN}z/
        # Herbie path: herbie_input_dir is already {date}_{run}z
        h_inf_ok = run_inference_on_path(
            herbie_input_dir, herbie_output_dir, date_str, run, "Herbie"
        )
        g_inf_ok = run_inference_on_path(
            gik_input_dir, gik_output_dir, date_str, run, "GIK"
        )
    else:
        print("\n  cGAN Inference: SKIPPED")

    # ── Post-inference comparison ──
    output_results = compare_outputs(herbie_output_dir, gik_output_dir, date_str)

    # ── Intensity ratios ──
    print("\n" + "=" * 70)
    print("INTENSITY RATIO: cGAN vs Raw GEFS")
    print("=" * 70)
    herbie_ratio = compute_intensity_ratio(
        herbie_input_dir, herbie_output_dir, date_str, run, "Herbie"
    )
    gik_ratio = compute_intensity_ratio(
        gik_input_dir, gik_output_dir, date_str, run, "GIK"
    )

    # ── Plots ──
    try:
        plot_comparison(
            herbie_input_dir, gik_input_dir,
            herbie_output_dir, gik_output_dir,
            date_str, run, plots_dir,
        )
    except Exception as e:
        print(f"\n  Plot generation failed: {e}")

    # ── Save all results ──
    all_results = {
        "date": date_str,
        "run": run,
        "max_members": args.max_members,
        "input_comparison": input_results,
        "output_comparison": output_results,
        "intensity_ratio_herbie": herbie_ratio,
        "intensity_ratio_gik": gik_ratio,
    }

    base_dir.mkdir(parents=True, exist_ok=True)
    results_file = base_dir / f"comparison_results_{date_str}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_file}")

    # ── Summary ──
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\nPre-inference input correlation (should be ~1.0):")
    for field in ALL_FIELDS:
        r = input_results.get(field, {})
        if "error" in r:
            print(f"  {field:6s}: ERROR — {r['error']}")
        else:
            print(f"  {field:6s}: r={r.get('corr', 'N/A'):.6f}")

    if output_results and "error" not in output_results:
        print("\nPost-inference output correlation:")
        for key, val in output_results.items():
            if isinstance(val, dict):
                print(f"  {key}: r={val.get('corr', 'N/A'):.6f}  "
                      f"H_max={val.get('herbie_max', 0):.2f}  "
                      f"G_max={val.get('gik_max', 0):.2f}")

    print("\nIntensity ratios (cGAN_max / raw_GEFS_max):")
    for label, ratio in [("Herbie", herbie_ratio), ("GIK", gik_ratio)]:
        if "error" in ratio:
            print(f"  {label}: N/A")
        else:
            print(f"  {label}: {ratio.get('ratio_max', 0):.2f}x "
                  f"(cGAN {ratio.get('gan_max', 0):.2f} mm vs "
                  f"raw {ratio.get('raw_max', 0):.2f} mm)")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results: {results_file}")
    print("=" * 70)

    # Diagnostic conclusion
    if all(
        isinstance(input_results.get(f, {}), dict) and
        input_results.get(f, {}).get("corr", 0) > 0.99
        for f in ALL_FIELDS
        if "error" not in input_results.get(f, {})
    ):
        print("\nDIAGNOSTIC: Inputs match (r>0.99) → data pipeline is consistent.")
        h_ratio = herbie_ratio.get("ratio_max", 0)
        g_ratio = gik_ratio.get("ratio_max", 0)
        if isinstance(h_ratio, (int, float)) and isinstance(g_ratio, (int, float)):
            if h_ratio < 0.5 and g_ratio < 0.5:
                print("DIAGNOSTIC: Both paths show low intensity → model behavior issue.")
                print("  Next: investigate training data format, checkpoint, wind level.")
            elif abs(h_ratio - g_ratio) > 0.5:
                print("DIAGNOSTIC: Paths differ significantly → data pipeline issue.")
                print("  Next: compare intermediate normalization values.")


if __name__ == "__main__":
    main()
