#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "numpy<2.0",
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "netcdf4",
# ]
# ///
"""
Plot comparison between input GEFS precipitation and cGAN downscaled output.
Overlays East Africa country boundaries using cartopy.

Usage:
    uv run plot_cgan_comparison.py                        # single +30h step
    uv run plot_cgan_comparison.py --mode daily_total     # 18h accumulated total
"""

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================================================================
# CONFIGURATION
# ==============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input GEFS data (from pipeline Stage 4)
INPUT_DIR = os.path.join(_SCRIPT_DIR, "gik_cgan_output/netcdf/20240520_00z")
INPUT_FILE = os.path.join(INPUT_DIR, "apcp_20240520_00z.nc")

# cGAN output (from pipeline Stage 5)
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "gik_cgan_output/cgan_output/2024")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GAN_20240520.nc")

# ICPAC region
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

# Color scale for single-step plots (mm / 6h)
PRECIP_LEVELS = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
PRECIP_COLORS = [
    '#ffffff', '#c6dbef', '#9ecae1', '#6baed6',
    '#3182bd', '#08519c', '#ffff00', '#ff7f00', '#ff0000'
]

# Color scale for daily-total plots (mm / 18h)
DAILY_LEVELS = [0, 1, 5, 10, 20, 40, 60, 80, 100, 150]
DAILY_COLORS = [
    '#ffffff', '#c6dbef', '#9ecae1', '#6baed6',
    '#3182bd', '#08519c', '#ffff00', '#ff7f00', '#ff0000'
]


# ==============================================================================
# SINGLE-STEP LOADERS  (existing behaviour)
# ==============================================================================

def load_input_data():
    """Load raw GEFS apcp data — member 1, first step (+30h)."""
    print(f"Loading input: {INPUT_FILE}")
    ds = xr.open_dataset(INPUT_FILE)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dims: {dict(ds.sizes)}")

    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
    lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values

    if 'time' in data.dims:
        data = data.isel(time=0)
    if 'member' in data.dims:
        data = data.isel(member=0)
    if 'step' in data.dims:
        data = data.isel(step=0)  # step=0 → +30h

    print(f"  Selected shape: {data.shape}")
    return data.values, lats, lons


def load_output_data():
    """Load cGAN output precipitation — ensemble mean, first valid time (+30h)."""
    print(f"Loading output: {OUTPUT_FILE}")
    ds = xr.open_dataset(OUTPUT_FILE)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dims: {dict(ds.sizes)}")

    data = ds['precipitation']
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    if 'time' in data.dims:
        data = data.isel(time=0)
    if 'valid_time' in data.dims:
        data = data.isel(valid_time=0)
    if 'member' in data.dims:
        data = data.mean(dim='member')

    print(f"  Selected shape: {data.shape}")
    return data.values, lats, lons


# ==============================================================================
# DAILY-TOTAL LOADERS
# ==============================================================================

def load_input_data_daily_total():
    """GEFS 18h accumulated total: ensemble mean of (apcp[+48h] - apcp[+30h]).

    The cumulative APCP at each step already represents total precipitation from
    hour 0, so subtracting step 30 from step 48 gives the 18h window total.
    Step +54h is excluded because it contains NaN for this run.
    """
    print(f"Loading input (daily total): {INPUT_FILE}")
    ds = xr.open_dataset(INPUT_FILE)
    apcp = ds['apcp']
    if 'time' in apcp.dims:
        apcp = apcp.isel(time=0)

    lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
    lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values

    # Drop the step axis; cumulative difference gives 18h total per member
    total = apcp.sel(step=48) - apcp.sel(step=30)   # (member, lat, lon)
    total = total.mean(dim='member')                  # ensemble mean

    total_vals = np.where(np.isfinite(total.values), total.values, 0.0)
    print(f"  18h total — max: {total_vals.max():.2f} mm, mean: {total_vals.mean():.3f} mm")
    print(f"  Selected shape: {total_vals.shape}")
    return total_vals, lats, lons


def load_output_data_daily_total():
    """cGAN 18h accumulated total: ensemble mean of sum over first 3 valid times.

    Each valid_time slice is a 6-hour precipitation total (mm).
    Summing valid_time indices 0–2 matches the GEFS 30→48h window.
    """
    print(f"Loading output (daily total): {OUTPUT_FILE}")
    ds = xr.open_dataset(OUTPUT_FILE)
    data = ds['precipitation']
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    if 'time' in data.dims:
        data = data.isel(time=0)

    # Sum first 3 of 4 valid times (30–36h, 36–42h, 42–48h) → 18h total
    total = data.isel(valid_time=slice(0, 3)).sum(dim='valid_time')  # (member, lat, lon)
    total = total.mean(dim='member')                                   # ensemble mean

    print(f"  18h total — max: {float(total.max()):.2f} mm, mean: {float(total.mean()):.3f} mm")
    print(f"  Selected shape: {total.shape}")
    return total.values, lats, lons


# ==============================================================================
# PLOT HELPERS
# ==============================================================================

def _add_map_features(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)


# ==============================================================================
# PLOT: SINGLE-STEP
# ==============================================================================

def plot_comparison():
    """Side-by-side comparison: GEFS member-1 +30h vs cGAN ensemble-mean +30h."""
    cmap = mcolors.ListedColormap(PRECIP_COLORS)
    norm = mcolors.BoundaryNorm(PRECIP_LEVELS, cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Panel 1: GEFS
    ax1 = axes[0]
    try:
        input_data, input_lats, input_lons = load_input_data()
        im1 = ax1.pcolormesh(
            input_lons, input_lats, input_data,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
        ax1.set_title('GEFS Input (apcp, member 1, +30h)', fontsize=14)
    except Exception as e:
        ax1.text(0.5, 0.5, f'Input not available:\n{e}',
                 transform=ax1.transAxes, ha='center', va='center', fontsize=10)
        ax1.set_title('GEFS Input (not available)', fontsize=14)
        im1 = None

    # Panel 2: cGAN
    ax2 = axes[1]
    try:
        output_data, output_lats, output_lons = load_output_data()
        im2 = ax2.pcolormesh(
            output_lons, output_lats, output_data,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
        ax2.set_title('cGAN Downscaled (ensemble mean, +30h)', fontsize=14)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Output not available:\n{e}',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.set_title('cGAN Output (not available)', fontsize=14)
        im2 = None

    for ax in axes:
        _add_map_features(ax)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    ref = im2 if im2 is not None else im1
    if ref is not None:
        cb = fig.colorbar(ref, cax=cbar_ax, orientation='horizontal', extend='max')
        cb.set_label('Precipitation (mm / 6h)', fontsize=12)

    fig.suptitle('GEFS vs cGAN Precipitation — 2024-05-20  (+30h forecast)',
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_path = os.path.join(_SCRIPT_DIR, "cgan_comparison_20240520.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")
    plt.close()


# ==============================================================================
# PLOT: DAILY TOTAL
# ==============================================================================

def plot_daily_total():
    """Side-by-side: GEFS vs cGAN ensemble-mean 18h accumulated total (30→48h)."""
    cmap = mcolors.ListedColormap(DAILY_COLORS)
    norm = mcolors.BoundaryNorm(DAILY_LEVELS, cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Panel 1: GEFS 18h total
    ax1 = axes[0]
    try:
        input_data, input_lats, input_lons = load_input_data_daily_total()
        im1 = ax1.pcolormesh(
            input_lons, input_lats, input_data,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
        ax1.set_title('GEFS 18h Total (ens. mean, +30h → +48h)', fontsize=13)
    except Exception as e:
        ax1.text(0.5, 0.5, f'Input not available:\n{e}',
                 transform=ax1.transAxes, ha='center', va='center', fontsize=10)
        ax1.set_title('GEFS Input (not available)', fontsize=13)
        im1 = None

    # Panel 2: cGAN 18h total
    ax2 = axes[1]
    try:
        output_data, output_lats, output_lons = load_output_data_daily_total()
        im2 = ax2.pcolormesh(
            output_lons, output_lats, output_data,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
        ax2.set_title('cGAN 18h Total (ens. mean, +30h → +48h)', fontsize=13)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Output not available:\n{e}',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.set_title('cGAN Output (not available)', fontsize=13)
        im2 = None

    for ax in axes:
        _add_map_features(ax)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    ref = im2 if im2 is not None else im1
    if ref is not None:
        cb = fig.colorbar(ref, cax=cbar_ax, orientation='horizontal', extend='max')
        cb.set_label('Accumulated Precipitation (mm)', fontsize=12)

    fig.suptitle(
        'GEFS vs cGAN — 18h Accumulated Precipitation  '
        '(2024-05-20 00z,  valid +30h → +48h)',
        fontsize=15, y=0.98
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_path = os.path.join(_SCRIPT_DIR, "cgan_daily_total_20240520.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot GEFS vs cGAN precipitation comparison"
    )
    parser.add_argument(
        '--mode',
        choices=['single_step', 'daily_total'],
        default='single_step',
        help=(
            "single_step: GEFS member-1 vs cGAN ens-mean at +30h (default)  |  "
            "daily_total: ensemble-mean 18h accumulated total (+30h to +48h)"
        )
    )
    args = parser.parse_args()

    if args.mode == 'daily_total':
        plot_daily_total()
    else:
        plot_comparison()
