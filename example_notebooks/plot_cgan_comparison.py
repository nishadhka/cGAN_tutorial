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
    uv run plot_cgan_comparison.py
"""

import os
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
INPUT_DIR = os.path.join(_SCRIPT_DIR, "gik_cgan_output/netcdf/2025")
INPUT_FILE = os.path.join(INPUT_DIR, "apcp_2025.nc")

# cGAN output (from pipeline Stage 5)
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "gik_cgan_output/cgan_output/2025")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GAN_20250918.nc")

# ICPAC region
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

# Precipitation color scale
PRECIP_LEVELS = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
PRECIP_COLORS = [
    '#ffffff', '#c6dbef', '#9ecae1', '#6baed6',
    '#3182bd', '#08519c', '#ffff00', '#ff7f00', '#ff0000'
]


def load_input_data():
    """Load raw GEFS apcp data."""
    print(f"Loading input: {INPUT_FILE}")
    ds = xr.open_dataset(INPUT_FILE)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dims: {dict(ds.sizes)}")

    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
    lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values

    # Select first time
    if 'time' in data.dims:
        data = data.isel(time=0)
    # Select first member
    if 'member' in data.dims:
        data = data.isel(member=0)
    # Select step ~30h (index 10 for 3-hourly, or 30 for hourly)
    if 'step' in data.dims:
        n_steps = data.sizes['step']
        step_idx = min(10, n_steps - 1)  # 3-hourly -> step 10 = 30h
        data = data.isel(step=step_idx)

    print(f"  Selected shape: {data.shape}")
    return data.values, lats, lons


def load_output_data():
    """Load cGAN output precipitation."""
    print(f"Loading output: {OUTPUT_FILE}")
    ds = xr.open_dataset(OUTPUT_FILE)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dims: {dict(ds.sizes)}")

    data = ds['precipitation']
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Select first time
    if 'time' in data.dims:
        data = data.isel(time=0)
    # Select first valid_time
    if 'valid_time' in data.dims:
        data = data.isel(valid_time=0)
    # Ensemble mean
    if 'member' in data.dims:
        data = data.mean(dim='member')

    print(f"  Selected shape: {data.shape}")
    return data.values, lats, lons


def plot_comparison():
    """Create side-by-side comparison plot."""
    cmap = mcolors.ListedColormap(PRECIP_COLORS)
    norm = mcolors.BoundaryNorm(PRECIP_LEVELS, cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # --- Panel 1: Input GEFS ---
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

    # --- Panel 2: cGAN Output ---
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

    # Add map features to both panels
    for ax in axes:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.add_feature(cfeature.LAKES, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    if im2 is not None:
        cb = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal', extend='max')
    elif im1 is not None:
        cb = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', extend='max')
    else:
        cb = None
    if cb:
        cb.set_label('Precipitation (mm/h)', fontsize=12)

    fig.suptitle('GEFS vs cGAN Precipitation - 2025-09-18 (+30h forecast)',
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_path = os.path.join(_SCRIPT_DIR, "cgan_comparison_20250918.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_comparison()
