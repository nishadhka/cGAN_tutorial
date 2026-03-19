#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "numpy<2.0",
#     "matplotlib",
#     "cartopy",
#     "xarray",
#     "netcdf4",
#     "scipy",
# ]
# ///
"""
Plot Herbie vs GIK cGAN comparison with GEFS converted to mm/h.

All panels use the same unit (mm/h) for direct visual comparison:
  - Raw GEFS APCP: divided by 6 to convert from mm/6h-bucket to mm/h
  - cGAN output: already in mm/h

Layout per valid time:
  Row 0: Raw GEFS (mm/h) — Herbie | GIK | Difference    [shared colorbar]
  Row 1: cGAN output (mm/h) — Herbie | GIK | Difference  [shared colorbar]

Also generates:
  - Scatter plot (raw GEFS mm/h vs cGAN mm/h) with 1:1 line
  - Percentile comparison plot

Usage:
    uv run plot_herbie_vs_gik_mmh.py
    uv run plot_herbie_vs_gik_mmh.py --date 20240520 --base-dir herbie_vs_gik_test
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

HOURS = 6
START_HOUR = 30
END_HOUR = 54
BUCKET_HOURS = 6  # GEFS APCP bucket duration

LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

# Shared color scale for both raw GEFS and cGAN (mm/h)
PRECIP_LEVELS = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
PRECIP_COLORS = [
    '#ffffff', '#edf8e9', '#c7e9c0', '#a1d99b',
    '#74c476', '#31a354', '#006d2c', '#ffff00', '#ff7f00', '#ff0000'
]


# ── Helpers ───────────────────────────────────────────────────────────

def add_map(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.LAKES, alpha=0.3)


def load_apcp_all_mmh(path):
    """Load APCP, convert from mm/6h-bucket to mm/h, return (steps, lat, lon) arrays."""
    if not path.exists():
        return None, None, None, None
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    d = ds[var].isel(time=0)
    if "member" in d.dims:
        d = d.mean(dim="member")
    lats, lons = ds.latitude.values, ds.longitude.values
    steps = [int(s) for s in ds.step.values]
    vals = d.values / BUCKET_HOURS  # mm/6h → mm/h
    ds.close()
    return vals, lats, lons, steps


def load_gan_all(path):
    """Load cGAN output (already in mm/h), return (valid_times, lat, lon) arrays."""
    if not path.exists():
        return None, None, None, None
    ds = xr.open_dataset(path)
    d = ds["precipitation"].isel(time=0).mean(dim="member")
    lats, lons = ds.latitude.values, ds.longitude.values
    n_vt = d.sizes.get("valid_time", 1)
    vals = d.values  # (valid_time, lat, lon) or (lat, lon)
    ds.close()
    return vals, lats, lons, n_vt


def load_gan_single_member(path, member_idx=0):
    """Load single cGAN member (mm/h) for scatter comparison."""
    if not path.exists():
        return None, None, None, None
    ds = xr.open_dataset(path)
    d = ds["precipitation"].isel(time=0, member=member_idx)
    lats, lons = ds.latitude.values, ds.longitude.values
    n_vt = d.sizes.get("valid_time", 1)
    vals = d.values
    ds.close()
    return vals, lats, lons, n_vt


# ── Per-timestep map plots ────────────────────────────────────────────

def plot_maps_per_timestep(herbie_input_dir, gik_input_dir,
                           herbie_output_dir, gik_output_dir,
                           date_str, run, output_dir):
    """Generate per-timestep 2×3 comparison maps, all in mm/h."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = mcolors.ListedColormap(PRECIP_COLORS)
    norm = mcolors.BoundaryNorm(PRECIP_LEVELS, cmap.N)

    year = date_str[:4]

    h_raw, h_lats, h_lons, h_steps = load_apcp_all_mmh(
        herbie_input_dir / f"apcp_{date_str}_{run}z.nc")
    g_raw, g_lats, g_lons, g_steps = load_apcp_all_mmh(
        gik_input_dir / f"apcp_{date_str}_{run}z.nc")
    h_gan, ho_lats, ho_lons, h_nvt = load_gan_all(
        herbie_output_dir / year / f"GAN_{date_str}.nc")
    g_gan, go_lats, go_lons, g_nvt = load_gan_all(
        gik_output_dir / year / f"GAN_{date_str}.nc")

    n_vt = 0
    if h_gan is not None:
        n_vt = h_gan.shape[0] if h_gan.ndim == 3 else 1
    elif g_gan is not None:
        n_vt = g_gan.shape[0] if g_gan.ndim == 3 else 1

    for vt_idx in range(n_vt):
        hour = START_HOUR + vt_idx * HOURS

        # Extract this timestep
        def get(arr, idx):
            if arr is None: return None
            if arr.ndim == 3: return arr[idx] if idx < arr.shape[0] else None
            return arr if idx == 0 else None

        hr = get(h_raw, vt_idx)
        gr = get(g_raw, vt_idx)
        hg = get(h_gan, vt_idx)
        gg = get(g_gan, vt_idx)

        fig = plt.figure(figsize=(24, 14))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.04],
                               hspace=0.25, wspace=0.12)

        # ── Row 0: Raw GEFS in mm/h ──
        im_raw = None
        for col, (data, lats, lons, src) in enumerate([
            (hr, h_lats, h_lons, "Herbie"),
            (gr, g_lats, g_lons, "GIK"),
        ]):
            ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
            if data is not None:
                mx = float(np.nanmax(data))
                mn = float(np.nanmean(data))
                im_raw = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                                       transform=ccrs.PlateCarree(), shading='auto')
                ax.set_title(f"Raw GEFS ({src})\nmax={mx:.2f} mm/h  mean={mn:.4f} mm/h",
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"Raw GEFS ({src})\n(no data)", fontsize=11)
            add_map(ax)

        # Difference
        ax = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        if hr is not None and gr is not None and hr.shape == gr.shape:
            diff = hr - gr
            vmax = max(float(np.nanmax(np.abs(diff))), 1e-6)
            ax.pcolormesh(h_lons, h_lats, diff, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')
            ax.set_title(f"Diff (H-G)\nmax|diff|={vmax:.4f} mm/h",
                         fontsize=11, fontweight='bold')
        else:
            ax.set_title("Diff (H-G)", fontsize=11)
        add_map(ax)

        # Colorbar
        if im_raw is not None:
            cax = fig.add_subplot(gs[0, 3])
            cb = fig.colorbar(im_raw, cax=cax, extend='max')
            cb.set_label("Raw GEFS APCP (mm/h)", fontsize=10)

        # ── Row 1: cGAN output in mm/h ──
        im_cgan = None
        for col, (data, lats, lons, src) in enumerate([
            (hg, ho_lats, ho_lons, "Herbie"),
            (gg, go_lats, go_lons, "GIK"),
        ]):
            ax = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree())
            if data is not None:
                mx = float(np.nanmax(data))
                mn = float(np.nanmean(data))
                im_cgan = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                                        transform=ccrs.PlateCarree(), shading='auto')
                ax.set_title(f"cGAN ({src})\nmax={mx:.2f} mm/h  mean={mn:.4f} mm/h",
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"cGAN ({src})\n(no data)", fontsize=11)
            add_map(ax)

        # Difference
        ax = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
        if hg is not None and gg is not None and hg.shape == gg.shape:
            diff = hg - gg
            vmax = max(float(np.nanmax(np.abs(diff))), 1e-6)
            ax.pcolormesh(ho_lons, ho_lats, diff, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')
            ax.set_title(f"Diff (H-G)\nmax|diff|={vmax:.4f} mm/h",
                         fontsize=11, fontweight='bold')
        else:
            ax.set_title("Diff (H-G)", fontsize=11)
        add_map(ax)

        # Colorbar
        if im_cgan is not None:
            cax = fig.add_subplot(gs[1, 3])
            cb = fig.colorbar(im_cgan, cax=cax, extend='max')
            cb.set_label("cGAN Precipitation (mm/h)", fontsize=10)

        fig.suptitle(
            f"Herbie vs GIK — {date_str} {run}Z — Valid +{hour}h   [ALL values in mm/h]\n"
            f"Top: Raw GEFS APCP ÷ 6  |  Bottom: cGAN Downscaled  |  Same color scale",
            fontsize=14, fontweight='bold', y=0.99)

        out_path = output_dir / f"mmh_comparison_{date_str}_T{hour:03d}h.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out_path}")


# ── Scatter plot: raw GEFS mm/h vs cGAN mm/h ─────────────────────────

def plot_scatter(herbie_input_dir, herbie_output_dir, gik_input_dir, gik_output_dir,
                 date_str, run, output_dir):
    """Scatter plot of raw GEFS rate vs cGAN output, both in mm/h."""
    from scipy import stats

    output_dir.mkdir(parents=True, exist_ok=True)
    year = date_str[:4]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (input_dir, output_dir_path, label) in zip(axes, [
        (herbie_input_dir, herbie_output_dir, "Herbie"),
        (gik_input_dir, gik_output_dir, "GIK"),
    ]):
        raw_all, r_lats, r_lons, r_steps = load_apcp_all_mmh(
            input_dir / f"apcp_{date_str}_{run}z.nc")
        gan_all, g_lats, g_lons, g_nvt = load_gan_all(
            output_dir_path / year / f"GAN_{date_str}.nc")

        if raw_all is None or gan_all is None:
            ax.set_title(f"{label} — no data")
            continue

        # Match timesteps — raw GEFS and cGAN have different grids
        # (GEFS: 153×141 at 0.25°, cGAN: 384×352 at ~0.1°)
        # Compare via sorted percentiles, not pixel-to-pixel
        n_match = min(
            raw_all.shape[0] if raw_all.ndim == 3 else 1,
            gan_all.shape[0] if gan_all.ndim == 3 else 1)

        all_raw, all_gan = [], []
        n_percentile_pts = 5000  # sample density for QQ-style scatter
        for vt in range(n_match):
            r = (raw_all[vt] if raw_all.ndim == 3 else raw_all).flatten()
            g = (gan_all[vt] if gan_all.ndim == 3 else gan_all).flatten()
            r = r[np.isfinite(r)]
            g = g[np.isfinite(g)]
            # Use matched percentiles since grids differ
            pcts = np.linspace(0, 100, n_percentile_pts)
            r_pct = np.percentile(r, pcts)
            g_pct = np.percentile(g, pcts)
            ok = r_pct > 0.001
            all_raw.extend(r_pct[ok])
            all_gan.extend(g_pct[ok])

        all_raw = np.array(all_raw)
        all_gan = np.array(all_gan)

        if len(all_raw) == 0:
            ax.set_title(f"{label} — no valid data")
            continue

        ax.scatter(all_raw, all_gan, s=1, alpha=0.3, c='steelblue', rasterized=True)

        lim = max(np.nanmax(all_raw), np.nanmax(all_gan)) * 1.1
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='1:1 line')

        # Fit
        r_val, _ = stats.pearsonr(all_raw, all_gan)
        slope, intercept = np.polyfit(all_raw, all_gan, 1)
        x_fit = np.linspace(0, lim, 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'k-', linewidth=1,
                label=f'Fit: y={slope:.2f}x+{intercept:.3f}')

        ax.set_xlabel("Raw GEFS APCP (mm/h)", fontsize=11)
        ax.set_ylabel("cGAN Precipitation (mm/h)", fontsize=11)
        ax.set_title(f"{label} — Raw GEFS vs cGAN (all valid times)\n"
                     f"r={r_val:.4f}  slope={slope:.2f}  N={len(all_raw):,}",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')

    plt.tight_layout()
    out_path = output_dir / f"scatter_mmh_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Percentile comparison plot ────────────────────────────────────────

def plot_percentiles(herbie_input_dir, herbie_output_dir,
                     date_str, run, output_dir):
    """Percentile-by-percentile comparison of raw GEFS (mm/h) vs cGAN (mm/h)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    year = date_str[:4]

    raw_all, _, _, _ = load_apcp_all_mmh(
        herbie_input_dir / f"apcp_{date_str}_{run}z.nc")
    gan_all, _, _, _ = load_gan_all(
        herbie_output_dir / year / f"GAN_{date_str}.nc")

    if raw_all is None or gan_all is None:
        print("  Cannot generate percentile plot — missing data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    n_match = min(
        raw_all.shape[0] if raw_all.ndim == 3 else 1,
        gan_all.shape[0] if gan_all.ndim == 3 else 1)

    for vt_idx in range(n_match):
        r = (raw_all[vt_idx] if raw_all.ndim == 3 else raw_all).flatten()
        g = (gan_all[vt_idx] if gan_all.ndim == 3 else gan_all).flatten()
        r = r[np.isfinite(r)]
        g = g[np.isfinite(g)]

        hour = START_HOUR + vt_idx * HOURS
        pcts = np.arange(50, 100.1, 0.5)
        r_pct = np.percentile(r, pcts)
        g_pct = np.percentile(g, pcts)

        # Panel 1: QQ plot
        axes[0].plot(r_pct, g_pct, 'o-', markersize=2, label=f'+{hour}h', alpha=0.7)

    lim = max(np.nanmax(r_pct), np.nanmax(g_pct)) * 1.1
    axes[0].plot([0, lim], [0, lim], 'k--', linewidth=1)
    axes[0].set_xlabel("Raw GEFS percentile (mm/h)", fontsize=11)
    axes[0].set_ylabel("cGAN percentile (mm/h)", fontsize=11)
    axes[0].set_title("QQ Plot: Raw GEFS vs cGAN\n(P50–P100, both mm/h)",
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_aspect('equal')

    # Panel 2: Ratio by percentile (using Herbie +30h)
    r0 = (raw_all[0] if raw_all.ndim == 3 else raw_all).flatten()
    g0 = (gan_all[0] if gan_all.ndim == 3 else gan_all).flatten()
    r0 = r0[np.isfinite(r0)]
    g0 = g0[np.isfinite(g0)]

    pcts = np.arange(10, 100.1, 1)
    r_pct = np.percentile(r0, pcts)
    g_pct = np.percentile(g0, pcts)
    ratio = np.where(r_pct > 0.001, g_pct / r_pct, np.nan)

    axes[1].plot(pcts, ratio, 'b-', linewidth=2)
    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1, label='1:1')
    axes[1].set_xlabel("Percentile", fontsize=11)
    axes[1].set_ylabel("Ratio (cGAN / raw GEFS)", fontsize=11)
    axes[1].set_title(f"Intensity Ratio by Percentile (+{START_HOUR}h)\n"
                      f"Both in mm/h — ratio=1 means perfect match",
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 3)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Bar chart of key percentiles
    key_pcts = [50, 75, 90, 95, 99, 100]
    r_vals = [np.percentile(r0, p) if p < 100 else np.max(r0) for p in key_pcts]
    g_vals = [np.percentile(g0, p) if p < 100 else np.max(g0) for p in key_pcts]

    x = np.arange(len(key_pcts))
    w = 0.35
    axes[2].bar(x - w/2, r_vals, w, label='Raw GEFS (mm/h)', color='steelblue', alpha=0.8)
    axes[2].bar(x + w/2, g_vals, w, label='cGAN (mm/h)', color='forestgreen', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'P{p}' for p in key_pcts])
    axes[2].set_ylabel("Precipitation (mm/h)", fontsize=11)
    axes[2].set_title(f"Key Percentiles (+{START_HOUR}h)\nBoth in mm/h",
                      fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)

    # Add ratio text
    for i, (rv, gv) in enumerate(zip(r_vals, g_vals)):
        if rv > 0.001:
            axes[2].text(i, max(rv, gv) * 1.05, f'{gv/rv:.2f}x',
                         ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = output_dir / f"percentile_mmh_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Herbie vs GIK cGAN comparison with GEFS in mm/h"
    )
    parser.add_argument("--date", type=str, default="20240520",
                        help="Target date (YYYYMMDD)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour")
    parser.add_argument("--base-dir", type=str, default=str(_SCRIPT_DIR / "herbie_vs_gik_test"),
                        help="Base directory with Herbie/GIK data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: {base-dir}/plots_mmh)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    date_str = args.date
    run = args.run
    year = date_str[:4]

    herbie_input_dir = base / "herbie" / f"{date_str}_{run}z"
    gik_input_dir = base / "gik" / "netcdf" / f"{date_str}_{run}z"
    herbie_output_dir = base / "herbie_cgan"
    gik_output_dir = base / "gik_cgan"
    output_dir = Path(args.output_dir) if args.output_dir else base / "plots_mmh"

    print("=" * 70)
    print(f"Herbie vs GIK cGAN Comparison — ALL values in mm/h")
    print(f"Date: {date_str} {run}Z")
    print(f"GEFS APCP converted: mm/6h-bucket ÷ 6 → mm/h")
    print("=" * 70)

    print("\n[1] Per-timestep comparison maps")
    plot_maps_per_timestep(herbie_input_dir, gik_input_dir,
                           herbie_output_dir, gik_output_dir,
                           date_str, run, output_dir)

    print("\n[2] Scatter plot (raw GEFS mm/h vs cGAN mm/h)")
    plot_scatter(herbie_input_dir, herbie_output_dir,
                 gik_input_dir, gik_output_dir,
                 date_str, run, output_dir)

    print("\n[3] Percentile comparison")
    plot_percentiles(herbie_input_dir, herbie_output_dir,
                     date_str, run, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
