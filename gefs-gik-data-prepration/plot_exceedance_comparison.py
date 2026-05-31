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
Exceedance probability comparison: GEFS raw vs cGAN downscaled precipitation.

For each cGAN valid time (+30h, +36h, +42h, +48h), computes the empirical
probability P(precip > T) across **raw individual ensemble members** for
thresholds T = 0.5, 1, 2, 5, 10, 20, 50 mm and plots GEFS vs cGAN side
by side.

Both GEFS and cGAN use every raw member independently — no ensemble mean
is taken before computing exceedance.  GEFS uses the individual step
accumulation at the lead time closest to each cGAN valid time.

Usage:
    uv run plot_exceedance_comparison.py --date 20240525
    uv run plot_exceedance_comparison.py --date 20240525 --input_dir gik_cgan_output
"""

import os
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

THRESHOLDS = [0.5, 1, 2, 5, 10, 20, 50]  # mm
VALID_TIME_HOURS = [30, 36, 42, 48]

# ICPAC region
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

# Probability colour scale
PROB_LEVELS = np.arange(0, 1.05, 0.1)
PROB_CMAP = "YlOrRd"


# ======================================================================
# Data loading
# ======================================================================
def load_gefs_precip(gefs_file):
    """Load GEFS apcp at individual steps (raw members, no averaging).

    Uses the first 4 of the 5 filtered steps [30, 36, 42, 48] to match
    the 4 cGAN valid times (+30h, +36h, +42h, +48h).

    Returns
    -------
    gefs_steps : list of DataArray  – one (member, lat, lon) per valid time
    lats, lons : 1-D arrays
    """
    print(f"Loading: {gefs_file}")
    ds = xr.open_dataset(gefs_file)
    apcp = ds["apcp"].isel(time=0)  # (member, step, lat, lon)

    n_members = int(apcp.sizes["member"])
    print(f"  Shape: {dict(apcp.sizes)}")
    print(f"  Steps: {apcp.step.values.tolist()}")
    print(f"  Members: {n_members}  (all used individually for exceedance)")
    print(f"  Range: [{float(apcp.min()):.1f}, {float(apcp.max()):.1f}] mm")

    gefs_steps = []
    n_vt = min(4, apcp.sizes["step"])
    for i in range(n_vt):
        step_data = apcp.isel(step=i).clip(min=0)
        gefs_steps.append(step_data)
        vt = VALID_TIME_HOURS[i]
        print(
            f"  +{vt}h  (step {int(apcp.step.values[i])}): "
            f"mean={float(step_data.mean()):.3f}  max={float(step_data.max()):.2f} mm"
        )

    lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
    lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
    return gefs_steps, lats, lons


def load_cgan_precip(cgan_file):
    """Load cGAN precipitation for each valid time.

    Returns
    -------
    cgan_vt : list of DataArray  – one (member, lat, lon) per valid time
    lats, lons : 1-D arrays
    """
    print(f"Loading: {cgan_file}")
    ds = xr.open_dataset(cgan_file)
    precip = ds["precipitation"].isel(time=0)  # (valid_time, member, lat, lon)

    n_members = int(precip.sizes["member"])
    print(f"  Shape: {dict(precip.sizes)}")
    print(f"  Members: {n_members}  (all used individually for exceedance)")
    print(f"  Range: [{float(precip.min()):.2f}, {float(precip.max()):.2f}] mm")

    cgan_vt = []
    for i in range(min(4, precip.sizes["valid_time"])):
        vt_data = precip.isel(valid_time=i).clip(min=0)
        cgan_vt.append(vt_data)
        vt = VALID_TIME_HOURS[i]
        print(
            f"  +{vt}h: mean={float(vt_data.mean()):.3f}  "
            f"max={float(vt_data.max()):.2f} mm"
        )

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    return cgan_vt, lats, lons


# ======================================================================
# Plotting
# ======================================================================
def _add_map_features(ax):
    """Add coastlines, borders, lakes, gridlines to a cartopy axis."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False


def plot_exceedance_for_valid_time(
    gefs_data, gefs_lats, gefs_lons,
    cgan_data, cgan_lats, cgan_lons,
    vt_idx, date_str, out_dir,
):
    """Create exceedance-probability maps for one valid time.

    Layout: 6 rows (thresholds) × 2 columns (GEFS, cGAN).
    """
    vt_h = VALID_TIME_HOURS[vt_idx]
    n_thr = len(THRESHOLDS)

    fig, axes = plt.subplots(
        n_thr, 2,
        figsize=(14, n_thr * 3.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    cmap = plt.get_cmap(PROB_CMAP)
    norm = mcolors.BoundaryNorm(PROB_LEVELS, cmap.N)

    n_gefs = int(gefs_data.sizes["member"])
    n_cgan = int(cgan_data.sizes["member"])

    for t, threshold in enumerate(THRESHOLDS):
        # GEFS exceedance: check each raw member independently, then fraction
        gefs_prob = (gefs_data > threshold).astype(float).mean(dim="member")
        ax = axes[t, 0]
        ax.pcolormesh(
            gefs_lons, gefs_lats, gefs_prob.values,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        )
        gefs_maxp = float(gefs_prob.max())
        ax.set_title(
            f"GEFS  P(precip > {threshold} mm)  [{n_gefs} mbr]  "
            f"max P={gefs_maxp:.2f}", fontsize=9,
        )
        _add_map_features(ax)

        # cGAN exceedance: check each raw member independently, then fraction
        cgan_prob = (cgan_data > threshold).astype(float).mean(dim="member")
        ax = axes[t, 1]
        ax.pcolormesh(
            cgan_lons, cgan_lats, cgan_prob.values,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        )
        cgan_maxp = float(cgan_prob.max())
        ax.set_title(
            f"cGAN  P(precip > {threshold} mm)  [{n_cgan} mbr]  "
            f"max P={cgan_maxp:.2f}", fontsize=9,
        )
        _add_map_features(ax)

    # Shared colour bar
    cbar_ax = fig.add_axes([0.15, 0.015, 0.7, 0.012])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation="horizontal",
    )
    cb.set_label("Exceedance Probability", fontsize=11)

    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    fig.suptitle(
        f"Precipitation Exceedance Probability\n"
        f"{date_fmt}  +{vt_h}h forecast   |   "
        f"GEFS ({n_gefs} members) vs cGAN ({n_cgan} members)",
        fontsize=13, y=1.0,
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.97])

    fname = os.path.join(out_dir, f"exceedance_{date_str}_+{vt_h:02d}h.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Exceedance probability: GEFS vs cGAN",
    )
    ap.add_argument("--date", default="20240525", help="Forecast date YYYYMMDD")
    ap.add_argument("--run", default="00", help="Model run (e.g. 00, 06, 12, 18)")
    ap.add_argument(
        "--input_dir", default="gik_cgan_output",
        help="Pipeline output root directory",
    )
    args = ap.parse_args()

    date_str = args.date
    run_id = args.run
    year = date_str[:4]
    base = os.path.join(_SCRIPT_DIR, args.input_dir)

    # New naming: netcdf/{YYYYMMDD}_{RUN}z/apcp_{YYYYMMDD}_{RUN}z.nc
    # Legacy fallback: netcdf/{year}/apcp_{year}.nc
    new_gefs = os.path.join(base, f"netcdf/{date_str}_{run_id}z/apcp_{date_str}_{run_id}z.nc")
    legacy_gefs = os.path.join(base, f"netcdf/{year}/apcp_{year}.nc")
    gefs_file = new_gefs if os.path.exists(new_gefs) else legacy_gefs

    cgan_file = os.path.join(base, f"cgan_output/{year}/GAN_{date_str}.nc")

    print("=" * 60)
    print("Exceedance Probability Comparison")
    print("=" * 60)
    print(f"Date       : {date_str}")
    print(f"Thresholds : {THRESHOLDS} mm")
    print(f"Valid times: +{', +'.join(str(h) for h in VALID_TIME_HOURS)}h")
    print()

    print("--- GEFS (raw individual members) ---")
    gefs_6h, gefs_lats, gefs_lons = load_gefs_precip(gefs_file)
    print()

    print("--- cGAN (raw individual members) ---")
    cgan_vt, cgan_lats, cgan_lons = load_cgan_precip(cgan_file)
    print()

    out_dir = os.path.join(_SCRIPT_DIR, "exceedance_plots")
    os.makedirs(out_dir, exist_ok=True)

    n_vt = min(len(gefs_6h), len(cgan_vt))
    for vt_idx in range(n_vt):
        print(f"Plotting +{VALID_TIME_HOURS[vt_idx]}h ...")
        plot_exceedance_for_valid_time(
            gefs_6h[vt_idx], gefs_lats, gefs_lons,
            cgan_vt[vt_idx], cgan_lats, cgan_lons,
            vt_idx, date_str, out_dir,
        )

    print(f"\nDone – {n_vt} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
