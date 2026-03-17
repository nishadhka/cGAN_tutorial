#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "herbie-data",
#     "xarray",
#     "numpy",
#     "netcdf4",
#     "cfgrib",
#     "eccodes",
# ]
# ///
"""
Fetch all 8 cGAN input variables from GEFS via Herbie for side-by-side comparison.

Downloads ensemble data for all 8 cGAN fields:
    cape, pres (sp), pwat, tmp (t2m), ugrd (u10), vgrd (v10), msl (mslet), apcp (tp)

Outputs per-variable NetCDF files in the same layout as GIK pipeline Stage 4:
    {var}_{date}_{run}z.nc  with dims (time, member, step, latitude, longitude)

Usage:
    uv run fetch_gefs_herbie_for_cgan.py --date 20250106 --max-members 5
    uv run fetch_gefs_herbie_for_cgan.py --date 20250106 --output-dir herbie_cgan_output
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
OUTPUT_DIR = Path("herbie_cgan_output")
LOG_FILE = OUTPUT_DIR / "herbie_cgan_log.txt"

# cGAN forecast hours: 30-54 in 6h steps → need hours 30,36,42,48,54,60
# (each valid time window uses step_1=vt and step_2=vt+6)
HOURS = 6
START_HOUR = 30
END_HOUR = 54
FORECAST_HOURS = list(range(START_HOUR, END_HOUR + HOURS + 1, HOURS))  # [30,36,42,48,54,60]

# ICPAC region (matching run_gefs_inference_raw.py / stream_gefs_for_cgan.py)
LAT_MIN, LAT_MAX = -13.65, 24.65
LON_MIN, LON_MAX = 19.15, 54.25

# cGAN variable mapping: cgan_name -> Herbie search pattern
# GEFS Herbie uses model="gefs", product="atmos", fxx=hour
CGAN_VARIABLES = {
    "cape": {"search": ":CAPE:surface:", "gefs_name": "CAPE"},
    "pres": {"search": ":PRES:surface:", "gefs_name": "PRES"},
    "pwat": {"search": ":PWAT:entire atmosphere", "gefs_name": "PWAT"},
    "tmp":  {"search": ":TMP:2 m above ground:", "gefs_name": "TMP"},
    "ugrd": {"search": ":UGRD:10 m above ground:", "gefs_name": "UGRD"},
    "vgrd": {"search": ":VGRD:10 m above ground:", "gefs_name": "VGRD"},
    "msl":  {"search": ":MSLET:mean sea level:", "gefs_name": "MSLET"},
    "apcp": {"search": ":APCP:surface:", "gefs_name": "APCP"},
}

# GEFS grid (0.25 degree)
GEFS_NLATS, GEFS_NLONS = 721, 1440


# ── Helpers ───────────────────────────────────────────────────────────

def setup_logging(out_dir: Path, log_path: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("herbie_cgan")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def subset_icpac(ds: xr.Dataset) -> xr.Dataset:
    """Subset an xarray Dataset to the ICPAC region."""
    lat_name = "latitude"
    lon_name = "longitude"

    # Handle longitude: GEFS uses 0-360
    lons = ds[lon_name].values
    if lons.max() > 180:
        ds = ds.assign_coords(
            {lon_name: ((ds[lon_name] + 180) % 360) - 180}
        )
        ds = ds.sortby(lon_name)

    # Handle latitude ordering (GEFS descending 90 -> -90)
    lats = ds[lat_name].values
    if lats[0] > lats[-1]:
        ds = ds.sel(**{
            lat_name: slice(LAT_MAX, LAT_MIN),
            lon_name: slice(LON_MIN, LON_MAX),
        })
    else:
        ds = ds.sel(**{
            lat_name: slice(LAT_MIN, LAT_MAX),
            lon_name: slice(LON_MIN, LON_MAX),
        })
    return ds


def fetch_single_field_herbie(
    date_str: str,
    run: str,
    fxx: int,
    member: int,
    search_pattern: str,
) -> Optional[np.ndarray]:
    """Fetch a single field for one member at one forecast hour via Herbie.

    Returns 2D array (lat, lon) or None on failure.
    """
    from herbie import Herbie

    try:
        H = Herbie(
            f"{date_str} {run}:00",
            model="gefs",
            product="atmos",
            member=member,
            fxx=fxx,
        )
        ds = H.xarray(search_pattern, verbose=False)

        if isinstance(ds, list):
            ds = ds[0]

        ds_sub = subset_icpac(ds)
        var_name = list(ds_sub.data_vars)[0]
        data = ds_sub[var_name].values

        # Remove any extra dims (step, etc.) — we want (lat, lon)
        while data.ndim > 2:
            data = data[0]

        ds.close()
        return data.astype(np.float32)

    except Exception:
        return None


# ── Core fetch routine ────────────────────────────────────────────────

def fetch_all_variables_for_date(
    date_str: str,
    run: str = "00",
    forecast_hours: List[int] = FORECAST_HOURS,
    max_members: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Path]:
    """
    Download all 8 cGAN input variables for one date via Herbie.

    For each variable, fetches all ensemble members at all needed forecast hours,
    and saves a NetCDF file with dims (time, member, step, latitude, longitude).

    Returns dict of variable name -> output file path.
    """
    from herbie import Herbie

    log = logger.info if logger else print
    log_warn = logger.warning if logger else print

    log(f"Fetching all cGAN variables for {date_str} {run}Z")
    log(f"  Forecast hours: {forecast_hours}")

    # Determine ensemble members available
    # GEFS has members 1-30 (gep01..gep30)
    n_total_members = 30
    if max_members:
        n_total_members = min(max_members, n_total_members)
    log(f"  Ensemble members: {n_total_members}")

    # Get ICPAC grid coordinates from a test fetch
    log("  Discovering ICPAC grid coordinates...")
    test_H = Herbie(f"{date_str} {run}:00", model="gefs", product="atmos",
                    member=1, fxx=forecast_hours[0])
    test_ds = test_H.xarray(CGAN_VARIABLES["tmp"]["search"], verbose=False)
    if isinstance(test_ds, list):
        test_ds = test_ds[0]
    test_sub = subset_icpac(test_ds)
    icpac_lats = test_sub.latitude.values
    icpac_lons = test_sub.longitude.values
    test_ds.close()
    log(f"  ICPAC grid: {len(icpac_lats)} lat x {len(icpac_lons)} lon")

    n_steps = len(forecast_hours)
    init_time = datetime.strptime(date_str, "%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for var_name, var_config in CGAN_VARIABLES.items():
        log(f"\n  Variable: {var_name}")
        search = var_config["search"]

        # Allocate array: (member, step, lat, lon)
        data = np.full(
            (n_total_members, n_steps, len(icpac_lats), len(icpac_lons)),
            np.nan, dtype=np.float32
        )

        success_count = 0
        total_fetches = n_total_members * n_steps

        for m_idx in range(n_total_members):
            member = m_idx + 1
            for s_idx, fxx in enumerate(forecast_hours):
                arr = fetch_single_field_herbie(
                    date_str, run, fxx, member, search
                )
                if arr is not None:
                    # Handle shape mismatch gracefully
                    if arr.shape == (len(icpac_lats), len(icpac_lons)):
                        data[m_idx, s_idx, :, :] = arr
                        success_count += 1
                    else:
                        log_warn(f"    Shape mismatch for {var_name} m{member} T+{fxx}h: "
                                 f"{arr.shape} vs expected ({len(icpac_lats)}, {len(icpac_lons)})")
                else:
                    log_warn(f"    Failed: {var_name} member {member} T+{fxx}h")

        log(f"    Fetched: {success_count}/{total_fetches}")

        # Create xarray Dataset matching GIK pipeline Stage 4 layout
        step_coord = np.array(forecast_hours, dtype=np.int64)
        member_coord = np.arange(1, n_total_members + 1)

        ds_out = xr.Dataset(
            {
                var_name: xr.DataArray(
                    data=data[np.newaxis, ...].astype(np.float32),  # (1, member, step, lat, lon)
                    dims=["time", "member", "step", "latitude", "longitude"],
                    attrs={"long_name": var_name},
                )
            },
            coords={
                "time": [np.datetime64(init_time)],
                "member": member_coord,
                "step": step_coord,
                "latitude": icpac_lats,
                "longitude": icpac_lons,
            },
        )

        out_file = output_dir / f"{var_name}_{date_str}_{run}z.nc"
        encoding = {var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        ds_out.to_netcdf(out_file, encoding=encoding)
        ds_out.close()

        size_kb = out_file.stat().st_size / 1024
        log(f"    Saved: {out_file.name} ({size_kb:.1f} KB)")
        saved_files[var_name] = out_file

    return saved_files


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch all 8 cGAN input variables from GEFS via Herbie"
    )
    parser.add_argument("--date", type=str, required=True,
                        help="Target date (YYYYMMDD)")
    parser.add_argument("--run", type=str, default="00",
                        help="Model run hour (default: 00)")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap ensemble members (default: 30)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_file = output_dir / "herbie_cgan_log.txt"

    logger = setup_logging(output_dir, log_file)
    logger.info("=" * 70)
    logger.info("Herbie GEFS Multi-Variable Fetch for cGAN")
    logger.info("=" * 70)
    logger.info(f"Date: {args.date}  |  Run: {args.run}Z")
    logger.info(f"Forecast hours: {FORECAST_HOURS}")
    logger.info(f"Max members: {args.max_members or 'all (30)'}")
    logger.info(f"Variables: {list(CGAN_VARIABLES.keys())}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    t0 = time.time()
    saved = fetch_all_variables_for_date(
        args.date, run=args.run, max_members=args.max_members,
        logger=logger, output_dir=output_dir,
    )
    elapsed = time.time() - t0

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Variables saved: {len(saved)}/{len(CGAN_VARIABLES)}")
    for var, path in saved.items():
        logger.info(f"  {var}: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
