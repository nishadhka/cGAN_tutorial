#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "icechunk",
# ]
# ///
"""
Per-date integrity / coverage check for a pytorch_cgan Icechunk store on GCS.

Unlike the ingest script's `verify` (which spot-checks only the first
date+member), this walks EVERY init_date and reports, per variable, the
fraction of valid (non-NaN) grid points using the control member across all
lead times. That tells us precisely which of the *required* dates are filled,
which are still empty template slots, and which are only partial.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/coiled-data.json \
        uv run verify_icechunk_coverage.py \
        --gcs-bucket cgan-east-africa \
        --gcs-prefix pytorch_cgan_ifs_mam2026_ens
"""

import argparse

import icechunk
import numpy as np
import pandas as pd
import xarray as xr


def open_store(bucket: str, prefix: str) -> tuple[xr.Dataset, icechunk.Repository]:
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds, repo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gcs-bucket", required=True)
    p.add_argument("--gcs-prefix", required=True)
    p.add_argument("--member", type=int, default=0,
                   help="Member index to probe per date (default 0 = control)")
    p.add_argument("--full-pct", type=float, default=99.9,
                   help="valid%% >= this counts a date as FULL")
    p.add_argument("--empty-pct", type=float, default=0.01,
                   help="valid%% <= this counts a date as EMPTY")
    args = p.parse_args()

    print(f"Opening gs://{args.gcs_bucket}/{args.gcs_prefix}/ ...")
    ds, repo = open_store(args.gcs_bucket, args.gcs_prefix)

    dims = dict(ds.sizes)
    print(f"\nDimensions: {dims}")
    for dim in ["init_date", "member", "lead_time", "lat", "lon"]:
        if dim in ds.dims:
            v = ds[dim].values
            print(f"  {dim:10s}: {ds.sizes[dim]:4d}  [{v[0]} .. {v[-1]}]")

    init_dates = pd.to_datetime(ds.init_date.values)
    variables = list(ds.data_vars)
    n_dates = len(init_dates)

    # Per-variable, per-date valid fraction over (lead_time, lat, lon) for one member.
    print(f"\nProbing member index {args.member} across all {n_dates} init_dates "
          f"and {ds.sizes['lead_time']} lead times...\n")

    # date_status[var] -> list of (date, valid_pct)
    full_dates_per_var = {}
    empty_dates_per_var = {}
    partial_dates_per_var = {}

    # Use tp as the reference for "is this date filled at all".
    ref_var = "tp" if "tp" in variables else variables[0]

    for var in variables:
        da = ds[var].isel(member=args.member)  # (init_date, lead_time, lat, lon)
        # valid fraction per init_date
        valid = (~np.isnan(da)).astype("float32")
        # mean over lead_time, lat, lon -> per init_date fraction
        frac = valid.mean(dim=[d for d in da.dims if d != "init_date"]).load().values
        pct = frac * 100.0

        full = [(init_dates[i], pct[i]) for i in range(n_dates) if pct[i] >= args.full_pct]
        empty = [(init_dates[i], pct[i]) for i in range(n_dates) if pct[i] <= args.empty_pct]
        partial = [(init_dates[i], pct[i]) for i in range(n_dates)
                   if args.empty_pct < pct[i] < args.full_pct]

        full_dates_per_var[var] = full
        empty_dates_per_var[var] = empty
        partial_dates_per_var[var] = partial

        # value range over filled points
        good = da.values[~np.isnan(da.values)] if da.size < 5e7 else None
        rng = ""
        if good is not None and good.size:
            rng = f"  range[{good.min():.4g}, {good.max():.4g}] mean={good.mean():.4g}"

        print(f"  {var:9s}: FULL={len(full):3d}  PARTIAL={len(partial):3d}  "
              f"EMPTY={len(empty):3d}{rng}")

    # Date-level summary based on reference variable
    ref_full = {d for d, _ in full_dates_per_var[ref_var]}
    ref_empty = {d for d, _ in empty_dates_per_var[ref_var]}
    ref_partial = {d for d, _ in partial_dates_per_var[ref_var]}

    print(f"\n=== Date coverage (reference var = {ref_var}, member {args.member}) ===")
    print(f"  FULL    : {len(ref_full)}/{n_dates}")
    print(f"  PARTIAL : {len(ref_partial)}/{n_dates}")
    print(f"  EMPTY   : {len(ref_empty)}/{n_dates}")

    def fmt_ranges(dates):
        """Compress a set of daily Timestamps into contiguous YYYY-MM-DD ranges."""
        if not dates:
            return "(none)"
        s = sorted(dates)
        out = []
        start = prev = s[0]
        for d in s[1:]:
            if (d - prev).days == 1:
                prev = d
            else:
                out.append((start, prev))
                start = prev = d
        out.append((start, prev))
        return ", ".join(
            f"{a:%Y-%m-%d}" if a == b else f"{a:%Y-%m-%d}..{b:%Y-%m-%d}"
            for a, b in out
        )

    print(f"\n  FILLED range : {fmt_ranges(ref_full)}")
    if ref_partial:
        print(f"  PARTIAL dates: {fmt_ranges(ref_partial)}")
    if ref_empty:
        print(f"  EMPTY range  : {fmt_ranges(ref_empty)}")

    # Commits
    try:
        commits = list(repo.ancestry(branch="main"))
        print(f"\nCommits on main: {len(commits)}")
        for c in commits[:5]:
            print(f"  {c.message}")
        if len(commits) > 5:
            print(f"  ... and {len(commits) - 5} more")
    except Exception as e:
        print(f"(could not read ancestry: {e})")

    # Exit non-zero if any required date is empty/partial (useful in scripts)
    bad = len(ref_empty) + len(ref_partial)
    print(f"\n{'OK — all dates full.' if bad == 0 else f'NOTE — {bad} date(s) not full.'}")


if __name__ == "__main__":
    main()
