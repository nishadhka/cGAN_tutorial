#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "matplotlib",
#     "icechunk",
# ]
# ///
"""
Plot N random (init_date, member, lead_time, variable) tiles from a
pytorch_cgan Icechunk store on GCS, overlaying country boundaries from
the East-Africa GHCF GeoJSON. Used to eyeball that the data we wrote
looks physically reasonable.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/coiled-data.json \
        uv run plot_icechunk_sanity_check.py \
        --gcs-bucket cgan-east-africa \
        --gcs-prefix pytorch_cgan_ifs_mam2026_ens \
        --n 5 --seed 42 \
        --out mam2026_ens_sanity_check.png
"""

import argparse
import json
import os
from pathlib import Path

import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# Per-variable display config: (label, unit, scale factor from store -> display, cmap, vmin/vmax hint)
VAR_DISPLAY = {
    "tp":       ("Total precipitation",     "mm / 3 h",   1000.0,  "Blues",     0,   20),
    "pw":       ("Precipitable water",      "kg m^-2",    1.0,     "YlGnBu",    0,   70),
    "msl":      ("Mean sea-level pressure", "hPa",        0.01,    "RdBu_r",    1000, 1025),
    "sp":       ("Surface pressure",        "hPa",        0.01,    "terrain",   650, 1030),
    "cp_proxy": ("Snowfall (cp proxy)",     "mm / 3 h",   1000.0,  "Greys",     0,   0.5),
    # If pl vars come back later
    "u":        ("U-wind",                  "m s^-1",     1.0,     "RdBu_r",   -30,  30),
    "v":        ("V-wind",                  "m s^-1",     1.0,     "RdBu_r",   -30,  30),
    "gh":       ("Geopotential height",     "gpm",        1.0,     "viridis",  None, None),
}


def open_store(bucket: str, prefix: str) -> xr.Dataset:
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def load_geojson_polygons(path: str):
    """Return a list of (xs, ys) ring arrays for plotting."""
    with open(path) as f:
        gj = json.load(f)
    rings = []
    for feat in gj.get("features", []):
        g = feat.get("geometry") or {}
        t = g.get("type")
        coords = g.get("coordinates")
        if t == "Polygon":
            for ring in coords:
                xs, ys = zip(*ring)
                rings.append((np.array(xs), np.array(ys)))
        elif t == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    xs, ys = zip(*ring)
                    rings.append((np.array(xs), np.array(ys)))
    return rings


def plot_boundaries(ax, rings, **kwargs):
    for xs, ys in rings:
        ax.plot(xs, ys, **kwargs)


def find_filled_init_indices(ds: xr.Dataset) -> np.ndarray:
    """Indices along init_date where at least some non-NaN data exists
    in tp (control member, first lead). Used to skip the unfilled
    template slots in a partially-populated store."""
    tp = ds["tp"].isel(member=0, lead_time=0).load()
    mask = (~np.isnan(tp.values)).any(axis=(1, 2))
    return np.where(mask)[0]


def pick_random_slices(ds: xr.Dataset, n: int, seed: int):
    rng = np.random.default_rng(seed)
    filled = find_filled_init_indices(ds)
    if filled.size == 0:
        raise SystemExit("No filled init dates in store — nothing to plot.")
    variables = [v for v in ds.data_vars if v in VAR_DISPLAY] or list(ds.data_vars)
    picks = []
    for _ in range(n):
        picks.append({
            "init_idx":   int(rng.choice(filled)),
            "member_idx": int(rng.integers(0, ds.sizes["member"])),
            "lead_idx":   int(rng.integers(0, ds.sizes["lead_time"])),
            "var":        str(rng.choice(variables)),
        })
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gcs-bucket", required=True)
    p.add_argument("--gcs-prefix", required=True)
    p.add_argument("--geojson",
                   default="/scratch/notebook/grib-index-kerchunk/gefs/dev-test/ea_ghcf_simple.geojson")
    p.add_argument("--n", type=int, default=5, help="Number of random tiles")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="sanity_check.png")
    args = p.parse_args()

    print(f"Opening gs://{args.gcs_bucket}/{args.gcs_prefix}/ ...")
    ds = open_store(args.gcs_bucket, args.gcs_prefix)
    print(ds)

    rings = load_geojson_polygons(args.geojson)
    print(f"Loaded {len(rings)} country boundary rings from {args.geojson}")

    picks = pick_random_slices(ds, args.n, args.seed)
    print("Picks:")
    for i, pk in enumerate(picks):
        d = str(ds.init_date.values[pk["init_idx"]])[:10]
        m = str(ds.member.values[pk["member_idx"]])
        h = int(ds.lead_time.values[pk["lead_idx"]])
        print(f"  {i+1}. var={pk['var']}  init={d}  member={m}  lead=+{h}h")

    # Layout: 2 columns × ceil(N/2) rows
    ncols = 2 if args.n > 1 else 1
    nrows = (args.n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.8 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()

    for i, pk in enumerate(picks):
        ax = flat_axes[i]
        da = ds[pk["var"]].isel(
            init_date=pk["init_idx"],
            member=pk["member_idx"],
            lead_time=pk["lead_idx"],
        ).load()

        cfg = VAR_DISPLAY.get(pk["var"],
                              (pk["var"], "", 1.0, "viridis", None, None))
        label, unit, scale, cmap, vmin, vmax = cfg
        vals = da.values * scale

        # For tp/cp_proxy, clip the upper bound dynamically if max is high
        if pk["var"] in ("tp", "cp_proxy") and vmax is not None:
            vmax = max(vmax, float(np.nanpercentile(vals, 99.5)))

        im = ax.pcolormesh(
            ds.lon.values, ds.lat.values, vals,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        plot_boundaries(ax, rings, color="black", linewidth=0.45, alpha=0.85)
        ax.set_xlim(float(ds.lon.min()), float(ds.lon.max()))
        ax.set_ylim(float(ds.lat.min()), float(ds.lat.max()))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        init_ts = pd.Timestamp(ds.init_date.values[pk["init_idx"]])
        member = str(ds.member.values[pk["member_idx"]])
        lead_h = int(ds.lead_time.values[pk["lead_idx"]])
        valid_ts = init_ts + pd.Timedelta(hours=lead_h)

        ax.set_title(
            f"{label} ({unit})\n"
            f"init {init_ts:%Y-%m-%d %H:%MZ}  +  lead {lead_h:>2}h  "
            f"→ valid {valid_ts:%Y-%m-%d %H:%MZ}\n"
            f"member: {member}",
            fontsize=10,
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(unit, fontsize=9)

    # Hide leftover panels
    for j in range(len(picks), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(
        f"gs://{args.gcs_bucket}/{args.gcs_prefix}/  —  "
        f"random {args.n} tiles (seed={args.seed})",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
