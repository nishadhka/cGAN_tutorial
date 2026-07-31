# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=2.1", "zarr>=3.2", "xarray>=2025.1", "numpy",
#   "gribberish>=1.4", "s3fs", "netcdf4", "dask[distributed]",
# ]
# ///
"""Materialise an East Africa subset of the published ECMWF IFS ensemble
Icechunk store (source.coop) onto a Dask cluster, one NetCDF per
(channel, year), ensemble-reduced to mean + sd.

Companion document:
  docs/ecmwf_icechunk_dask_variable_extraction.md

Design constraints that drive the task decomposition (all verified 2026-07-30):

  * Resolving ANY chunk of an array loads that array's ENTIRE icechunk
    manifest. For a 49r1 pressure-level array that is 794 dates x 51 members
    x 85 steps x 13 levels = 44.7 M refs ~= 9 GB RSS. So a worker can hold
    ~one pl manifest at a time, and a task must be BIG enough to amortise
    that load. One task = (one store variable, all its levels, one year).
  * Subsetting to the East Africa box does NOT reduce bytes read from S3: a
    virtual chunk is one whole global GRIB message. Only the write side
    shrinks. Run the cluster in eu-central-1, where `ecmwf-forecasts` lives,
    so the reads are same-region and free.
  * The `49r1/00z` group is a UNION of two schema sub-eras. `cape` is finite
    only before 2025-01-14 and `mucape` only after; `tcw`/`tprate`/`ptype`
    only after; `tcc`/`sf` never. Arrays exist either way and read back
    all-NaN outside their window -- so a silent all-NaN channel is the
    failure mode this script guards against with --check.

Usage
-----
  # 1. cheap availability check, no cluster, one date per year
  uv run materialize_ea_icechunk_dask.py check --years 2024 2025 2026

  # 2. one-day dry run to calibrate throughput before committing
  uv run materialize_ea_icechunk_dask.py run --years 2024 --months 3 \
      --dates 2024-03-01 --cluster local --out ./ea_out

  # 3. full extraction on Coiled, eu-central-1
  uv run materialize_ea_icechunk_dask.py run --years 2024 2025 2026 \
      --months 3 4 5 --cluster coiled --workers 18 --out s3://.../ea_ifs
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────
ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
PREFIX = "forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd"
CONTAINER = "s3://ecmwf-forecasts/"          # virtual chunks, public AWS, anon
AWS_REGION_OF_GRIB = "eu-central-1"          # where `ecmwf-forecasts` lives

# Era boundaries, from the store's own time axes (verified).
ERA_WINDOWS = {
    "0p4":  ("2023-01-18", "2024-02-28"),
    "49r1": ("2024-02-29", "2026-05-12"),
    "50r1": ("2026-05-13", None),
}
# Sub-era break inside the 49r1 group: 9-level -> 13-level.
BREAK_13LEVEL = np.datetime64("2025-01-14")

# ─────────────────────────────────────────────────────────────────────────────
# Domain — a 0.25 deg SUPERSET box that strictly contains BOTH consumers, with
# a 2-cell halo so 0.25 -> 0.1 deg interpolation never extrapolates:
#
#   TF pipeline (training today) : 0.1 deg, 384 x 352,
#                                  -13.65..24.65 N, 19.15..54.25 E
#                                  (IFS_training/<yr>/<field>.nc)
#   PyTorch EP ingest            : 0.25 deg, 161 x 133, -15..25 N, 20..53 E
#                                  (ingest_ecmwf_pytorch_cgan_variables.py)
#
# Note the EP box does NOT cover the TF frame (53 E < 54.25 E). Extract the
# superset once; crop per consumer downstream.  163 x 147 @ 0.25 deg.
# ─────────────────────────────────────────────────────────────────────────────
LAT_MAX, LAT_MIN = 25.25, -15.25   # store latitude DESCENDS 90 -> -90
LON_MIN, LON_MAX = 18.5, 55.0      # store longitude is 0..359.75, no wrap here

# Optional wide box for the basin-scale channels (SST proxy, OLR) only.
BASIN_LAT_MAX, BASIN_LAT_MIN = 25.0, -25.0
BASIN_LON_MIN, BASIN_LON_MAX = 30.0, 120.0
BASIN_COARSEN = 8                  # 0.25 deg -> 2 deg

# Lead times. 3-hourly out to 144 h in the store; 24..54 covers the single
# 24 h accumulation window (variant A) and the 30/36/42/48 h set (variant B),
# including the step-before needed to difference accumulated fields.
STEPS_H = list(range(24, 57, 3))

# Ensemble. 0 = control. Reducing to mean+sd is what the cGAN consumes
# (2 channels per field), so the write side is tiny; the READ side still
# scales with len(MEMBERS), which is the main cost knob in this script.
MEMBERS_FULL = list(range(51))
MEMBERS_CHEAP = [0] + list(range(1, 51, 5))   # control + 10 perturbed

# ─────────────────────────────────────────────────────────────────────────────
# Channel specification
#   (store_var, level_hPa|None, out_name, eras_ok)
# `eras_ok` is a tuple of era keys where the array is actually POPULATED --
# not merely declared. Verified by reading MAM 2024 / MAM 2025 samples.
# ─────────────────────────────────────────────────────────────────────────────
ALL_ERAS = ("0p4", "49r1", "50r1")
E_2425 = ("49r1", "50r1")

SURFACE_SPEC = [
    # store   lvl   out          eras
    ("tp",    None, "tp",        ALL_ERAS),
    ("tcwv",  None, "pw",        ALL_ERAS),
    ("sp",    None, "sp",        ALL_ERAS),
    ("msl",   None, "msl",       ALL_ERAS),
    ("t2m",   None, "t2m",       ALL_ERAS),
    ("skt",   None, "skt",       ALL_ERAS),   # -> SST proxy over water
    ("ssr",   None, "ssr",       E_2425),
    ("ttr",   None, "ttr",       E_2425),     # -> OLR = -ttr / dt
    ("tcw",   None, "tcw",       E_2425),     # 49r1: only from 2025-01-14
    ("cape",  None, "cape",      ("49r1",)),  # 49r1: only BEFORE 2025-01-14
    ("mucape", None, "mucape",   E_2425),     # 49r1: only from 2025-01-14
]

PRESSURE_SPEC = [
    ("u",  925, "u925", ALL_ERAS),
    ("v",  925, "v925", ALL_ERAS),
    ("u",  850, "u850", ALL_ERAS),
    ("v",  850, "v850", ALL_ERAS),
    ("u",  700, "u700", ALL_ERAS),
    ("v",  700, "v700", ALL_ERAS),
    ("u",  500, "u500", ALL_ERAS),
    ("v",  500, "v500", ALL_ERAS),
    ("u",  200, "u200", ALL_ERAS),
    ("v",  200, "v200", ALL_ERAS),
    ("gh", 500, "gh500", ALL_ERAS),
    ("w",  925, "w925", E_2425),              # no `w` at all in 0p4
    ("w",  850, "w850", E_2425),
    ("w",  700, "w700", E_2425),
    ("w",  500, "w500", E_2425),
    ("r",  850, "r850", ALL_ERAS),            # the "RH 800" ask -> 800 hPa
    ("r",  700, "r700", ALL_ERAS),            #   does not exist; 850 is it
    ("t",  850, "t850", ALL_ERAS),            # t850/700/500 + r850/700 feed
    ("t",  700, "t700", ALL_ERAS),            #   the K-index and theta-w
    ("t",  500, "t500", ALL_ERAS),
]

STATIC_SPEC = [("lsm", None, "lsm", ALL_ERAS)]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ea-materialize")


# ─────────────────────────────────────────────────────────────────────────────
# Store access (called on the worker — an icechunk session is not shipped)
# ─────────────────────────────────────────────────────────────────────────────
def open_era(era: str):
    """Open one era group anonymously. ~5-60 s; pay it once per task."""
    import icechunk
    import gribberish.zarr  # noqa: F401  registers the Zarr v3 codec

    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT, region="us-east-1",
        anonymous=True,          # public read of the store metadata
        from_env=False,          # ignore stray AWS_* env vars
        force_path_style=True,   # source.coop needs path-style addressing
    )
    auth = icechunk.containers_credentials(
        {CONTAINER: icechunk.s3_anonymous_credentials()})
    cfg = icechunk.RepositoryConfig.default()
    # Eager manifest preload draws source.coop's sporadic 500s; go lazy.
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0,
                                               max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg,
                                    authorize_virtual_chunk_access=auth)
    return xr.open_zarr(repo.readonly_session("main").store,
                        group=f"{era}/00z", consolidated=False,
                        zarr_format=3, decode_timedelta=True)


def era_for(date: np.datetime64) -> str:
    for era, (lo, hi) in ERA_WINDOWS.items():
        if date >= np.datetime64(lo) and (hi is None or date <= np.datetime64(hi)):
            return era
    raise ValueError(f"{date} outside every era window")


def season_dates(year: int, months: list[int]) -> list[np.datetime64]:
    out = []
    for m in months:
        d = np.datetime64(f"{year}-{m:02d}-01")
        while d.astype("datetime64[M]").astype(int) % 12 + 1 == m:
            out.append(d)
            d = d + np.timedelta64(1, "D")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The unit of work: one store variable, all its levels, one year
# ─────────────────────────────────────────────────────────────────────────────
def extract_task(task: dict) -> dict:
    """Read (var, levels, dates, steps, members) for one era-year, reduce over
    `number` to mean+sd, and write one NetCDF per output channel."""
    t0 = time.time()
    era, var = task["era"], task["var"]
    levels = task["levels"]          # list[(level|None, out_name)]
    dates = [np.datetime64(d) for d in task["dates"]]
    steps = [np.timedelta64(h, "h") for h in task["steps"]]
    members = task["members"]
    outdir = Path(task["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    ds = open_era(era)
    if var not in ds:
        return {"task": f"{era}/{var}", "status": "absent", "seconds": 0.0}

    box = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    if task.get("basin"):
        box = dict(latitude=slice(BASIN_LAT_MAX, BASIN_LAT_MIN),
                   longitude=slice(BASIN_LON_MIN, BASIN_LON_MAX))

    written, skipped = [], []
    for level, out_name in levels:
        da = ds[var].sel(number=members, step=steps, **box)
        if level is not None:
            da = da.sel(isobaricInhPa=level)

        chunks = []
        for d in dates:
            try:
                v = da.sel(time=d).compute()
            except KeyError:
                continue                      # date not in this era's axis
            chunks.append(v.expand_dims(time=[d]))
        if not chunks:
            skipped.append(out_name)
            continue

        cube = xr.concat(chunks, dim="time")          # (time, number, step, y, x)
        if task.get("basin"):
            cube = cube.coarsen(latitude=BASIN_COARSEN, longitude=BASIN_COARSEN,
                                boundary="trim").mean()

        stacked = xr.concat(
            [cube.mean("number"), cube.std("number")],
            dim=xr.DataArray(["mean", "sd"], dims="stat", name="stat"),
        ).transpose("time", "step", "stat", "latitude", "longitude")

        finite = float(np.isfinite(stacked.sel(stat="mean").values).mean())
        if finite == 0.0:
            # The all-NaN trap: the array exists for this era but is not
            # populated in this window (e.g. `cape` after 2025-01-14).
            skipped.append(f"{out_name}(all-NaN)")
            continue

        path = outdir / f"{out_name}_{task['year']}.nc"
        stacked.rename(out_name).to_netcdf(
            path, encoding={out_name: {"zlib": True, "complevel": 4,
                                       "dtype": "float32"}})
        written.append(f"{out_name}({finite:.2f})")

    return {"task": f"{era}/{var}/{task['year']}", "status": "ok",
            "written": written, "skipped": skipped,
            "seconds": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Derived channels — no extra S3 reads, computed from what was written
# ─────────────────────────────────────────────────────────────────────────────
def _dewpoint_from_rh(t_K, rh_pct):
    """Magnus/Alduchov-Eskridge dewpoint. t in K, rh in %, returns K."""
    a, b = 17.625, 243.04
    tc = t_K - 273.15
    rh = np.clip(rh_pct, 1.0, 100.0) / 100.0
    g = np.log(rh) + a * tc / (b + tc)
    return b * g / (a - g) + 273.15


def derive_channels(outdir: Path, year: int, step_hours: float) -> list[str]:
    """K-index, wet-bulb potential temperature at 850 hPa, OLR, SST."""
    made = []

    def _load(name):
        p = outdir / f"{name}_{year}.nc"
        return xr.open_dataarray(p) if p.exists() else None

    t850, t700, t500 = _load("t850"), _load("t700"), _load("t500")
    r850, r700 = _load("r850"), _load("r700")

    if all(x is not None for x in (t850, t700, t500, r850, r700)):
        td850 = xr.apply_ufunc(_dewpoint_from_rh, t850, r850)
        td700 = xr.apply_ufunc(_dewpoint_from_rh, t700, r700)
        # K = (T850 - T500) + Td850 - (T700 - Td700), in degC
        kidx = (t850 - t500) + (td850 - 273.15) - (t700 - td700)
        kidx.rename("kindex").to_netcdf(outdir / f"kindex_{year}.nc")
        made.append("kindex")

        # theta-e via Bolton (1980) eq. 43 at 850 hPa, then Davies-Jones (2008)
        # eq. 3.8 inversion to theta-w. Adequate for a normalised NN channel.
        p = 850.0
        e = 6.112 * np.exp(17.67 * (td850 - 273.15) / (td850 - 29.65))
        w = 0.622 * e / (p - e)                       # mixing ratio, kg/kg
        tl = 56.0 + 1.0 / (1.0 / (td850 - 56.0) + np.log(t850 / td850) / 800.0)
        the = (t850 * (1000.0 / p) ** (0.2854 * (1 - 0.28 * w))
               * np.exp((3036.0 / tl - 1.78) * w * (1 + 0.448 * w)))
        x = the / 273.15
        thw = the - np.exp((3.504 / x - 0.6992) / (1 + 0.1745 * x ** 4)) * 100.0
        thw = xr.where(the > 173.15, thw, the) - 273.15
        thw.rename("thetaw850").to_netcdf(outdir / f"thetaw850_{year}.nc")
        made.append("thetaw850")

    ttr = _load("ttr")
    if ttr is not None:
        # ttr is accumulated top net thermal radiation, J m-2 since step 0.
        # Verified: -2.32e7 J/m2 at +24 h => -268 W/m2, i.e. OLR ~ 268 W/m2.
        olr = -ttr / (step_hours * 3600.0)
        olr.rename("olr").to_netcdf(outdir / f"olr_{year}.nc")
        made.append("olr")

    skt, lsm = _load("skt"), _load("lsm")
    if skt is not None and lsm is not None:
        sst = skt.where(lsm.squeeze(drop=True) < 0.5)
        sst.rename("sst").to_netcdf(outdir / f"sst_{year}.nc")
        made.append("sst")

    return made


# ─────────────────────────────────────────────────────────────────────────────
# Availability check — cheap, one date per year, no cluster
# ─────────────────────────────────────────────────────────────────────────────
def cmd_check(args):
    box = dict(latitude=slice(2.0, 0.0), longitude=slice(36.0, 38.0))
    step = np.timedelta64(24, "h")
    for year in args.years:
        probe = np.datetime64(f"{year}-03-15")
        era = era_for(probe)
        ds = open_era(era)
        print(f"\n=== {year}  era={era}  "
              f"levels={list(ds.isobaricInhPa.values.astype(int))}")
        for store_var, level, out_name, eras_ok in (
                SURFACE_SPEC + PRESSURE_SPEC + STATIC_SPEC):
            if era not in eras_ok:
                print(f"  {out_name:10s} -- not published in {era}")
                continue
            try:
                da = ds[store_var].sel(time=probe, number=0, step=step, **box)
                if level is not None:
                    da = da.sel(isobaricInhPa=level)
                f = float(np.isfinite(da.compute().values).mean())
                print(f"  {out_name:10s} {'OK ' if f > 0 else 'ALL-NaN'} finite={f:.2f}")
            except Exception as exc:                  # noqa: BLE001
                print(f"  {out_name:10s} ERROR {type(exc).__name__}: {str(exc)[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def build_tasks(args) -> list[dict]:
    """One task per (era, store variable, year). Big enough to amortise the
    per-array manifest load, which is the binding constraint."""
    spec = SURFACE_SPEC + PRESSURE_SPEC
    tasks = []
    for year in args.years:
        dates = ([np.datetime64(d) for d in args.dates] if args.dates
                 else season_dates(year, args.months))
        by_era: dict[str, list] = {}
        for d in dates:
            by_era.setdefault(era_for(d), []).append(d)

        for era, era_dates in by_era.items():
            per_var: dict[str, list] = {}
            for store_var, level, out_name, eras_ok in spec:
                if era not in eras_ok:
                    continue
                per_var.setdefault(store_var, []).append((level, out_name))
            for store_var, levels in per_var.items():
                tasks.append(dict(
                    era=era, var=store_var, levels=levels, year=year,
                    dates=[str(d) for d in era_dates], steps=args.steps,
                    members=args.members, outdir=str(Path(args.out) / str(year)),
                    basin=False,
                ))
            # Basin-scale SST/OLR pair on the wide, coarsened box.
            if args.basin:
                for store_var, out_name in (("skt", "skt_basin"),
                                            ("ttr", "ttr_basin")):
                    if era in dict((s[0], s[3]) for s in SURFACE_SPEC).get(store_var, ()):
                        tasks.append(dict(
                            era=era, var=store_var, levels=[(None, out_name)],
                            year=year, dates=[str(d) for d in era_dates],
                            steps=args.steps, members=[0],
                            outdir=str(Path(args.out) / str(year)), basin=True,
                        ))
    return tasks


def cmd_run(args):
    from dask.distributed import Client

    tasks = build_tasks(args)
    log.info("%d tasks, %d dates/yr, %d steps, %d members",
             len(tasks), len(args.months) * 31 if not args.dates else len(args.dates),
             len(args.steps), len(args.members))

    if args.cluster == "coiled":
        import coiled
        cluster = coiled.Cluster(
            name=f"ea-ifs-{int(time.time()) % 100000}",
            n_workers=args.workers,
            worker_vm_types=["r7i.xlarge"],      # 4 vCPU / 32 GiB -- one 9 GB
            worker_options={"nthreads": 2},      #   pl manifest fits with room
            region=AWS_REGION_OF_GRIB,           # same region as the GRIB bytes
            workspace="e4drr", package_sync=True, idle_timeout="20 minutes",
        )
        client = Client(cluster)
    else:
        client = Client(n_workers=min(args.workers, 4), threads_per_worker=2,
                        memory_limit="12GiB")
    log.info("dashboard: %s", client.dashboard_link)

    t0 = time.time()
    for res in client.gather(client.map(extract_task, tasks)):
        log.info("%(task)s %(status)s %(seconds)ss written=%(written)s "
                 "skipped=%(skipped)s", {**{"written": [], "skipped": []}, **res})
    log.info("extraction done in %.1f min", (time.time() - t0) / 60)

    for year in args.years:
        made = derive_channels(Path(args.out) / str(year), year,
                               step_hours=float(args.steps[-1]))
        log.info("%d derived: %s", year, made)
    client.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("check", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
        if name == "run":
            sp.add_argument("--months", type=int, nargs="+", default=[3, 4, 5])
            sp.add_argument("--dates", nargs="*", default=None,
                            help="explicit YYYY-MM-DD list, overrides --months")
            sp.add_argument("--steps", type=int, nargs="+", default=STEPS_H)
            sp.add_argument("--members", type=int, nargs="+", default=MEMBERS_FULL)
            sp.add_argument("--cheap-members", action="store_true",
                            help="control + 10 perturbed instead of all 51")
            sp.add_argument("--basin", action="store_true",
                            help="also extract coarse SST/OLR on 30-120E, 25S-25N")
            sp.add_argument("--cluster", choices=["local", "coiled"], default="local")
            sp.add_argument("--workers", type=int, default=18)
            sp.add_argument("--out", default="./ea_out")

    args = p.parse_args(argv)
    if getattr(args, "cheap_members", False):
        args.members = MEMBERS_CHEAP
    return cmd_check(args) if args.command == "check" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
