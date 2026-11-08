## About this run

The same end-to-end run as [`2026-10-30-local-9day-endtoend.md`](2026-10-30-local-9day-endtoend.md)
(20261021–29), continued for the next cycles: 00z from 20261030 to 20261108,
run one after the other on 2026-11-08 as a daily job would run them. The last
cycle, 20261108, was taken as soon as ECMWF had published it: the driver
waited for the four 360 h `.index` files, checking every 5 min (see Notes: the `.grib2` files can come later).

The summary and the per-cycle sections are generated from the run's own
log, timing file and stores by [`make_local_run_report.py`](make_local_run_report.py)
(none of their numbers is typed in); the sections from here to "Notes" come from
[`2026-11-08-local-10day-intro.md`](2026-11-08-local-10day-intro.md).

## What was run

| | |
|---|---|
| Code | `cGAN_tutorial/ifs-0p1-gik`: `run_local.sh`, step 1–7 as of **`222084c`** (no change to the scripts since `185b789`, except `grep --line-buffered` in `run_local.sh`) |
| Driver | `/srv/state/ifs0p1/run10.sh`: `run_local.sh <YYYYMMDD>` once per date, with `BUILD_VD=1 PRODUCTS="jurre-6h mvua-7d"`, the cycles in sequence, each waited for until ECMWF has published its 360 h `.index` files (below) |
| Machine | `jupyterhub-gateway2`, AMD EPYC 7713, 8 vCPU, 31 GB RAM; stores and outputs on a local 100 GB disk (`/srv/state`) |
| Cluster | **none**: step4 runs its block tasks on 12 threads inside one process (`--no-cluster --threads 12`, 4 GETs per block); no Dask, Frisky or EWC workers |
| Uploads | none; nothing is written outside this machine |
| Python | 3.12.15: icechunk 2.1.1, zarr 3.2.1, xarray 2026.7.0, gribberish 1.6.0, numpy 2.5.1, imagecodecs 2026.10.10 (libaec), netCDF4 1.7.4 |
| Data | ECMWF open data, IFS ENS 0.1°, `s3://ecmwf-forecasts` with the byte-identical `gs://ecmwf-open-data` copy as fallback; read anonymously |

```bash
export PY=/srv/state/ifs0p1/venv/bin/python TFPY=/srv/state/ifs0p1/venv-infer/bin/python
export IFS0P1_LOCAL_ROOT=/srv/state/ifs0p1/icechunk OUTDIR=/srv/state/ifs0p1/nc9 GAN_OUTDIR=/srv/state/ifs0p1/gan9
export TIMING=/srv/state/ifs0p1/timing10.tsv BUILD_VD=1 PRODUCTS="jurre-6h mvua-7d"
for d in 20261030 20261031 20261101 20261102 20261103 20261104 20261105 20261106 20261107 20261108; do
  # (run10.sh first waits for $d's four 360 h .index files on GCS; 20261108 was rerun once the .grib2 were there too)
  bash run_local.sh $d
done
```

## The stages, per cycle

| Stage | Script | What it does |
|---|---|---|
| step1 | `step1_build_virtual.py --verify 8` | reads the cycle's 340 `.index` files from GCS and adds ~742k byte-range references to a **local virtual Icechunk store** (`ecmwf-ifs-0p1-icechunk-vd`); 8 chunks read back and compared with a direct decode |
| step4 | `step4_realize.py --product jurre-6h mvua-7d --no-cluster --threads 12` | **local realized Icechunk store**, one per cycle: 14 fields × the 30 lead times 0..174 h the two products need × 51 members over 25°N–15°S, 19–55°E (401 × 361). Each message is a prefix read (message start → the box's last row, decoded with libaec), ~67% of its bytes |
| step5 | `step5_check.py --samples 8` | every chunk written; 8 random boxes bit-exact against whole messages decoded through the virtual store by gribberish; fields finite and plausible |
| step6 | `step6_sewaa_netcdf.py --product jurre-6h mvua-7d` | the two IFS NetCDFs the SEWAA scripts read: ensemble mean and sd of 51 members (sd ddof 0) on the 384 × 352 cGAN frame; `cgan-ifs-6h-ens/` (+30..54 h) and `cgan-ifs-7d-ens/` (+0..174 h) |
| step7 | `step7_infer.py --dry-run`, per product | builds every network input (jurre-6h: 4 windows × 56 channels; mvua-7d: 7 days × 28 channels) and checks it is finite. **The model is not run** |

## Weights: not used in this run

No trained weights were loaded; step7 ran as `--dry-run`, so the times stop
just before the generator. The weights were located during the run:

- **6 h (Jurre-Brishti):** `Fenwick-Cooper/SEWAA-forecasts` @ `2fe8a40`,
  `6h_accumulations/cGAN/ICPAC-big-ensmeansd/models/gen_weights-0316800.h5`. This is
  the operational checkpoint 316800 (`sewaa-forecasts-package`
  `configs/jurre-brishti/forecast-ens.yaml`), same 14 fields and 56-channel
  input; it loads into `vendor/ensemble_cgan` with 56 input channels. Its
  `FCSTNorm2018.pkl` and `cGAN_data/elev.nc`, `lsm.nc` have the same md5 as the
  constants `run_local.sh` uses.
- **7 days:** the operational Mvua-Kubwa checkpoint 300800 (14 fields, 28
  channels) is not public. SEWAA-forecasts has newer per-lead-day 24 h models
  (`24h_accumulations/cGAN/logs_1` @ 203776 day 0, `logs_5` @ 163840 day 1,
  `logs_17` @ 115200 days 2–6) with **13 fields (no cape), 26 channels** and
  their own normalisation. They load only with 26 input channels, so they need
  a separate step7 mode; the `cgan-ifs-7d-ens` NetCDF already holds what they read.

## Notes

- **The network sets the pace.** step4 is mostly network time: the decode is
  ~38 ms per message throughout. 20261030–1102 ran with a mean GET of
  ~0.68 s at ~57 msg/s (375–378 s per cycle). From 20261103 the GETs slowed to
  0.87–1.17 s and step4 to 468–688 s. The reader falls back to the
  byte-identical GCS copy on any S3 error. Over the nine published cycles 43%
  of messages came from GCS, against 85–90% on 2026-10-30 (S3 throttled less
  this time).
- **On disk:** realized stores ~2.4–2.6 GB per cycle; the local virtual store
  holds all 19 cycles since 20261021 in 302 MB; the SEWAA NetCDFs are ~45 MB
  (6 h) and ~280 MB (7 d) per cycle.
- **20261108** was not on ECMWF's buckets when the run started (07:03 UTC):
  the 0.25° run was complete, but no 0.1° file yet. The 0.1° stream is the
  later of the two (20261107's last 0.1° `.index` is stamped 13:25 UTC), so
  the driver waited for it.
- **20261108: the first attempt failed and the rerun passed.**
  - The driver started at 13:44 UTC, as soon as the four 360 h `.index` files
    were on GCS. GCS had published them **before the `.grib2` files**.
  - step1 committed the cycle to the virtual store (742,478 refs, 44 s). Its
    `--verify` read-back then failed on 1 of 8 chunks (`r600 m50 +360 h`,
    "error fetching virtual reference"), so `run_local.sh` stopped.
  - Once all four 360 h `.grib2` were on GCS (13:48 UTC) the cycle was rerun.
    step1 found it already in the store (1 s), and step5 checked 8 random
    messages bit-exact against it.
  - The section below is the rerun. `step1_append_when_published.sh` now waits
    for the `.grib2` files too.
- The per-stage seconds are also in [`2026-11-08-local-10day-timing.tsv`](2026-11-08-local-10day-timing.tsv).
