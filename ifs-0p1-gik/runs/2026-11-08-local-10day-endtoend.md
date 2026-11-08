# Local end-to-end run, daily cycles 20261030–1108 00z, no cluster (in progress: 9 of 10 done; 20261108 waiting for ECMWF)

## Summary, seconds per stage

| Cycle | step1 virtual store | step4 realize | step5 check | step6 NetCDF | step7 jurre-6h | step7 mvua-7d | **total** | step4 msg/s | fetched GB | step5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20261030 00z | 46 | 379 | 22 | 50 | 2 | 7 | **506** (8.4 min) | 56.6 | 30.19 | PASS |
| 20261031 00z | 48 | 375 | 21 | 46 | 2 | 3 | **495** (8.2 min) | 57.1 | 30.04 | PASS |
| 20261101 00z | 46 | 377 | 21 | 51 | 2 | 5 | **502** (8.4 min) | 57.0 | 29.97 | PASS |
| 20261102 00z | 46 | 375 | 20 | 52 | 4 | 8 | **505** (8.4 min) | 57.2 | 30.07 | PASS |
| 20261103 00z | 46 | 469 | 20 | 47 | 2 | 4 | **588** (9.8 min) | 45.8 | 29.78 | PASS |
| 20261104 00z | 48 | 559 | 22 | 50 | 12 | 6 | **697** (11.6 min) | 38.4 | 29.66 | PASS |
| 20261105 00z | 49 | 552 | 22 | 55 | 3 | 10 | **691** (11.5 min) | 38.8 | 29.67 | PASS |
| 20261106 00z | 47 | 689 | 21 | 43 | 2 | 3 | **805** (13.4 min) | 31.1 | 29.69 | PASS |
| 20261107 00z | 40 | 524 | 20 | 49 | 3 | 7 | **643** (10.7 min) | 41.0 | 29.41 | PASS |

9 cycles: total mean **10.1 min**, min 8.2, max 13.4; all 1.51 h.

## About this run

The same end-to-end run as [`2026-10-30-local-9day-endtoend.md`](2026-10-30-local-9day-endtoend.md)
(20261021–29), continued for the next cycles: 00z from 20261030 to 20261108,
run one after the other on 2026-11-08 as a daily job would run them. The last
cycle, 20261108, was taken as soon as ECMWF had published it: the driver
waited for the four 360 h `.index` files, checking every 5 min. The summary and the per-cycle sections are generated from the run's own
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
  # (run10.sh first waits for $d's four 360 h .index files on GCS)
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
- The per-stage seconds are also in [`2026-11-08-local-10day-timing.tsv`](2026-11-08-local-10day-timing.tsv).

## 20261030 00z

Run 2026-11-08T07:03:08Z → 2026-11-08T07:11:34Z, **506 s (8.4 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 46 | 742478 refs from 340 .index files on GCS in 24.3s; virtual snapshot `GVYGR0MCAZ33D4QRR0C0`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 379 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 56.6 msg/s; fetched 30.19 of 44.38 GB whole-message bytes (saved 32.0%); 21,401 GETs (1 top-ups), 120 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `25J1Q1DZHNMHK87VK93G`, 2.57 GB on disk |
| step5_check.py `--samples 8` | 22 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 50 | `cgan-ifs-6h-ens/IFS_20261030_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261030_00Z.nc` 272 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 7 | dry run  OK: 7 windows of finite 1x384x352x28 input in 6s |

## 20261031 00z

Run 2026-11-08T07:11:36Z → 2026-11-08T07:19:51Z, **495 s (8.2 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 23.7s; virtual snapshot `9D5XBK4PBM1M0DZMZZTG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 375 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 57.1 msg/s; fetched 30.04 of 44.01 GB whole-message bytes (saved 31.7%); 21,400 GETs (0 top-ups), 146 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `GZJJWMH5FM9MYFZZ88SG`, 2.55 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 46 | `cgan-ifs-6h-ens/IFS_20261031_00Z.nc` 43 MB, `cgan-ifs-7d-ens/IFS_20261031_00Z.nc` 271 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261101 00z

Run 2026-11-08T07:19:54Z → 2026-11-08T07:28:16Z, **502 s (8.4 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 46 | 742478 refs from 340 .index files on GCS in 23.7s; virtual snapshot `289D04KY398J1B1QS1SG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 377 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 57.0 msg/s; fetched 29.97 of 44.05 GB whole-message bytes (saved 32.0%); 21,400 GETs (0 top-ups), 138 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `3Q158JQDAHMWSKHAVWX0`, 2.54 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 51 | `cgan-ifs-6h-ens/IFS_20261101_00Z.nc` 43 MB, `cgan-ifs-7d-ens/IFS_20261101_00Z.nc` 271 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 5 | dry run  OK: 7 windows of finite 1x384x352x28 input in 5s |

## 20261102 00z

Run 2026-11-08T07:28:19Z → 2026-11-08T07:36:44Z, **505 s (8.4 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 46 | 742478 refs from 340 .index files on GCS in 23.4s; virtual snapshot `T2S5AA3E18AQ425GQZM0`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 375 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 57.2 msg/s; fetched 30.07 of 44.33 GB whole-message bytes (saved 32.2%); 21,402 GETs (2 top-ups), 775 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `X2K6HN7P8E1FZR6E1100`, 2.54 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 52 | `cgan-ifs-6h-ens/IFS_20261102_00Z.nc` 43 MB, `cgan-ifs-7d-ens/IFS_20261102_00Z.nc` 272 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 4 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 8 | dry run  OK: 7 windows of finite 1x384x352x28 input in 7s |

## 20261103 00z

Run 2026-11-08T07:36:46Z → 2026-11-08T07:46:34Z, **588 s (9.8 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 46 | 742478 refs from 340 .index files on GCS in 23.3s; virtual snapshot `73J4T34G42TT1776FNM0`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 469 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 45.8 msg/s; fetched 29.78 of 43.98 GB whole-message bytes (saved 32.3%); 21,411 GETs (11 top-ups), 10,302 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `Z604TDDVEVR6DE09HZQ0`, 2.56 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 47 | `cgan-ifs-6h-ens/IFS_20261103_00Z.nc` 43 MB, `cgan-ifs-7d-ens/IFS_20261103_00Z.nc` 275 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 4 | dry run  OK: 7 windows of finite 1x384x352x28 input in 4s |

## 20261104 00z

Run 2026-11-08T07:46:36Z → 2026-11-08T07:58:13Z, **697 s (11.6 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 48 | 742478 refs from 340 .index files on GCS in 22.8s; virtual snapshot `C517T3EP957K0D7H6HV0`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 559 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 38.4 msg/s; fetched 29.66 of 43.88 GB whole-message bytes (saved 32.4%); 21,407 GETs (7 top-ups), 19,150 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `A2XT1449TT193R3Q32WG`, 2.59 GB on disk |
| step5_check.py `--samples 8` | 22 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 50 | `cgan-ifs-6h-ens/IFS_20261104_00Z.nc` 44 MB, `cgan-ifs-7d-ens/IFS_20261104_00Z.nc` 276 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 12 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 6 | dry run  OK: 7 windows of finite 1x384x352x28 input in 5s |

## 20261105 00z

Run 2026-11-08T07:58:15Z → 2026-11-08T08:09:46Z, **691 s (11.5 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 49 | 742478 refs from 340 .index files on GCS in 22.8s; virtual snapshot `B488C3KRZS3DNF763RHG`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 552 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 38.8 msg/s; fetched 29.67 of 43.89 GB whole-message bytes (saved 32.4%); 21,410 GETs (10 top-ups), 18,506 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `SJJYDXPGV9J2QZNN8PEG`, 2.65 GB on disk |
| step5_check.py `--samples 8` | 22 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 55 | `cgan-ifs-6h-ens/IFS_20261105_00Z.nc` 45 MB, `cgan-ifs-7d-ens/IFS_20261105_00Z.nc` 279 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 3 | dry run  OK: 4 windows of finite 1x384x352x56 input in 2s |
| step7_infer.py mvua-7d `--dry-run` | 10 | dry run  OK: 7 windows of finite 1x384x352x28 input in 9s |

## 20261106 00z

Run 2026-11-08T08:09:48Z → 2026-11-08T08:23:13Z, **805 s (13.4 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 47 | 742478 refs from 340 .index files on GCS in 24.2s; virtual snapshot `XMB1A3QM4CAN5R5X2430`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 689 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 31.1 msg/s; fetched 29.69 of 44.02 GB whole-message bytes (saved 32.6%); 21,404 GETs (4 top-ups), 18,083 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `FQFYS2WR0WBH1XVVGSV0`, 2.72 GB on disk |
| step5_check.py `--samples 8` | 21 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 43 | `cgan-ifs-6h-ens/IFS_20261106_00Z.nc` 46 MB, `cgan-ifs-7d-ens/IFS_20261106_00Z.nc` 282 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 2 | dry run  OK: 4 windows of finite 1x384x352x56 input in 1s |
| step7_infer.py mvua-7d `--dry-run` | 3 | dry run  OK: 7 windows of finite 1x384x352x28 input in 2s |

## 20261107 00z

Run 2026-11-08T08:23:14Z → 2026-11-08T08:33:57Z, **643 s (10.7 min)**.

| Stage | s | What happened |
|---|---:|---|
| step1_build_virtual.py | 40 | 742478 refs from 340 .index files on GCS in 17.3s; virtual snapshot `6PB38D4RMWRAX65WYP20`; read-back 8/8 chunks equal a direct decode |
| step4_realize.py `--no-cluster --threads 12` | 524 | 14 fields x 30 steps = 420 blocks x 51 members = 21,420 messages; 41.0 msg/s; fetched 29.41 of 43.70 GB whole-message bytes (saved 32.7%); 21,403 GETs (3 top-ups), 14,837 of the first 21,400 messages (counter saved every 200) read from the GCS copy after AWS 503 Slow Down; store `AESZ9K95TFCPV4SQGSS0`, 2.74 GB on disk |
| step5_check.py `--samples 8` | 20 | yes -- 420/420 chunks written; 8/8 bit-exact vs whole messages through the virtual store; **PASS** |
| step6_sewaa_netcdf.py | 49 | `cgan-ifs-6h-ens/IFS_20261107_00Z.nc` 46 MB, `cgan-ifs-7d-ens/IFS_20261107_00Z.nc` 282 MB (51 members, sd ddof 0) |
| step7_infer.py jurre-6h `--dry-run` | 3 | dry run  OK: 4 windows of finite 1x384x352x56 input in 2s |
| step7_infer.py mvua-7d `--dry-run` | 7 | dry run  OK: 7 windows of finite 1x384x352x28 input in 7s |

