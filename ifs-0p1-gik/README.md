# ifs-0p1-gik: ECMWF IFS ENS at 0.1° to cGAN inference inputs over East Africa

This folder runs the full path from the ECMWF open-data GRIB files to the NetCDF file the cGAN inference reads:

1. Build a **virtual** Icechunk store from the GRIB `.index` files.
2. Optionally publish that store to source.coop.
3. Start Frisky workers on the EWC VMs.
4. **Realize** the 14 cGAN fields over the East Africa box into a real Icechunk store.
5. Check the realized store.
6. Optionally publish the realized store to source.coop.
7. Write `IFS_YYYYMMDD_HHZ.nc` on the TF cGAN's own 384 × 352 frame.
8. Run the cGAN inference on it (`step7_infer.py`) and optionally upload the output; the IFS NetCDF is then deleted.

The workflow runs in two modes:

| Mode | Where it runs | What it uses |
|---|---|---|
| **local** (`run_local.sh`) | One machine, no cluster | Reads the **public virtual store** anonymously, or builds it locally with `BUILD_VD=1`. It realizes only the lead times the SEWAA products need into a local directory: step 4 runs on threads in one process. It writes the SEWAA NetCDFs (`step6_sewaa_netcdf.py`), then runs both operational models (`step7_infer.py`). **Nothing is uploaded: no source.coop writes, no Ceph, no EWC workers, no Dask/Frisky cluster.** About 12 min per cycle (§0). |
| **ewc** (`run_ewc.sh`) | EWC Frisky workers | Writes to Ceph, publishes to source.coop, builds the full per-member NetCDF from the public copy, runs the inference. This is the daily driver of the public stores. |

**The SEWAA products (§11)** are the two models the SEWAA package runs operationally: Jurre-Brishti (6 h) and Mvua-Kubwa (7 × 24 h). Their method, input layout and code come from `icpac-igad/sewaa-forecasts-package`, `icpac-igad/fast-cgan` and `jaysnm/ensemble-cgan`, at the commits listed in **`vendor/PROVENANCE.md`**. Here they are made self-contained: the IFS NetCDF that production pulls by SFTP is built from the ECMWF open data, and the generator code is vendored.

The code was moved here from `grib-index-kerchunk` (branch `ecmwf-0p1-survey`, `1d12c38`). It comes from:
- `ecmwf/gcs-source`
- `ecmwf/icechunk-par`
- `ecmwf/0p1/realize-0p1`
- `ecmwf/0p1/wide-box/realize`

It has been cut down to this one box and these 14 fields; §12 lists what came from where.

---

## 0. Measured

### 2026-11-08, daily end-to-end continued, ten cycles 20261030–1108 00z

Report: **[`runs/2026-11-08-local-10day-endtoend.md`](runs/2026-11-08-local-10day-endtoend.md)**; timings in [`runs/2026-11-08-local-10day-timing.tsv`](runs/2026-11-08-local-10day-timing.tsv). This used the same setup as the run below. The last cycle, 20261108, was taken as soon as ECMWF finished publishing it.

| | Mean per cycle | Range |
|---|---|---|
| step 4, realize 21,420 messages (~29.4–30.2 GB fetched) | 473 s | 375–689 s |
| **total** | **9.9 min** | **8.2–13.4 min**; 1.65 h for all ten |

- **Every cycle passed step 5.**
- **The four fastest cycles, 20261030–1102,** had S3 answering quickly: GET ~0.68 s, ~57 msg/s, step 4 ~377 s. Only 43% of messages came from GCS overall.
- **20261108 had to be rerun once.** GCS publishes a step's `.index` before its `.grib2`, so step 1's read-back failed on a reference into a file that was not yet there. `step1_append_when_published.sh` now waits for the `.grib2` files too.

### 2026-10-30, daily end-to-end, local mode with no cluster, nine cycles 20261021–29 00z

Full report: **[`runs/2026-10-30-local-9day-endtoend.md`](runs/2026-10-30-local-9day-endtoend.md)**. It gives one section per date, plus the commands, code commit, environment and weights. Per-stage seconds are in [`runs/2026-10-30-local-9day-timing.tsv`](runs/2026-10-30-local-9day-timing.tsv).

**What was run.** `run_local.sh <date>` once per cycle, in sequence, as a daily job would run, with `BUILD_VD=1 PRODUCTS="jurre-6h mvua-7d"`:
- one 8-vCPU machine;
- step 4 with **no cluster** (`--no-cluster --threads 12`);
- the virtual store built locally for each cycle;
- step 7 as `--dry-run` (no weights loaded), so the times stop just before the generator.

The code is at `185b789`.

| Stage | Mean per cycle | Range |
|---|---|---|
| step 1, local virtual store (340 `.index` files, 742k refs) | 48 s | 47–49 s |
| step 4, realize 21,420 messages (~29.5 of ~43.6 GB fetched) | 595 s | 551–747 s |
| step 5, check (8/8 bit-exact) | 20 s | 19–21 s |
| step 6, both SEWAA NetCDFs (~45 MB + ~280 MB) | 41 s | 37–48 s |
| step 7, both products, input check only | 6 s | 4–10 s |
| **total** | **11.8 min** | **11.0–14.3 min**; 1.78 h for all nine |

- **Every cycle passed step 5.**
- **The download dominates.** Step 4 is ~82% of each cycle and is network time: mean GET 1.1–1.3 s against ~35 ms decode. S3 answered 503 Slow Down for most GETs, so ~85–90% of messages came from the byte-identical GCS copy.
- **On disk:** 26 GB of realized stores, 151 MB of virtual store and 2.8 GB of NetCDFs for the nine cycles.

### 2026-10-29, SEWAA products, local mode on one 8-core machine, cycle 20261026T00z

All of it reads the **public virtual store** (`DH6AJJSWJZ3VE2BCMERG`) anonymously and writes only local files (`run_local.sh`, local Frisky cluster 4 × 4, all 51 members).

| | `PRODUCTS=jurre-6h` | `PRODUCTS="jurre-6h mvua-7d"` |
|---|---|---|
| Lead times realized | +30 … 54 h (5) | +0 … 174 h (30) |
| Realize (step 4) | 3,570 msgs in **74 s**, 48 msg/s; **4.9 of 7.2 GB** fetched (31.9% saved) | 21,420 msgs in **424 s**, 51 msg/s; 29.7 of 43.7 GB fetched (32.1% saved) |
| Realized store | 0.43 GB | 2.5 GB |
| Check (step 5) | PASS: 70/70 chunks, 8/8 bit-exact | PASS: 420/420 chunks, 8/8 bit-exact |
| SEWAA NetCDF (step 6) | `cgan-ifs-6h-ens/IFS_20261026_00Z.nc` **45 MB**, 6 s | the same 45 MB file (identical) + `cgan-ifs-7d-ens/IFS_20261026_00Z.nc` **278 MB**, 36 s for both |
| Inference (step 7) | dry run OK: 4 windows × 56 ch; `--untrained`, 4 samples: 42 s | dry run OK: 7 days × 28 ch, 6–30 … **150–174 h**; `--untrained`, 4 samples: 42 s |

For comparison, the per-member file of §4 step 6 is 0.48 GB for the same 5 lead times and ~2.75 GB for 29.

### 2026-10-27, box 19–55°E, seven cycles 20261020T18z … 20261026T00z (ewc mode)

| Cycle | Realize (5 VMs × 6 tasks) | Saved by prefix read | Store | Publish | TF-frame NetCDF | step 7 dry run |
|---|---|---|---|---|---|---|
| 20261020T18z (25 lead times) | 266 s, 67.0 msg/s | 32.6% | 2.17 GB | 23 s | 2.31 GB, 218 s | OK, 24 windows |
| 20261021T00z | 330 s, 62.8 msg/s | 32.4% | 2.56 GB | 33 s | 2.72 GB, 310 s | OK, 28 windows |
| 20261022T00z | 302 s, 68.5 msg/s | 32.3% | 2.58 GB | 58 s | 2.75 GB, 287 s | OK |
| 20261023T00z | 328 s, 63.1 msg/s | 32.3% | 2.58 GB | 64 s | 2.75 GB, 263 s | OK |
| 20261024T00z | 425 s, 48.8 msg/s | 32.2% | 2.58 GB | 67 s | 2.76 GB, 250 s | OK |
| 20261025T00z | **2363 s, 8.8 msg/s** | 32.0% | 2.59 GB | 36 s | 2.77 GB, 480 s | OK |
| 20261026T00z | **1553 s, 13.3 msg/s** | 31.8% | 2.59 GB | 56 s | 2.76 GB, 395 s | OK |

- Every cycle passed step 5 on Ceph (406/406 chunks, 350/350 for 18z, 12/12 bit-exact) and again on source.coop, read anonymously.
- **The two slow cycles were the GRIB reads, not this setup.** Worker memory stayed at 1.5–1.9 GB per VM and CRMA was not running, but the mean GET time doubled (3.4 s against 1.6 s), and the last 86 of 406 blocks took 22 min. That looks like AWS throttling or slow servers on ECMWF's side.
- **NetCDF size.** On the TF frame the file is ~2.75 GB, more than on the store grid (2.36 GB): the four-point mean makes the values compress worse. The HDF5 shuffle filter did not help.
- **Inference.** Tested with an untrained generator (`--untrained`, the GEFS v3 `setup_params.yaml` as the stand-in architecture): 2 windows × 4 samples in 16 s on 8 CPU cores, output 384 × 352 on the frame. A full cycle (28 windows × 50 samples) extrapolates to ~30 min on CPU. The trained IFS weights are not on this machine (§8).

### 2026-10-27, first trial, box 20–53°E, cycle 20261021 00z

| | local mode (small trial) | ewc mode (full cycle) |
|---|---|---|
| Virtual store, one cycle | 742,478 refs from 340 `.index` files, 44 s, 17 MB; 8/8 read-back equal | (public store, snapshot `6AK8YKP0TDZJ085CV960`) |
| Realize | 14 fields × 4 steps × 6 members = 336 msgs, 12 s, 28 msg/s, gateway 2 × 4 | **14 × 29 × 51 = 20,706 msgs in 325 s, 63.8 msg/s**, 5 VMs × 6 tasks, 0 failed blocks |
| Bytes fetched | — | **28.4 of 42.1 GB (32.5% saved)** by the prefix read; 5 top-up GETs |
| Worker memory | — | VM used ≤ 1.8 GB of 16 GB |
| Realized store | — | **2.35 GB per cycle**, 436 objects |
| Check (step 5) | PASS: 56/56 chunks, 8/8 bit-exact | PASS on Ceph: 406/406 chunks, 12/12 bit-exact. PASS on source.coop (anonymous): 4/4 |
| Publish to source.coop | not used | 2.35 GB in 25 s (95 MB/s), 0 errors |
| NetCDF (step 6) | 5 members, 4 lead times, 0.03 GB, 2 s | from source.coop, anonymous: 14 fields × 29 lead times × **10 members, 0.43 GB in 404 s** (first field 259 s, cold; then ~10 s per field; a repeat read of one field takes 9.5 s). The full 50-member file extrapolates to **~2.2 GB** |

---

## 1. What is built

**Box.** lat 25°N → 15°S and lon 19°E → 55°E at 0.1°, which is **401 × 361** points. Every edge lies on the 0.1° grid (origin −180°, 90°N), so the cut is exact. It covers the TF cGAN training frame (384 × 352 at −13.65 … 24.65°N, 19.15 … 54.25°E) with a margin on every side for interpolating onto it. The first trial used 20 … 53°E (`20e53e-25n15s`), which did not cover that frame; that store is superseded.

**Step-6 output frame.** The TF cGAN does not read latitude/longitude from its input: it assumes the 384 × 352 frame at −13.65 … 24.65°N, 19.15 … 54.25°E. `step6_make_cgan_netcdf.py --frame tf` (the default) writes that frame. Each frame point lies exactly halfway between four 0.1° points (x.x5), so it is the mean of those four, which is bilinear interpolation at a cell centre. `elev.nc`, `lsm.nc` and `FCSTNorm2018.pkl` are on the same frame, ascending latitude.

**Fields.** These are the 14 cGAN forecast fields (`data.all_fcst_fields`, see `RFE2_cGAN_RUNBOOK.md`):

| cGAN name | `.index` param | level | notes |
|---|---|---|---|
| `cape` | `mucape` | sfc | ECMWF publishes most-unstable CAPE under `mucape` since 2024-11-13. There is no `cape` in the 0.1° files. |
| `cp`, `tp` | `cp`, `tp` | sfc | m, **accumulated from the init** |
| `ssr` | `ssr` | sfc | J m⁻², **accumulated from the init** |
| `mcc` `sp` `tciw` `tclw` `tcrw` `tcw` `tcwv` | same | sfc | |
| `t2m` | `2t` | sfc | |
| `u700`, `v700` | `u`, `v` | 700 hPa | |

**Lead times.** 0 … 168 h every 6 h, which is 29 steps. These are the `valid_time`s that `ensemble-forecast.py` steps through with `HOURS = 6`. All of them exist in both the 00z and 12z runs. The 06z and 18z runs stop at 144 h, so their stores and NetCDFs get the **25** lead times 0 … 144 h; `step4_realize.py` trims them and `step6` warns.

**Members.** All 51: the control (`number = 0`, from `oper/fc`) and the 50 perturbed members (from `enfo`). Step 6 chooses which go into the NetCDF.

Everything above is set in **`config.py`**, which is the only place to change it.

---

## 2. Data flow

```
 ECMWF open data (same bytes in both)
   s3://ecmwf-forecasts/<date>/<HH>z/ifs/0p1/{enfo,oper}/...-{pl,sfc}.grib2  (+ .index)
   gs://ecmwf-open-data/   (same paths)
        |
        |  step1_build_virtual.py: reads only the .index files (340 per cycle)
        v
 VIRTUAL store  (Icechunk, references only, ~13-17 MB per cycle)
   local:  ~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd
   public: source.coop forecasts/ecmwf-ifs-0p1-icechunk-vd    <- step 2, optional
        |
        |  step4_realize.py: coordinates and provenance from the virtual store;
        |  the bytes come from one Range GET of each message's prefix,
        |  decoded with libaec (cgan_dag.read_prefix)
        v
 REALIZED store  (Icechunk, real float32 values, 2.35 GB per cycle, one store per cycle)
   local:  ~/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
   EWC:    ceph  must-icechunk/cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
   public: source.coop forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z
        |
        |  step5_check.py: complete, bit-exact, sane
        |  step6_make_cgan_netcdf.py --frame tf   (384 x 352, 4-point mean)
        v
 IFS_<YYYYMMDD>_<HH>Z.nc  (temporary, ~2.75 GB; deleted after step 7)
        |
        |  step7_infer.py   (TF 2.15, trained generator weights)
        v
 GAN_IFS_<YYYYMMDD>_<HH>Z.nc  precipitation mm/h, (time, member, valid_time, 384, 352)
        |
        |  UPLOAD_GAN=1 only (off for now): upload_files.py
        v
 source.coop forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/
```

What each step reads and writes, by mode:

| Step | local mode | ewc mode |
|---|---|---|
| 1 virtual store | `.index` from GCS → local dir | same, then optionally step 2 |
| 2 publish virtual | — | local dir → source.coop (`step1_append_when_published.sh` with `PUBLISH=1`) |
| 3 workers | — | `step3_deploy_workers.py` |
| 4 realize | local Frisky cluster, local virtual store → local dir | EWC Frisky workers, public virtual store → Ceph |
| 5 check | local | Ceph, then the source.coop copy, anonymously |
| publish realized | — | Ceph → source.coop (`publish_ceph_to_source_coop.py`) |
| 6 NetCDF | from the local dir | from source.coop, anonymously; or from Ceph or local |
| 7 inference | `step7_infer.py` on the NetCDF | same; output uploaded only with `UPLOAD_GAN=1` |

**source.coop is never required.** It is the public copy, for other people and other machines. Every script also takes `local:` and `ceph:` stores (`stores.py`). The only network dependency the local mode keeps is ECMWF's own buckets, which it reads anonymously.

---

## 3. Setup

```bash
cd cGAN_tutorial/ifs-0p1-gik
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

The pins match the worker venv that step 3 builds on each VM. `imagecodecs` supplies libaec for the prefix decode.

Step 7 needs TensorFlow 2.15 (the version the TF cGAN was trained with, `env/env-tf.yaml`), which needs Python ≤ 3.11, so it has its own venv (~1.3 GB):

```bash
uv venv .venv-infer --python 3.11
uv pip install --python .venv-infer/bin/python -r requirements-infer.txt
```

Credentials are needed only in EWC mode. They are read from files and never printed.

| For | What | Used by |
|---|---|---|
| Ceph | `KEY=VALUE` file with `AK=` and `SK=` (`CEPH_ENV`) | step 4 sink, step 5, publish |
| source.coop writes | `.env.sourcecoop` (`SC_ENV`) plus the `source-coop` CLI (`source-coop creds --format env`) | step 2, publish |

Reading from source.coop needs **no** credentials.

**The `AWS_*` trap.** icechunk caches its object-store configuration for the whole process. A Ceph `AWS_ENDPOINT_URL` left in the environment would redirect the anonymous reads of `s3://ecmwf-forecasts`. So:
- `cgan_dag.py` deletes every `AWS_*` variable on import, and every script imports it first;
- the workers are started without them;
- the publish scripts receive the source.coop STS credentials in a subshell only.

---

## 4. Step by step

### Step 1: virtual store (`step1_build_virtual.py`)

```bash
.venv/bin/python step1_build_virtual.py --date 20261021 --run 00 --resolution 0p1 \
    --store ~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd --verify 12
# or wait for publication, append, and optionally publish:
bash step1_append_when_published.sh 2026102700                 # local only
PUBLISH=1 SC_ENV=... bash step1_append_when_published.sh 2026102700
```

- **Inputs.** It reads the 340 `.index` files of one cycle from GCS: 85 steps × {enfo, oper} × {pl, sfc}.
- **Commit.** It writes all 742,478 references (all 61 variables, not only the 14) as one commit to group `0p1/<HH>z`.
- **Verification.** It reads N random chunks back through the store and compares each with a direct gribberish decode.
- **No templates.** Nothing from the old pars/template pipeline is needed. `grids.py` and `build_ecmwf_icechunk.py` sit next to it. `grids.py` is **byte-identical** to `grib-index-kerchunk/ecmwf/grids.py`, and the builder refuses to run if two copies it can reach differ. That is the guard against the 180° longitude bug.
- **When the data appears.** 0.1° files appear about 2 h after the 0.25° files of the same run. `step1_append_when_published.sh` polls for the last step's four `.index` files every 5 min, for up to 10 h.

### Step 2: publish the virtual store (optional)

`publish_local_to_source_coop.py` copies the local store directory to `forecasts/ecmwf-ifs-0p1-icechunk-vd`:
- the copies are plain object copies;
- icechunk objects are immutable, so objects already there at the same size are skipped;
- the `repo` pointer goes up **last**, so a reader never sees a pointer to missing objects.

### Step 3: EWC workers (`step3_deploy_workers.py`), EWC mode only

```bash
D=/opt/mamba/envs/dask/bin/python          # has `distributed`; the Dask cluster is only a remote-exec channel
$D step3_deploy_workers.py --install       # venv in /tmp/frisky-cgan-ea on every VM (~40 s)
$D step3_deploy_workers.py --scheduler-on 192.168.1.164 --start --nthreads 6 --memory-limit 12GB
$D step3_deploy_workers.py --status
$D step3_deploy_workers.py --stop          # this workflow's workers and scheduler only
```

- **Isolated.** The workers run in their own directory (`/tmp/frisky-cgan-ea`) on their own ports: scheduler **8798**, dashboard 8793. The CRMA daily routine (`/tmp/frisky-ea`, :8796) is never touched. `--stop` matches only this directory's processes.
- **No credentials.** The workers are started with `AWS_*` scrubbed from *their* environment only.
- **One file shipped.** `cgan_dag.py` is the only file the workers need. It imports nothing from this folder; the box and field table arrive as task arguments. So changing `config.py` needs no redeploy.
- **Not alongside CRMA.** Don't run this while the CRMA daily run is active; both use every VM. `run_ewc.sh` waits for it.

### Step 4: realize (`step4_realize.py`)

```bash
# local, no cluster: 12 block tasks on threads in this process (run_local.sh's default)
.venv/bin/python step4_realize.py --date 20261021 \
    --vd local:$HOME/icechunk-local/ecmwf-ifs-0p1-icechunk-vd \
    --sink local:$HOME/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z \
    --no-cluster --threads 12
# local Frisky cluster instead: --workers 2 --threads 4
# EWC
.venv/bin/python step4_realize.py --date 20261021 --sink ceph:cgan14-19e55e-25n15s/20261021T00z \
    --env $CEPH_ENV --scheduler 192.168.1.164:8798
```

What it does:

1. **Opens the virtual store.** It checks that the cycle and all 29 steps are there. It takes the box's latitude/longitude from the store and records the store's **snapshot ID** in the output, so you can always see which version of the store a realization came from.
2. **Lays down the schema.** One commit sets up the 14 arrays, each `float32 (time=1, step=29, number=51, latitude=401, longitude=361)` (step=25 for 06z/18z). A chunk is one (field, step) block, 29.5 MB raw. The **fill is NaN**, so a block that was never written can't pass as data.
3. **Submits one task per block** (406 tasks). Each `cgan_dag.read_write_block`:
   - reads its 51 messages with 4 threads into one preallocated array;
   - **writes the chunk from the same worker** into a shared icechunk *fork*;
   - returns the fork and exact byte counts.

   A sliding window (`--max-inflight 40`) bounds what is queued. Failed blocks are resubmitted in up to 4 rounds.
4. **Commits once.** It merges all forks into **one commit**, then adds `realize_summary` (bytes, rate) to the root attributes.

**The prefix read** (`cgan_dag.read_prefix`). This is a stand-in until gribberish can decode the first N rows of a message.
- **Why a prefix holds the box.** ECMWF packs 0.1° fields with CCSDS (libaec), rows north → south, and Section 7's first block starts at bit 0. So rows 90°N … 15°S are a **prefix** of each message.
- **The read.** One HTTP Range GET from the message start, decoded with libaec, gives the box exactly. gribberish 1.6.0 rejects a truncated message, so the GRIB2 header walk and the libaec call live in `cgan_dag.py` for now.
- **Byte offsets.** They come from the same GCS `.index` files step 1 used (icechunk 2.1 has no public call that returns a virtual reference).
- **AWS first, GCS fallback.** Data GETs go to AWS first. On a `503 Slow Down` they go to the byte-identical GCS copy.
- **Sizing the first GET.** Each worker learns how much of each field's message the box needs, from libaec's consumed-byte count. Its first GET for that field is the largest share seen so far plus 1.5%. A short guess costs a top-up GET, never a wrong value: 5 top-ups in 20,706 messages.
- **Saving.** 32.5% on this cycle. The rows kept are 58% of the message, but each field's compressed bytes are not spread evenly by row, and the first messages of each field on each worker start from a cautious guess.
- **Exactness.** Bit-exact, re-checked on every run by step 5 (and 736/736 against eccodes and gribberish in `grib-index-kerchunk/ecmwf/0p1/wide-box/README.md` §1).

`--full-read` reads whole messages through the virtual store and gribberish instead: slower, but with no libaec step.

**`--no-cluster`** runs the same `read_write_block` tasks on `--threads` threads inside the step 4 process, with no Frisky. Each task gets a pickled copy of the fork, as a worker would, so the merge and the commit are unchanged. The GETs and the libaec decode release the GIL. On 20261026 `jurre-6h`, in the same hour, it took 126 s with 12 threads against 143 s on the 2 × 4 local cluster. The executor is recorded in `realize_summary`.

**Why one task per block, not one per message.** With separate read tasks, every member's box lands on whichever worker read it, and the writer then pulls all of them across the LAN. On the wide box that limited the run to 2.6–5.7 msg/s instead of 16 msg/s (`grib-index-kerchunk/ecmwf/0p1/wide-box/README.md` §4).

### Step 5: check (`step5_check.py`)

It checks three things:

1. **complete:** every (field, step) chunk is in the manifest. A missing chunk reads as NaN, not as an error, so this has to be asked.
2. **exact:** `--samples` random (field, step, member) boxes are compared with a whole-message read through the virtual store. They must be **equal**, not close.
3. **sane:** all values are finite; members differ; `tp`, `cp` and `ssr` are 0 at step 0 and never decrease with lead time.

It exits non-zero on any failure, and `run_ewc.sh` publishes nothing unless it passes.

### Publish the realized store

`publish_ceph_to_source_coop.py` copies Ceph to source.coop, sending the `repo` pointer last. It uses multipart upload above 64 MiB because source.coop refuses single PUTs over ~100 MB; chunks here are smaller than that. For a local store, use `publish_local_to_source_coop.py --src <dir> --dest forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<cycle>`.

### Step 6: cGAN NetCDF (`step6_make_cgan_netcdf.py`)

```bash
.venv/bin/python step6_make_cgan_netcdf.py \
    --store sourcecoop:forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z --outdir nc
```

It writes `IFS_20261021_00Z.nc` in the layout **`scripts/ensemble-forecast.py`** reads, on the model's 384 × 352 frame (`--frame tf`, default; `--frame native` keeps the 401 × 361 store grid):

| variable | dims | content |
|---|---|---|
| `time` | (time=1) | init, hours since 1900-01-01 00:00:00.0 |
| `valid_time` | (valid_time=29; 25 for 06z/18z) | init + 0 … 168 h, hours since 1900-01-01 |
| `member`, `latitude` (ascending), `longitude` | | |
| `<field>_ensemble_mean` × 14 | (time, member, valid_time, latitude, longitude) | **every member**, float32, zlib 1 |

- **Variable name.** Despite the name, `_ensemble_mean` holds the members. That is the name the inference reads (`nc_in[f"{field}_ensemble_mean"][0, :, i:i+2]`), and the inference takes the mean and sd over members itself.
- **Accumulated fields.** `tp`, `cp` and `ssr` stay accumulated; the inference differences consecutive valid times.
- **Members.** `--members` defaults to `perturbed` (1 … 50, the 50 the script's comments describe). Use `all` to add the control.
- **Memory.** It reads one field at a time, as 29 chunks fetched `--threads` (6) at once, so memory peaks at ~0.8 GB.
- **Size.** 14 × 50 × 29 × 384 × 352 float32 is 11 GB raw, and **~2.75 GB** with zlib 1 (§0). `--members N` and `--fields` cut it down for tests.
- **No averaging needed beforehand.** The inference takes the ensemble mean and sd itself, and differences the accumulated fields.

### Step 7: cGAN inference (`step7_infer.py`)

```bash
.venv-infer/bin/python step7_infer.py --nc nc/IFS_20261021_00Z.nc \
    --model-folder /path/ICPAC-big-ensmeansd --checkpoint 316800 \
    --constants ../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS --outdir gan
.venv/bin/python step7_infer.py --nc nc/IFS_20261021_00Z.nc --constants ... --dry-run   # no TF
```

- **The same computation as `scripts/ensemble-forecast.py`.** Per 6-hour window and field it builds the 4 channels (mean/sd at start and end, or mean/sd of the accumulation), applies the same normalisation with `FCSTNorm2018.pkl`, adds `elev`/`lsm`, and runs the generator `--ensemble-members` times (50) with fresh noise. Output `GAN_IFS_<YYYYMMDD>_<HH>Z.nc`, `precipitation` in mm h⁻¹, `fcst_valid_time` = start of each window, as the original writes it.
- **Differences from the original script.** Paths are arguments, not `forecast.yaml`/`data_paths.yaml`. The input's frame is checked first. Members are predicted in batches (`--batch 10`). It loads only `model/models.py` and `model/noise.py` from `tensorflow-dev-test`, not `model/__init__.py`, which pulls in the training stack (cartopy, data paths); `keras.layers._Merge`, gone from Keras 2.15's top level, is aliased.
- **`--end-hour`** defaults to the file's last valid time (168 h; 144 h for 06z/18z).
- **`--dry-run`** builds and checks every network input (finite, 1 × 384 × 352 × 56) without TensorFlow or weights. `run_ewc.sh` falls back to it when `MODEL_FOLDER`/`CHECKPOINT` are unset.
- **`--untrained`** is for testing only: it builds the generator without loading weights.
- **Generator code.** It is built from `vendor/ensemble_cgan/` (`--gen-code`), the production `models.py`/`blocks.py`/`layers.py`/`noise.py`, no longer from `tensorflow-dev-test`.
- **Layout.** It is read from the file: the `cgan_product` attribute (`jurre-6h`, `mvua-7d`) selects the SEWAA computation (§11); a file without it is the per-member layout.

### Upload the inference output (off for now)

`run_ewc.sh` with `UPLOAD_GAN=1` copies `GAN_IFS_*.nc` to source.coop `forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/` with `upload_files.py` (multipart, size-checked; `--to ceph:<prefix>` also works). Without it the output stays in `gan/`.

---

## 5. Running it

```bash
# the SEWAA products on this machine: public virtual store in, nothing uploaded (§11)
bash run_local.sh 20261026                                  # 00z, jurre-6h + mvua-7d
PRODUCTS=jurre-6h bash run_local.sh 2026102612              # 6 h model only, 12z
JURRE_MODEL_FOLDER=/path/jurre-brishti/model-weights JURRE_CONSTANTS=/path/jurre-brishti/constants \
  MVUA_MODEL_FOLDER=/path/mvua-kubwa/model-weights MVUA_CONSTANTS=/path/mvua-kubwa/constants \
  bash run_local.sh 20261026                                  # with the trained models
BUILD_VD=1 bash run_local.sh 20261026     # build the cycle's virtual store locally (step 1) instead
# several days, timed per stage into $TIMING (default logs/timing_local.tsv), report as in runs/
for d in 21 22 23; do BUILD_VD=1 bash run_local.sh 202610$d; done

# EWC + Ceph + source.coop: the daily run
APPEND_VD=1 CEPH_ENV=~/.../.env SC_ENV=~/.../.env.sourcecoop \
  MODEL_FOLDER=/path/ICPAC-big-ensmeansd CHECKPOINT=316800 TFPY=$PWD/.venv-infer/bin/python \
  bash run_ewc.sh 20261027
# several cycles in one go (workers started once)
CEPH_ENV=... SC_ENV=... bash run_ewc.sh 2026102018 2026102100 2026102200
# cycles already realized and published: NetCDF + inference only
SKIP_REALIZE=1 SC_ENV=... bash run_ewc.sh 20261021
```

`run_local.sh` env switches:
- `THREADS`: step 4 block threads, default 12;
- `CLUSTER=1`: use a Frisky local cluster, `WORKERS` × `THREADS`;
- `TIMING`: the per-stage seconds file;
- `MEMBERS`, `NC_MEMBERS`, `DDOF`;
- `*_MODEL_FOLDER`, `*_CHECKPOINT`, `*_CONSTANTS`;
- `GAN_MEMBERS`, `UNTRAINED`.

`run_ewc.sh` env switches: `UPLOAD_GAN=1` uploads the inference output (off by default), `KEEP_NC=1` keeps the IFS NetCDF (deleted after step 7 by default), `CONSTANTS` (default `../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS`), `NC_MEMBERS`, `GAN_MEMBERS`.

The order inside `run_ewc.sh`: realize and check every cycle with the workers up, stop the workers, publish and check every cycle (~30–60 s each), then NetCDF + inference per cycle. Publishing first means one slow NetCDF does not hold back the public stores.

**Disk needed in local mode:** see §11 for the realized store and NetCDF per product. Also ~17 MB per cycle of local virtual store (`BUILD_VD=1` only), and 1.3 GB for `.venv-infer`.

The gateway has only a few GB free, so `run_ewc.sh` keeps one NetCDF at a time and deletes it after step 7.

---

## 6. Where everything is stored

| What | local | EWC Ceph (`must-icechunk`) | source.coop (`e4drr-project`) |
|---|---|---|---|
| virtual store | `~/icechunk-local/ecmwf-ifs-0p1-icechunk-vd` | — | `forecasts/ecmwf-ifs-0p1-icechunk-vd` |
| cGAN inference output (`UPLOAD_GAN=1`) | `gan/GAN_IFS_<YYYYMMDD>_<HH>Z.nc` | — | `forecasts/ecmwf-ifs-0p1-cgan-gan6h/<YYYYMMDD>T<HH>z/` |
| realized, one store per cycle | `~/icechunk-local/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` | `cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` | `forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/<YYYYMMDD>T<HH>z` |

Published realized cycles: `20261020T18z`, `20261021T00z` … `20261026T00z` under `forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/`. The first trial's 20–53°E store has been deleted from source.coop and Ceph. Opening one anonymously:

```python
import icechunk as ic, xarray as xr
st = ic.s3_storage(bucket="e4drr-project", endpoint_url="https://data.source.coop",
                   prefix="forecasts/ecmwf-ifs-0p1-cgan14-19e55e-25n15s/20261021T00z",
                   region="us-east-1", anonymous=True, from_env=False, force_path_style=True)
ds = xr.open_zarr(ic.Repository.open(st).readonly_session("main").store, consolidated=False)
ds.tp.isel(step=4).mean("number")        # 24 h accumulated, m
```

---

## 7. Retention

**One store per cycle** is deliberate: deleting a cycle is a prefix delete. Deleting one date inside a multi-date icechunk store would need `expire_snapshots` and `garbage_collect`.

Once the NetCDF (or a downstream product) is made and checked, the realized cycle can go. It can be **recreated exactly**:
- the virtual store keeps the references;
- the GRIB files stay in ECMWF's buckets;
- the realized store's attributes record the virtual-store snapshot and the read method;
- recreating takes ~6 min of the EWC cluster per cycle.

Keep the virtual store; it is ~17 MB per cycle.

The IFS NetCDF is an intermediate: it is rebuilt from the realized store in ~5 min, so `run_ewc.sh` deletes it once the inference has run, and nothing keeps it on Ceph.

---

## 8. Caveats

- **Trained weights.** None are in this repo; only a GEFS model is in `tensorflow-dev-test/cgan_compact_20260202`. The SEWAA Docker images carry none either: they mount `data/models-config/` at run time, and their two `ensemble-cgan` code layers are 0.3–0.4 MB. The public tarballs named in `sewaa-forecasts-package/manager.sh` returned 404 on 2026-10-29 and 2026-10-30. Checked 2026-10-30 against [`Fenwick-Cooper/SEWAA-forecasts`](https://github.com/Fenwick-Cooper/SEWAA-forecasts) @ `2fe8a40`:
  - **6 h, available.** `6h_accumulations/cGAN/ICPAC-big-ensmeansd/models/gen_weights-0316800.h5` is the operational checkpoint 316800. It has the same folder name as the Jurre_brishti branch's `forecast.yaml`, the same 14 fields and the same input. It loads into `vendor/ensemble_cgan` with 56 channels. Its `FCSTNorm2018.pkl`, `elev.nc` and `lsm.nc` have the same md5 as `../tensorflow-dev-test/cgan_compact_20260202/CONSTANTS`. Use `JURRE_MODEL_FOLDER=~/SEWAA-forecasts/6h_accumulations/cGAN/ICPAC-big-ensmeansd`.
  - **7 days, not available.** The operational Mvua-Kubwa checkpoint 300800 (14 fields, 28 channels) is in none of these places, only in a SEWAA deployment's `data/models-config/mvua-kubwa`.
  - **Not a substitute for the 7-day model.** SEWAA-forecasts' 24 h models are newer and per lead day: `logs_1` @ 203776 (day 0), `logs_5` @ 163840 (day 1) and `logs_17` @ 115200 (days 2–6). They have **13 fields (no `cape`), 26 channels** and their own `FCSTNorm2018.pkl`. They load only with 26 channels, so they would need their own step 7 mode. The `cgan-ifs-7d-ens` file already holds everything they read.

  Without weights, step 7 runs as `--dry-run`, or with `--untrained` for a test.
- **`scripts/ensemble-forecast.py`** had two typos (`sys.path.insert("../data/")` without its index, `fron config import`), now fixed. It still imports `data_old`, which is not in this repo; `step7_infer.py` is the runnable equivalent.
- **Normalisation.** `FCSTNorm*.pkl` must come from the same field set and grid as the model.
- **`cape` is `mucape`.** It is the same channel as the model's `cape` from 2024-11-13; before that, ECMWF published `cape`.
- **06z/18z** runs stop at 144 h, so they can't provide the 150–168 h lead times: their NetCDF has 25 valid times, not 29. `step7_infer.py` stops at 144 h for them by itself; `ensemble-forecast.py` needs `end_hour: 144`.
- **The prefix read is a stand-in for a gribberish feature.** It assumes grid template 3.0, packing 5.42 (CCSDS), no bitmap and scan mode 0, and raises on anything else. Every 0.1° field checked so far meets those conditions.
- **The local Frisky cluster** must be reached with `cluster.get_client()`. `frisky.Client(cluster)` raises, and the worker processes then keep the interpreter alive.

---

## 9. Files

| File | Role |
|---|---|
| `config.py` | Box, 14 fields, lead times, store names: **the one place to edit** |
| `stores.py` | Store specs `local:` / `ceph:` / `sourcecoop:`, open helpers |
| `cgan_dag.py` | Worker module: prefix read, virtual read, `read_write_block` |
| `step1_build_virtual.py` | `.index` → virtual Icechunk store (with `build_ecmwf_icechunk.py`, `grids.py`) |
| `step1_append_when_published.sh` | Wait for a cycle, append, optionally publish (step 2) |
| `step3_deploy_workers.py` | EWC Frisky workers: install / start / status / stop |
| `step4_realize.py` | Realize one cycle (local cluster or EWC) |
| `step5_check.py` | Complete, bit-exact, sane |
| `step6_make_cgan_netcdf.py` | Realized store → `IFS_YYYYMMDD_HHZ.nc` (TF frame by default) |
| `step6_sewaa_netcdf.py` | Realized store → the SEWAA `IFS_YYYYMMDD_HHZ.nc` (ensemble mean + sd) for `jurre-6h` / `mvua-7d` (§11) |
| `step7_infer.py` | cGAN inference on any of the three layouts → `GAN_*.nc` (`requirements-infer.txt`, `.venv-infer`) |
| `vendor/ensemble_cgan/` | Generator code from `jaysnm/ensemble-cgan` (MIT), byte-identical, plus the two operational scripts verbatim under `reference/` |
| `vendor/PROVENANCE.md` | Source repositories, branches, commits, and the operational routine this follows |
| `publish_local_to_source_coop.py`, `publish_ceph_to_source_coop.py` | Copy a store up, `repo` last |
| `mem_monitor.py` | EWC: per-VM memory every N s |
| `run_local.sh`, `run_ewc.sh` | End-to-end drivers (`run_ewc.sh` takes several cycles) |
| `upload_files.py` | Copy finished files (the inference output) to source.coop or Ceph |
| `logs/` | Logs of the runs measured in §0 |
| `runs/` | Run reports. `make_local_run_report.py` turns a `run_local.sh` log and timing TSV into a Markdown report with one section per cycle. `2026-10-30-local-9day-*` is the nine-day run in §0 |

## 10. Provenance

All of the following come from `grib-index-kerchunk` at `1d12c38` (branch `ecmwf-0p1-survey`):

| Here | From |
|---|---|
| `step1_build_virtual.py` | `ecmwf/gcs-source/index_to_icechunk.py`, import path changed |
| `build_ecmwf_icechunk.py` | `ecmwf/icechunk-par/` |
| `grids.py` | `ecmwf/grids.py`, identical |
| `step1_append_when_published.sh` | `ecmwf/0p1/realize-0p1/append_when_published.sh` |
| `publish_*` | `ecmwf/0p1/realize-0p1/mirror_local_to_source_coop.py` and `wide-box/realize/mirror_ceph_to_source_coop.py` |
| `step3_deploy_workers.py` | `wide-box/realize/deploy_workers.py` |

`cgan_dag.py` and `step4`–`step7` are new. `step7_infer.py` follows `scripts/ensemble-forecast.py` for the per-member layout. For `jurre-6h` and `mvua-7d` it follows `test_forecast.py` and `forecast_date.py` of `jaysnm/ensemble-cgan`, and `step6_sewaa_netcdf.py` writes the file those scripts read. The repositories, branches and commits are in `vendor/PROVENANCE.md`; the generator code in `vendor/ensemble_cgan/` is copied unchanged. They are written for this box and field set, after:
- `wide-box/realize/frisky_daily_dag.py`;
- `wide-box/realize/build_corpus.py`;
- `wide-box/realize/sink_icechunk.py`.

---

## 11. The SEWAA operational products, self-contained (`run_local.sh`)

**What production does.** SEWAA (`icpac-igad/sewaa-forecasts-package`) runs the cGAN as four Docker jobs from `icpac-igad/fast-cgan`. Every hour, `fastcgan/jobs/cgan_ifs.py` pulls a finished `IFS_<YYYYMMDD>_<HH>Z.nc` by SFTP from an external IFS server (`/data/Operational`, `/data/Operational_7d`), or from the ICPAC FTP. It then runs `test_forecast.py` (Jurre-Brishti, 6 h) or `forecast_date.py` (Mvua-Kubwa, 24 h) from `jaysnm/ensemble-cgan`. The script that **makes** that IFS file is in none of those repositories. Here it is `step6_sewaa_netcdf.py`, fed from the open data through the Icechunk stores. Repositories, branches, commits and the full routine: **`vendor/PROVENANCE.md`**.

**What each model reads, and so what has to be fetched:**

| Product | Operational script, config | Valid times in its IFS file | Windows | Network input | Messages to fetch (14 × 51) |
|---|---|---|---|---|---|
| `jurre-6h` | `test_forecast.py`, `start_hour 30`, `end_hour 54`, ckpt 316800 | **+30 … 54 h (5)**: the script hard-codes that the file starts at +30 h | 4 × 6 h, 30–36 … 48–54 h | 56 ch: mean/sd at start and end; accumulations mean/sd of the 6 h increment, 0, 0 | **3,570** (17% of the 29-step cycle) |
| `mvua-7d` | `forecast_date.py`, `start_hour 6`, `end_hour 174`, ckpt 300800 | **+0 … 174 h (30)**: indices 1 … 29 are read, index 0 is not | 7 × 24 h, 6–30 … 150–174 h | 28 ch: trapezium-rule mean/sd over the 5 instants; accumulations mean of the 4 increments, RMS of their sds | 21,420 |
| both | | union = +0 … 174 h | | | 21,420 (one realized store serves both) |

- **+174 h is possible.** The 00z and 12z runs are 3-hourly to 144 h and then **6-hourly to 360 h**, so 150, 156, 162, 168 and 174 h are all published. The public virtual store holds them for every cycle (checked: steps 144, 150 … 186 present for 00z and 12z). The old `config.STEPS` stopped at 168 h only because `ensemble-forecast.py` did. The last 24 h day (150–174 h) needs the instants 150 … 174 h and the increments 150→156 … 168→174 h, and both exist.
- **06z/18z** stop at 144 h. They can make `jurre-6h` (it ends at 54 h) but not `mvua-7d`; `run_local.sh` and step 4 refuse to try.
- **Pre-differenced accumulations.** In the SEWAA file, index j of `tp`, `cp` and `ssr` holds the per-member increment from valid time j to j+1, then mean/sd over members (`ensemble-cgan data.py load_fcst`: "data[:, j] has accumulation between times j and j+1"). The last index has no j+1 and is NaN; neither script reads it.
- **Members and sd.** No source repository states them. The default here is all 51 members and population sd (`ddof=0`); `--members`/`--ddof` on step 6 (`NC_MEMBERS`/`DDOF` in `run_local.sh`) change them. Check against one operational `IFS_*.nc` when one is at hand.
- **Mvua output name.** `forecast_date.py` writes `GAN_<date>_00Z.nc` whatever the run; `step7_infer.py` keeps the run's hour in the name.

**The flow in `run_local.sh`.** No step needs credentials or writes anywhere but this machine:

1. **Virtual store.** It reads the public one (`sourcecoop:forecasts/ecmwf-ifs-0p1-icechunk-vd`) anonymously; nothing is built or uploaded. The public store only has the cycles the EWC daily run appended. For any other cycle, `BUILD_VD=1` builds it locally with step 1, which is also only reads from ECMWF.
2. **Step 4** `--product …` realizes only the needed lead times, all 51 members, with no cluster (`--no-cluster`, 12 threads; `CLUSTER=1` for a local Frisky cluster), into `$IFS0P1_LOCAL_ROOT/ecmwf-ifs-0p1-cgan14-<box>/<YYYYMMDD>T<HH>z-<products>`.
3. **Step 5** checks it: complete, bit-exact against whole messages, and sane.
4. **`step6_sewaa_netcdf.py`** writes `$OUTDIR/cgan-ifs-6h-ens/IFS_<YYYYMMDD>_<HH>Z.nc` and `$OUTDIR/cgan-ifs-7d-ens/…`, under the folder names the SEWAA jobs read. A SEWAA deployment, or the IFS server's `Operational*` directories, can take these files as they are.
5. **`step7_infer.py`** runs each product's computation with the vendored production generator. It falls back to `--dry-run` when that model's folder is not set.

**Checks made on 20261026T00z:**
- The 6 h file's mean and sd match the realized store, recomputed separately at 40 random points: worst relative error 5 × 10⁻⁸.
- The 7-day file was checked the same way: worst relative error 5 × 10⁻⁸; the 6 h file written from the 30-step store is identical to the one written from the 5-step store.
- The `mvua-7d` channels equal those from the expressions in `forecast_date.py`, run verbatim on the same file, to float32 rounding: max |difference| ≈ 1 × 10⁻⁷ of each field's range.
- The `jurre-6h` network input equals the one the per-member path (`ensemble-forecast.py` logic) builds from the same members with the same sd: max |difference| 1.6 × 10⁻⁵ on O(1) inputs, which is float32 storage precision.
- The vendored generator builds and runs on both layouts with `--untrained`: 56 and 28 input channels.
