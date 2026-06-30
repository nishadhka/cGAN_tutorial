# RFE2 cGAN — Data Prep → source.coop → GPU Training Runbook

End-to-end, **run-one-after-another** steps to build the 14-field training
TFRecords on a cheap CPU box, publish them to **source.coop**, free local disk
as you go, then train on an expensive GPU reading from local NVMe.

> **Golden rule:** the GPU is expensive — everything in Phases 0–4 (download,
> norm, TFRecords, upload) is **CPU-only** and must be 100% finished and verified
> **before** the GPU is started. The GPU only ever sees the ~30–35 GB of
> TFRecords (+ small constants), never the ~388 GB of raw IFS.

---

## 0. Layout & sizing (read first)

| Stage | Data | Approx size | Lives where |
|-------|------|-------------|-------------|
| Raw IFS forecasts (14 fields × 4 years) | `IFS_training/<year>/*.nc` | **~388 GB** (~97 GB/yr) | CPU prep box (transient) |
| Regridded RFE2 truth | `RFE/<year>/<YYYYMMDD>.nc` | small (<1 GB) | CPU prep box |
| Constants | `cGAN_data/elev.nc`, `lsm.nc` | a few MB | everywhere |
| Norm constants | `FCSTNorm2018.pkl` | KB | everywhere |
| **TFRecords (the deliverable)** | `rfe_tfrecords/<year>_1.<bin>.tfrecords` | **~30–35 GB** | source.coop → GPU local |

Channels per patch with 14 fields: input `128×128×28` (2 ch/field) + constants
`×2` + truth `×1`. `input_channels = 2 × len(all_fcst_fields)` is derived
automatically — no manual channel edits.

**Disk-frugal strategy:** process **one year at a time** so peak disk ≈ one
year of raw IFS (~97 GB), not all four. Upload + delete each year's TFRecords
before moving on.

---

## 1. Field set — already configured for all 14 variables

These edits are **already applied** in the repo:

- `SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan/data.py`
  ```python
  all_fcst_fields   = ['cape','cp','mcc','sp','ssr','t2m','tciw','tclw',
                       'tcrw','tcw','tcwv','tp','u700','v700']
  accumulated_fields = ['cp','ssr','tp']
  nonnegative_fields = ['cape','cp','mcc','sp','ssr','t2m','tciw','tclw',
                        'tcrw','tcw','tcwv','tp']   # winds (u700/v700) stay signed
  ```
- `Scripts/download_ifs.py` → `VARS` = the same 14.

> Want exactly **13** (the original set)? Delete `'cape'` from both lists in
> `data.py` **and** from `VARS` in `download_ifs.py`. It then drops out of the
> norm, the TFRecords, and the model automatically.

---

## 2. One-time setup

### 2.0 TensorFlow env (`tf215gpu`, per Fenwick-Cooper SEWAA-forecasts)
```bash
micromamba create -y -n tf215gpu -c conda-forge python=3.11 pip
micromamba run -n tf215gpu pip install "tensorflow==2.15" xarray netcdf4 numpy pyyaml
# (add numba properscoring cartopy ... only if you also run eval on this box)
micromamba run -n tf215gpu python -c "import tensorflow as tf; print(tf.__version__)"
```
Use it for every Python step here: `export CGAN_PY='micromamba run -n tf215gpu python'`.
> ✅ Verified on real Oxford data (2021, all 14 fields): `write_data` emits
> `2021_1.{0,1,2,3}.tfrecords` that parse back with `generator_input (128,128,28)`,
> `constants (128,128,2)`, `generator_output (128,128,1)`. See `test_realdata/`.

```bash
# 2.1 Pick a Linux paths entry. Edit data_paths.yaml and add e.g.:
#   LINUX_PREP:
#       GENERAL: {TRUTH_PATH: '/data/CGAN/RFE/', FORECAST_PATH: '/data/CGAN/IFS_training/',
#                 CONSTANTS_PATH: '/data/CGAN/cGAN_data/', NORMALISATION_PATH: '/data/CGAN/', LEAD_IDX: 1}
#       TFRecords: {tfrecords_path: '/data/CGAN/rfe_tfrecords/'}
# 2.2 Point local_config.yaml at it:
#   data_paths: "LINUX_PREP"

# 2.3 Make sure the destinations exist
mkdir -p /data/CGAN/{IFS_training,RFE,cGAN_data,rfe_tfrecords}
# copy elev.nc + lsm.nc into /data/CGAN/cGAN_data/
```

---

## 3. Per-year prep loop (repeat for 2018, 2019, 2020, 2021)

Do **all four** sub-steps for one year, reclaim its disk, then move to the next.
`cd` into the model dir for the Python steps:

```bash
cd SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan
```

### 3a. Download raw IFS for the year (~97 GB)
Edit `Scripts/download_ifs.py` `YEARS` to just `['2018']` (then `['2019']`, …),
or leave all years if you have the disk. Then:
```bash
python ../../../../Scripts/download_ifs.py     # 14 fields, resumable, verifies each .nc
```

### 3b. (2018 only, once) Generate normalisation constants
```bash
python run_gen_fcst_norm.py    # writes FCSTNorm2018.pkl ; iterates all 14 fields
```
On import, `data.py` must print **"Loading forecast normalisations"** — if you
see the `*** NOT LOADED ***` banner, fix the path before continuing.

### 3c. Write the TFRecords for the year (~8 GB)
```python
python -c "from tfrecords_generator import write_data; write_data(2018)"
```
Produces `rfe_tfrecords/2018_1.{0,1,2,3}.tfrecords` (4 rain-rate class bins).

### 3d. Upload that year to source.coop, then delete it locally
```bash
cd /scratch/notebook/cGAN_tutorial    # where the uploader lives

# fresh 1-hour STS creds in .env (shell-export syntax):
cat > .env <<'EOF'
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-west-2"
EOF

# point the uploader at the local tfrecords dir
export TFRECORDS_DIR=/data/CGAN/rfe_tfrecords
export CONSTANTS_DIR=/data/CGAN/cGAN_data
export NORM_PKL=/data/CGAN/FCSTNorm2018.pkl

uv run cgan_tfrecords_source_coop.py upload --year 2018
#  -> uploads (resumable), verifies remote size, then DELETES local files (--clean default).
#  -> if it prints "Credential budget reached" (exit code 2): refresh .env, re-run the same line.
```

### 3e. Reclaim the raw IFS disk for the year
```bash
rm -rf /data/CGAN/IFS_training/2018      # safe: TFRecords already published & verified
```

🔁 **Repeat 3a → 3e for 2019, 2020, 2021.**

> Skip 3b after 2018. Keep `RFE/` (truth) — it's tiny and `write_data` needs it.

---

## 4. One-time finalize + verify

```bash
cd /scratch/notebook/cGAN_tutorial

# push constants + FCSTNorm (no --year => uploads the small extras too)
uv run cgan_tfrecords_source_coop.py upload

# anonymous read-back: expect >=16 tfrecords (4 yrs x 4 bins) + extras present
uv run cgan_tfrecords_source_coop.py verify
```
`verify` prints object count, total GB, **per-class-bin counts**, and presence of
`FCSTNorm2018.pkl`, `cGAN_data/*`, `SHA256SUMS.txt`. All four bins must be
non-empty for every year (the weighted sampler reads `[0.4,0.3,0.2,0.1]`).

> ⚠️ **source.coop write access is gated.** The public proxy is read-only; you
> need publisher credentials provisioned for the `e4drr-project` repo before
> `upload` authenticates. `verify`/`download` work anonymously once it's public.

---

## 5. GPU node — pull to local NVMe and train

```bash
# 5.1 Pull the published store to LOCAL disk (NOT a network mount / Drive)
cd /scratch/notebook/cGAN_tutorial
uv run cgan_tfrecords_source_coop.py download --dest /scratch/CGAN
#   add --authed if the repo isn't public yet (uses .env creds)

# 5.2 Point data_paths.yaml's tfrecords_path at /scratch/CGAN/rfe_tfrecords
#     and CONSTANTS_PATH at /scratch/CGAN/cGAN_data, NORMALISATION_PATH at /scratch/CGAN

# 5.3 Train
cd SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan
python main.py --config config.yaml
#   resume after a preemption:
python main.py --config config.yaml --restart
```

> For live validation plots (`progress.pdf`) and `--evaluate`, the GPU also needs
> the **2020 raw IFS** (`val_years: [2020]`). Either keep 2020 raw out of step 3e,
> publish it under `IFS_training/2020/`, or run evaluation later on a CPU box.

### Evaluate / predict
```bash
python main.py --config config.yaml --no_train --evaluate --eval_blitz --plot_ranks
python predict.py --log_folder ../logs_RFE2_run03 --model_number 0015872 --num_samples 5
```

---

## 6. Monitor the run — Trackio (Hugging Face's W&B alternative)

`main.py` already writes `log.txt` (CSV: `training_samples, disc_loss,
disc_loss_real, disc_loss_fake, disc_loss_gp, gen_loss_total, gen_loss_disc,
gen_loss_ct`), `run_status.json`, `progress.pdf`, and `models/gen_weights-*.h5`.

Use **Trackio** (local-first, wandb-API-compatible, syncs to a free HF Space):
```bash
pip install trackio
```
Add a few lines in the checkpoint loop of `main.py` (right where it writes
`log.txt`, ~lines 207–212):
```python
import trackio
trackio.init(project="cgan-rfe2", name="run03")   # once, before the loop
...
trackio.log(loss_log, step=training_samples)       # each checkpoint
```
Dashboard is local by default; pass `space_id="org/cgan-dashboard"` to
`trackio.init` to watch the GPU run remotely and back up metrics as Parquet.

---

## Quick reference — command order

```text
# CPU prep box, per year Y in 2018..2021:
python Scripts/download_ifs.py                                   # 3a (edit YEARS)
python run_gen_fcst_norm.py                                      # 3b (2018 only)
python -c "from tfrecords_generator import write_data; write_data(Y)"   # 3c
uv run cgan_tfrecords_source_coop.py upload --year Y            # 3d (+clean)
rm -rf /data/CGAN/IFS_training/Y                                 # 3e

# once:
uv run cgan_tfrecords_source_coop.py upload                     # 4 (constants+norm)
uv run cgan_tfrecords_source_coop.py verify                     # 4

# GPU node:
uv run cgan_tfrecords_source_coop.py download --dest /scratch/CGAN
python main.py --config config.yaml [--restart]
```
