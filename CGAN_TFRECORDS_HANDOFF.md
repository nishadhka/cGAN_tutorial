# cGAN TFRecords → source.coop — Session Handoff (copyable)

One-page summary of everything built/decided this session. All paths are under
`/scratch/notebook/cGAN_tutorial/` unless noted.

---

## 1. Deliverables (files created/edited)

| File | What it is |
|---|---|
| `RFE2_cGAN_RUNBOOK.md` | End-to-end runbook (download → norm → TFRecords → source.coop → GPU). Variant A. |
| `TFRECORDS_PIPELINE_VARIANTS.md` | **Read this** — A vs B pipeline comparison + corrected sizes. |
| `prep_year.sh` | Per-year driver: 3a download → 3b norm → 3c write_data → 3d upload+clean → 3e rm raw. |
| `transfer_oxford_to_sourcecoop.sh` | Loops all 4 years through `prep_year.sh`; preflight + STS-resume. |
| `cgan_tfrecords_source_coop.py` | source.coop uploader (1h-STS, resumable, verified delete-after-upload, verify/download). |
| `make_local_tfrecords.py` | Self-contained synthetic local TFRecords generator (testing, no network). |
| `docs/pytorch_cgan_direction_oxford_ifs.md` | PyTorch EP-cGAN direction: how far achievable with Oxford NetCDF + Zarr. |
| `docs/swapping_truth_imerg_rfe2_keeping_ifs.md` | Can truth (IMERG↔RFE2) be swapped keeping IFS work? Yes via Zarr decoupling. |
| `.gitignore` | Ignores `local_tfrecords_test/`, `*.tfrecords`, venvs (keeps tracked `env/`). |

Edits also applied in **SEWAA-forecasts-RFE2**: `data.py` + `Scripts/download_ifs.py` → all 14 fields.

---

## 2. The two pipelines (verified from code 2026-06-30)

| | **A. SEWAA-forecasts-RFE2** | **B. cGAN_tutorial** (this repo) |
|---|---|---|
| Code path | `…/cGAN/dsrnngan/` | `tensorflow-dev-test/data/` |
| Cadence / truth | RFE2 daily, `HOURS=24` | IMERG 6-hourly, `HOURS=6` |
| Lead times written | **1** (`range(1,2)`) | **4** (`np.arange(30,54,6)`=30/36/42/48h) |
| Input channels | **28** (`2×14`) | **56** (`4×14`) |
| TFRecords (GZIP) | **~14–20 GB** | **~100 GB** |
| Files/year | 4 | 16 |

Both use the same 14 fields. **Both write only a lead-time SUBSET — neither is the
full 388 GB.** All-lead-times would be ~700 GB+ (patch overlap) and is not needed.

⚠️ cGAN_tutorial `write_data` currently loops `range(nsamples)` (8 dates) not
`range(len(dgc))` (all dates) — likely a dev cap; the ~100 GB assumes the all-dates fix.

---

## 3. Why TFRecords ≪ raw NetCDF (≈20× for variant A)

Data **selection**, not compression:
1. **Lead-time collapse (~30×)** — raw keeps all 29 valid-times; training uses 1 window.
2. **Spatial ~1×** — 8 patches of 128² ≈ one image-area/day.
3. **GZIP ~1.5–2×** — the only real compression (raw `.nc` already deflate'd, so cancels).

`raw 1 field·yr = 365×29×384×352×2×4B ≈ 11.4 GB` → `tfrec that field = 365×8×128²×2×4B ≈ 0.38 GB` → **~30×**.

---

## 4. TensorFlow env (verified working)

```bash
micromamba create -y -n tf215gpu -c conda-forge python=3.11 pip
micromamba run -n tf215gpu pip install "tensorflow==2.15" xarray netcdf4 numpy pyyaml s3fs
export CGAN_PY='micromamba run -n tf215gpu python'
```
✅ Tested on **real Oxford data** (byte-range pull, all 14 fields): `write_data`
emits `2021_1.{0,1,2,3}.tfrecords`, parse back as `generator_input (128,128,28)`.

---

## 5. How to run the transfer routine

```bash
# 0. fresh 1-hour source.coop STS creds in .env (shell-export syntax):
cat > .env <<'EOF'
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-west-2"
EOF

# 1. pick the pipeline variant via REPO_DIR:
export REPO_DIR=/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan   # A
# export REPO_DIR=/scratch/notebook/cGAN_tutorial/tensorflow-dev-test/data                                # B

# 2. run all years (download→tfrecords→upload→clean, disk-frugal):
export CGAN_PY='micromamba run -n tf215gpu python'
export DATA_ROOT=/data/CGAN
./transfer_oxford_to_sourcecoop.sh            # or: ./transfer_oxford_to_sourcecoop.sh 2020 2021

# 3. verify (anonymous) / pull on GPU node:
$CGAN_PY cgan_tfrecords_source_coop.py verify
$CGAN_PY cgan_tfrecords_source_coop.py download --dest /scratch/CGAN
```

Publishes to `s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/cgan_tfrecords_ou/`
(`https://source.coop/e4drr-project/forecasts/cgan_tfrecords_ou`). **Only TFRecords +
tiny constants + `FCSTNorm.pkl` are uploaded — never the raw `.nc`.**

---

## 6. DECISION + go-status (2026-06-30)

✅ **Chosen pipeline: variant A — 24 h RFE2** (1 lead time, 28 channels, ~14–20 GB).
The routine already defaults to this via `REPO_DIR`. The routine is **go-ready**;
to actually run the full creation you still need two inputs not present in this env:

1. **RFE2 truth data** on the 384×352 IFS grid — run `Scripts/download_RFEv2_ICPAC.py`
   (NOAA CPC FTP → regrid). The IFS forecasts are reachable now (byte-range verified).
2. **source.coop publisher STS creds** in `.env` (the `e4drr-project` repo; verify/
   download work anonymously once published).

Verified already this session: the variant-A `write_data` path runs end-to-end on
**real Oxford IFS** (14 fields) → real `.tfrecords` parsing back as
`generator_input (128,128,28)`. So the only gates to a production run are the two
data/credential inputs above, plus a box with ~120 GB scratch.

**Run it (once the two inputs exist):**
```bash
export CGAN_PY='micromamba run -n tf215gpu python'
export REPO_DIR=/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan
export DATA_ROOT=/data/CGAN
./transfer_oxford_to_sourcecoop.sh      # all 4 years → source.coop, disk-frugal
```

## 7. Future direction (documented, not started)

- **PyTorch EP-cGAN** with Oxford IFS: achievable as a **1× calibration** cGAN
  (super-resolution blocked — RFE2/IMERG are 0.1°, same grid as IFS). Use **Zarr**
  stores, not TFRecords. See `docs/pytorch_cgan_direction_oxford_ifs.md`.
- **Truth swap (IMERG↔RFE2):** not hot-swappable with fused TFRecords (must
  re-run `write_data`; IFS download + norm reused). True hot-swap needs decoupled
  X/Y **Zarr** stores. See `docs/swapping_truth_imerg_rfe2_keeping_ifs.md`.

## 8. Commit status
`prep_year.sh`, `cgan_tfrecords_source_coop.py`, `RFE2_cGAN_RUNBOOK.md` are
committed on branch `cgan-rfe2-data-pipeline`; the rest (transfer script, variant
docs, the two new direction docs, gitignore, this file) are **uncommitted**.

---

## 7. Git identity used
`nishadhka <nishadhka@gmail.com>` (repo-local).
