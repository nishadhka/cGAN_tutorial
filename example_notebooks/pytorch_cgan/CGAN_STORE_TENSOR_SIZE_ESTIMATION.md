# cGAN Icechunk Store & Training-Tensor Size Estimation

**East Africa PyTorch EP-cGAN (Xu et al. 2026) — ECMWF IFS ENS inputs**
Scope: scale the completed **3-month, 5-surface-channel, ensemble** Coiled/Dask
Icechunk build up to (a) the full **11-channel** PyTorch EP variable set with
pressure levels, (b) **12-month** coverage over **2024-03 → 2026-05**, and (c) the
GPU training tensors. Includes a reflection on the heavier TensorFlow-cGAN
(~14 channel) variable set.

_Last updated 2026-05-29. Storage numbers in the "Measured" section are real
`gsutil`-equivalent byte counts off `gs://cgan-east-africa/`; everything else is
linearly extrapolated from them._

---

## 0. Glossary (terms used throughout)

| Term | Meaning |
|---|---|
| **LR — Low Resolution** | The **coarse predictor input (X)**: the ECMWF IFS fields in our Icechunk store, 0.25° grid (161×133). This is what the cGAN *conditions on*. Patched at **32×32** in the EP method. |
| **HR — High Resolution** | The **fine-scale target (Y)**: the super-resolved precipitation field the generator must produce, ~×8 finer (256×256 patch ≈ 0.03°). Sourced from IMERG/CHIRPS observations — **a separate store, not yet built** (see §4). |
| **×8 super-resolution** | The generator upsamples LR→HR by a factor of 8 in each dimension (32×32 → 256×256). |
| **Channel** | One input variable-field (e.g. `tp`, or `u@700hPa`). The EP method uses **11 channels**. |
| **Member / ensemble** | One of the 51 ECMWF ENS realisations (`control` + `ens_01..50`). The "ensemble dimension" is the ×51 storage multiplier. |
| **Patch** | A cropped LR tile (+ its co-located HR target) fed to the GPU. Training samples ~220k patches/epoch on the fly — it does **not** materialise the whole store. |
| **cGAN** | conditional GAN. **EP** = the PyTorch "Extreme Precipitation" method (Xu et al. 2026) we are porting; **TF** = the legacy TensorFlow tutorial cGAN. |

> LR/HR is the core of super-resolution: **LR in (coarse IFS) → HR out (fine
> precip)**. The Icechunk stores in this document are the **LR side only**.

---

## 1. Measured baseline (what is already built)

Three ensemble MAM stores, **5 surface channels** (`tp, pw, msl, sp, cp_proxy`),
51 members, 9 lead times (6–30 h), grid **161 × 133** (0.25°, 20–53°E / −15–25°N),
`float32`, Zstd-compressed, chunk = `(1 init_date, 1 member, 9 lead, 161, 133)`.

| Store | Dates filled | Compressed size | Per date |
|---|---|---|---|
| `pytorch_cgan_ifs_mam2026_ens` | 38 / 92 | **2.16 GB** | 56.9 MB |
| `pytorch_cgan_ifs_mam2025_ens` | 92 / 92 | **5.13 GB** | 55.7 MB |
| `pytorch_cgan_ifs_mam2024_ens` | 92 / 92 | **5.15 GB** | 56.0 MB |

> Note: `cp_proxy` is all-NaN in 2024/2025 and ~zero in 2026, so it compresses to
> ≈0 bytes. The measured ~56 MB/date is therefore effectively **4 real channels**.

### Derived unit costs (the numbers everything below is built from)

| Unit | Raw (`float32`) | Compressed (Zstd, measured) |
|---|---|---|
| 1 grid field (9 lead × 161 × 133) | 0.771 MB | ~0.28 MB |
| **1 channel · 1 date · 51 members** | **39.3 MB** | **~14 MB** |
| 1 channel · 1 date · **1 member** (control) | 0.771 MB | ~0.28 MB |
| 1 channel · 92-date MAM season · 51 mem | 3.62 GB | ~1.28 GB |
| 1 channel · 365-day year · 51 mem | 14.4 GB | ~5.1 GB |
| 1 channel · 822 days (2024-03→2026-05) · 51 mem | 32.3 GB | ~11.4 GB |

**Compression ratio ≈ 2.8×** on real meteorological fields (`tp` sparser/better,
winds `u/v` smoother/slightly worse). Storage scales **linearly** with
`channels × dates × members` because every channel is the same shape.

---

## 2. Channel sets

| Set | # channels | Composition |
|---|---|---|
| **Surface pilot (built)** | 5 (4 effective) | `tp, pw, msl, sp, cp_proxy` |
| **PyTorch EP full (target)** | **11** | 5 surface + 5 pressure-level (`u, v, ub, vb, gh`) + 1 derived |
| **TF cGAN reflection** | ~14 | EP set + extra IFS fields (`cape, tcwv, t2m, …`) |

The 5 pressure-level channels are **single-level fields** (each a specific
var@level, e.g. `u700, v700, u925→ub, v925→vb, gh`), not a multi-level cube — so
the full EP set is **+6 channels over the surface pilot (≈ 2.2× the bytes)**.
They are currently disabled pending the GIK per-level-keys parquet fix
(`GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`).

> If a fuller multi-level pressure cube were ever wanted (e.g. `u,v,gh,t,q` × 7
> levels = 35 channels), multiply the per-channel unit cost by 35 — but the EP
> method does **not** require this; it uses 11 curated channels.

---

## 3. Icechunk store size — projection

### 3a. Ensemble (51 members) — compressed GB

| Coverage (init dates) | 5 ch (surface) | **11 ch (EP)** | 14 ch (TF) |
|---|---|---|---|
| 3-mo MAM season (92) | ~5.1 *(measured)* | **~14.1** | ~17.9 |
| 1 full year (365) | ~20 | **~56** | ~71 |
| **2024-03→2026-05, 12-mo (822)** | ~46 | **~126** | ~160 |

### 3b. Control-only (1 member) — compressed GB

Use for fast deterministic experiments; ÷51 of the ensemble.

| Coverage | 5 ch | **11 ch (EP)** | 14 ch (TF) |
|---|---|---|---|
| 3-mo MAM (92) | ~0.13 | **~0.28** | ~0.35 |
| 1 year (365) | ~0.5 | **~1.1** | ~1.4 |
| **822 days** | ~1.1 | **~2.5** | ~3.1 |

### 3c. Raw (uncompressed) — what matters if you ever load it whole

Multiply compressed by ~2.8. The headline cases:

| Case | Compressed | Raw |
|---|---|---|
| **EP 11 ch · 822 d · ensemble** (the full target store) | **~126 GB** | **~355 GB** |
| EP 11 ch · 822 d · control | ~2.5 GB | ~7 GB |
| TF 14 ch · 822 d · ensemble | ~160 GB | ~450 GB |

**Takeaway:** the full operational target (11-channel, 27-month, 51-member) is a
**~125 GB GCS Icechunk store**. The ensemble dimension is the dominant cost
(×51); a control-only version of the same is only **~2.5 GB**.

---

## 4. The high-resolution target store (not yet built — required for training)

The Icechunk stores above are the **coarse predictor (X)** only. cGAN training
also needs the **HR observation target (Y)** (IMERG / CHIRPS), regridded to the
×8 super-resolution grid (~1288 × 1064), **deterministic (no ensemble)**.

| HR target, 1 channel | Per day (matched valid times) | 822 days raw | Compressed |
|---|---|---|---|
| ~1288×1064 float32 | ~49 MB | **~40 GB** | **~12–15 GB** |

Budget **~15 GB** extra for the precipitation target store. This is a separate
ingest (IMERG half-hourly → accumulated to the 3-h buckets / valid times).

---

## 5. Training tensors on the GPU

### 5.1 You do **not** load the store into VRAM
The full 11-channel ensemble store is ~355 GB raw — far beyond any single GPU
(L4 = 22 GB, A100 = 80 GB). Training **streams patches** from Icechunk on the fly
(`WeightedRandomSampler`, ~220k patches/epoch in the paper). The store stays on
GCS; only the current mini-batch is materialised as a tensor.

### 5.2 Per-batch tensor footprint (tiny)
EP-cGAN patches: LR **32×32**, HR **256×256** (×8), batch `B`, `C=11` input ch,
ensemble `E=10` in the content loss.

| Tensor | Shape | `float32` @ B=64 |
|---|---|---|
| Conditioning X | B × 11 × 32 × 32 | 2.95 MB |
| Noise + constants | B × ~2 × 32 × 32 | <1 MB |
| HR target Y | B × 1 × 256 × 256 | 16.8 MB |
| Generator ensemble out | B × 10 × 256 × 256 | 168 MB |
| **Raw data tensors / step** | | **< 200 MB** |

The data tensors are **trivial**. VRAM is dominated by **model activations**.

### 5.3 What actually fills VRAM
- **Params (G + D):** ~10–60 M. With Adam (param + grad + 2 moments = 4×) at
  fp32 → **~0.2–1.0 GB**.
- **Activations:** the binding cost. A residual-UNet G producing 256×256 at base
  width 64, for B=64, peaks at **~10–20+ GB**. This is what sets the batch size.

### 5.4 Practical batch sizing

| GPU | VRAM | Workable batch (bf16) | Strategy |
|---|---|---|---|
| **L4 (this box)** | 22 GB | **8–16** | grad-accumulation → effective 64–256; `precision=16` |
| V100 | 32 GB | 16–32 | DDP across 4 → global 64–128 |
| A100 | 80 GB | 64 (paper's per-GPU) | paper used 4×A100 = global 256 |

**On the single L4:** run `B≈8–16` with **gradient accumulation** to reach the
paper's effective global batch, keep **`precision=16` (bf16/fp16)**, and stream
from Icechunk with `num_workers≥8`. This is the only configuration that fits.

### 5.5 I/O, not VRAM, is the likely bottleneck on one GPU
~220k patches/epoch read from GCS Icechunk. Mitigate with: a local SSD cache of
the (init_date, member) chunks actually sampled, generous `num_workers`,
`pin_memory=True`, and prefetch. Per-epoch the dataloader touches a small subset
of the ~126 GB store, not all of it.

---

## 6. Running on a Coiled GPU VM — data locality & avoiding transfer

Example launch under discussion:

```bash
coiled notebook start --name cgan-env-test \
  --vm-type g2-standard-8 \
  --software cgan-tf-torch-v1 \
  --workspace=gcp-sewaa-nka \
  --region us-west4 \
  --disk-size 100
```

**`g2-standard-8`** = 1 × NVIDIA **L4 (24 GB GPU)**, 8 vCPU, **32 GB system RAM**,
boot disk **100 GB**. That is a single-GPU box — exactly the L4 the §5 batch-size
advice targets.

### 6.1 ⚠️ Region mismatch is the real cost driver (fix this first)
The bucket `gs://cgan-east-africa/` is **EU multi-region**. The VM above is in
**`us-west4` (US)**. So every read is **EU → US cross-continent egress**:

- **Billed** at ~$0.12/GB. Reading the 126 GB store once ≈ **$15**; re-read each
  epoch with no cache → tens of $ per training run.
- **Slow**: trans-Atlantic latency on ~220k random small-object GETs/epoch will
  starve the GPU.

**Two clean fixes:**
1. **Put the GPU VM in an EU region** (e.g. `--region europe-west4`, matching the
   bucket and the original Coiled fill cluster). GCS reads from a VM **colocated
   with the bucket's location are free** and low-latency → no staging needed.
2. **Or stage to a local disk once** (below) — works from any region, pay the
   one-time copy, then read locally for free.

### 6.2 The 100 GB disk does not fit 450 GB of tensors
You cannot "pre-fill the SSD with the 450 GB tensors" on `--disk-size 100`:

| What you'd put on disk | Size | Fits 100 GB? |
|---|---|---|
| Raw materialised tensors, 14 ch ensemble | ~450 GB | ❌ (needs ≥500 GB) |
| Raw, 11-ch EP ensemble | ~355 GB | ❌ |
| **Compressed Icechunk**, 11-ch EP ensemble | **~126 GB** | ❌ (needs ≥150 GB) |
| Compressed Icechunk, **control-only** 11-ch | **~2.5 GB** | ✅ trivially |
| HR target store | ~15 GB | ✅ |

**Do not pre-materialise raw patch tensors to disk.** Keep the data as the
**compressed Icechunk store on disk** and let the dataloader decompress on read
(Zstd is ~GB/s) — same code as GCS, just `icechunk.local_filesystem_storage(path=…)`
instead of `gcs_storage(…)`. This is 2.8× smaller and far more flexible than
frozen `.npy`/`.pt` dumps. Also note **32 GB RAM < 126 GB**, so you can't hold the
ensemble store in memory either — disk staging is the route, not a RAM cache.

### 6.3 Three data-access patterns

| Pattern | How | Egress | Random-read speed | Persistence |
|---|---|---|---|---|
| **A. Stream from GCS** | dataloader opens `gcs_storage(...)` | per-epoch (free if VM colocated in EU) | network-bound | n/a |
| **B. Stage to local disk once** | one-time `gcs→disk` copy, then `local_filesystem_storage` | one-time only | **NVMe/SSD speed** | see §6.4 |
| **C. Hybrid chunk cache** | stream from GCS but cache touched chunks on local SSD | first epoch only | SSD after warm-up | per-session |

For the **random patch-sampling** access pattern (220k random reads/epoch),
**local SSD (B/C) is materially faster** than repeated GCS GETs regardless of
region — it removes per-object request latency.

### 6.4 "Pre-filled SSD" — the persistence caveat with Coiled notebooks
A Coiled **notebook VM and its boot disk are ephemeral**: `coiled notebook stop`
**deletes the VM and the `--disk-size` disk**, so anything staged there is gone
next session. To genuinely "attach a pre-filled SSD that persists" you have one
of:

1. **Bigger boot disk + re-stage per session** — set `--disk-size 200`, then on
   each session start run a one-time sync (minutes; free if EU-colocated):
   ```bash
   # one-time per session, GCS -> local
   python - <<'PY'
   import icechunk, xarray as xr
   src = icechunk.Repository.open(icechunk.gcs_storage(
       bucket="cgan-east-africa", prefix="pytorch_cgan_ifs_mam2025_ens", from_env=True))
   # ... xr.open_zarr(...).to_zarr(local icechunk session) , or rclone/gsutil rsync the objects
   PY
   ```
   Simplest and Coiled-native. The copy is cheap if the VM is EU-colocated.
2. **A standalone GCP Persistent Disk (pd-ssd) managed outside Coiled** — create
   a `pd-ssd` once, populate it, snapshot it, and attach it to a plain GCE VM you
   control. Coiled's notebook CLI has **no first-class flag to attach an existing
   external PD**, so full persistent-disk control means running the GPU VM
   directly on GCE rather than via `coiled notebook`. The disk survives
   stop/start and re-attaches instantly — this is the closest match to
   "pre-filled SSD, no transfer."
3. **GCP Local SSD (NVMe scratch)** — fastest (~GB/s, 375 GB units) but **wiped
   on every stop**, so it's a per-session cache (pattern C), never durable.

### 6.5 Recommended setup for this box
- **Move the VM to `europe-west4`** (colocate with the EU bucket) → GCS reads
  become free + low-latency; you may not need staging at all.
- **For the pilot, use the control-only store (~2.5 GB)** — it fits the 100 GB
  boot disk with room to spare; stage it once with pattern B and read locally.
- **For the full ensemble (~126 GB)**, either keep streaming from a colocated EU
  VM (pattern A), or provision a **standalone GCE VM + 200 GB pd-ssd** (option 2)
  and copy the compressed Icechunk store onto it once.
- Keep `precision=16`, batch 8–16 + grad-accum, `num_workers≥8`, `pin_memory=True`.

> Bottom line on your question: **yes, a local SSD avoids repeated transfer** —
> but (a) on Coiled notebooks the disk is ephemeral so you re-stage per session
> or use a standalone GCE PD; (b) 450 GB won't fit on a 100 GB disk — stage the
> **126 GB compressed store** (or the **2.5 GB control store**), not raw tensors;
> and (c) the cheaper first move is simply **putting the VM in the bucket's EU
> region** so GCS reads are free anyway.

---

## 7. Bottom-line numbers

| Question | Answer |
|---|---|
| Current store scale (per MAM season, 5 ch, ensemble) | **~5.1 GB** (measured) |
| 3-month, **11-channel** EP store (pressure levels added), ensemble | **~14 GB** |
| **12-month, 2024-03→2026-05, 11-channel EP**, ensemble | **~126 GB compressed** (~355 GB raw) |
| Same, **control-only** | **~2.5 GB** |
| TF-cGAN reflection (14 ch, full window, ensemble) | **~160 GB** |
| HR precip target store (needed, not built) | **~15 GB** |
| Per-training-step GPU data tensor | **< 200 MB** (negligible) |
| Binding GPU constraint | **activations** → batch 8–16 on L4 w/ grad-accum + bf16 |
| ⚠️ Bucket region | **EU multi-region** — a `us-west4` VM pays EU→US egress; put the GPU VM in **`europe-west4`** (free, colocated) |
| Local-SSD staging | stage the **126 GB compressed store** (or 2.5 GB control), **not** 450 GB raw; 100 GB boot disk only fits control-only |

### Recommended path
1. **Finish the GIK per-level-keys fix** → enable the 5 pressure-level channels
   (5 → 11 ch). Adds ~+9 GB per MAM season ensemble.
2. **Build control-only first** for the full 2024-03→2026-05 window: only
   **~2.5 GB**, trains fast, validates the pipeline end-to-end.
3. **Scale to ensemble** (~126 GB) once the control pilot's skill is confirmed —
   the ×51 cost is only justified if ensemble spread improves CRPS/FSS.
4. **Ingest the HR target store (~15 GB)** in parallel — it is the missing half
   of the training pair.
5. On the L4: **bf16 + batch 8–16 + gradient accumulation + chunk cache.**
   Move to multi-A100 only for the full 8-lead-time production run.
6. **Colocate the GPU VM with the bucket (`europe-west4`)** so GCS reads are free
   and fast; only stage to a local/standalone SSD if you stay on a US VM or want
   maximum random-read throughput (see §6). Stage the **compressed Icechunk
   store**, never raw tensors.

---

## 8. Detachable pd-ssd plan — runbook & assessment

**The plan:** create a 500 GB pd-ssd, populate it from a cheap CPU VM, then
detach and re-attach it to the GPU VM for multi-day training (so the GPU never
sits idle during the slow download).

**Verdict: architecturally sound, 500 GB is well-sized, pd-ssd is the correct
disk type** (persistent + detachable; Local NVMe SSD is faster but wiped on stop,
so it can't be detached/re-attached). Two blockers and several gotchas below.

### 8.1 Sizing check (500 GB pd-ssd)
| Contents to stage | Size | Fits 500 GB |
|---|---|---|
| Compressed Icechunk, 11-ch EP ensemble (LR) | ~126 GB | ✅ |
| + HR target store (Y) | ~15 GB | ✅ (141 GB total) |
| Raw tensors instead (not recommended) | ~355 GB + 15 | ✅ but wasteful |
**Stage the compressed Icechunk store (~141 GB total), leaving ~360 GB headroom.**

### 8.2 Hard requirements (silent failure if missed)
1. **Same zone** — a pd-ssd is *zonal*; it attaches only to a VM in the
   **identical zone**. Pin CPU VM + disk + GPU VM to one zone (e.g.
   `europe-west4-a`).
2. **`--no-auto-delete`** when attaching, so deleting/recreating a VM never
   deletes the data disk.
3. **Clean `umount` before detach** — single-writer read-write disk; detaching
   while mounted risks filesystem corruption.
4. **Work in `europe-west4`** (bucket is EU multi-region): EU→EU download is
   free; staging to a `us-west4` disk costs ~$15 EU→US egress per 126 GB.
5. **Disk and GPU VM throughput** — pd-ssd IOPS scale with size; 500 GB gives
   ample random-read IOPS for the ~220k patches/epoch sampling pattern (far
   better than cross-continent GCS).

### 8.3 Coiled-specific friction (the real risk for multi-day runs)
- A `coiled notebook` VM is a GCE instance in `gcp-sewaa-nka`; you *can*
  `gcloud compute instances attach-disk` to it out-of-band, **but Coiled does not
  track that disk** — on stop/idle Coiled tears the VM down and will not
  re-attach it. Expect to re-attach manually on every restart.
- **Coiled notebooks idle-timeout / auto-shutdown.** A multi-day run can be
  killed by a disconnect or idle window. Mitigate: run training under
  `tmux`/`nohup`, disable idle shutdown — **or, preferred for multi-day, use a
  plain GCE GPU VM you control** (GCP Deep Learning VM image or your
  `cgan-tf-torch-v1` env) so the lifecycle is yours. Use Coiled only if you
  specifically want its prebuilt CUDA env, and accept the babysitting.

### 8.4 Sketch of the workflow (europe-west4-a, all one zone)
```bash
# 1. Create the data disk
gcloud compute disks create cgan-data --type=pd-ssd --size=500GB --zone=europe-west4-a

# 2. Attach to a cheap CPU VM, mkfs, mount, populate from GCS (EU->EU = free)
gcloud compute instances attach-disk cpu-loader --disk=cgan-data \
    --zone=europe-west4-a --device-name=cgan-data
#   ... mkfs.ext4 /dev/disk/by-id/google-cgan-data ; mount ; copy compressed
#       Icechunk store + HR target onto it ; umount

# 3. Detach, then attach to the GPU VM (no-auto-delete keeps the disk safe)
gcloud compute instances detach-disk cpu-loader --disk=cgan-data --zone=europe-west4-a
gcloud compute instances attach-disk <gpu-vm> --disk=cgan-data \
    --zone=europe-west4-a --device-name=cgan-data --no-auto-delete
#   ... mount read-only/rw on the GPU VM ; point the dataloader at the local
#       path via icechunk.local_filesystem_storage(path=...) ; train in tmux
```
The disk survives stop/start of either VM and re-attaches instantly — this is
the closest thing to "pre-filled SSD, zero repeated transfer."

### 8.5 ⛔ Two blockers before tensors are "good to go" (independent of disk)
1. **The HR target store (Y) does not exist yet.** A super-resolution cGAN needs
   matched (LR input, HR target) pairs. You must ingest IMERG/CHIRPS to the ×8
   grid (~15 GB) and stage it on the **same disk**. Without it there is nothing
   for the generator to learn against. **This is the missing half of training.**
2. **Pressure-level channels are still disabled** (GIK per-level-keys fix). The
   current store is 5 surface channels only; the 11-ch EP set requires that
   ingest to complete first.

**So:** disk plan ✅, sizing ✅, zone/region discipline required, Coiled
lifecycle is the operational risk — but **you cannot start training until the HR
target store is built** (and the 11-ch ingest if full EP is wanted). The LR
predictor tensors themselves stream fine from a mounted pd-ssd.
