# PyTorch EP-cGAN direction with the Oxford IFS data — how far can we go?

A direction note for adopting the **PyTorch "Extreme Precipitation" cGAN**
([NMC-DAVE/CGAN_extreme_precipitation](https://github.com/NMC-DAVE/CGAN_extreme_precipitation),
Xu et al. 2026, *Wea. Forecasting*) with the **Oxford IFS NetCDF** training data
(`rain.physics.ox.ac.uk/ICPAC/training/IFS/<year>/`) and RFE2/IMERG truth.

Companion to [`tf_vs_pytorch_cgan_comparison.md`](tf_vs_pytorch_cgan_comparison.md)
(architecture/loss comparison) — this note is specifically **what is achievable
with the data we actually have**, and **where Zarr fits**.

---

## TL;DR

- ✅ **Predictors:** the Oxford IFS files give us a *superset* of the paper's
  inputs (14 fields vs their 11) — portable as-is.
- ✅ **The valuable parts of the paper port cleanly:** differentiable-CSI loss,
  3-stage training, DDP+FP16, and an **on-the-fly Zarr/NetCDF dataloader**
  (no TFRecords).
- ⛔ **The 8× super-resolution does NOT transfer with our current truth.** Both
  our pipelines use `downscaling_factor: 1` — IFS (0.1°) and RFE2/IMERG truth
  (regridded to the **same** 0.1° grid). There is no finer HR target, so we get a
  **1× bias-correction/calibration cGAN**, not the paper's 8 km→1 km super-resolution.
- ⚠️ **Cadence:** the paper uses **3-hourly** (EP defined as >20 mm/3 h); our IFS
  is **6-hourly** (cGAN_tutorial) or **24 h** (RFE2). 3-hourly isn't recoverable
  from the Oxford files.

---

## 1. What the paper needs vs what Oxford gives us

| Ingredient | PyTorch EP paper | Oxford IFS + RFE2/IMERG | Status |
|---|---|---|---|
| LR predictors | 11 IFS fields | **14 fields** (`cape,cp,mcc,sp,ssr,t2m,tciw,tclw,tcrw,tcw,tcwv,tp,u700,v700`) | ✅ superset |
| Ensemble stats | mean (+ implied) | **mean + sd** per field | ✅ richer |
| Synoptic padding channel | 768 km context tile | derivable by down-sizing a larger IFS tp patch | ✅ buildable |
| HR truth | CMA obs at **1 km** (8× finer than 8 km input) | RFE2/IMERG at **0.1° = same grid as IFS** | ⛔ not finer |
| Accumulation | **3-hourly** | 6-hourly / 24 h | ⚠️ coarser |
| Super-resolution factor | **8×** | **1×** (`downscaling_factor: 1`) | ⛔ no SR |

**Conclusion on scope:** with today's data you can build a **PyTorch EP-style
*calibration* cGAN at 1×** — same grid in, same grid out — keeping the paper's
real innovation (differentiable-CSI EP loss + 3-stage WGAN-GP). You **cannot**
reproduce the 8× downscaling unless a genuinely sub-0.1° truth is sourced
(e.g. gauge-merged radar, a ~1 km satellite/blended product). RFE2 and IMERG at
0.1° will not provide super-resolution.

> This is not a blocker for the science goal (better **extreme-precip** skill) —
> the differentiable-CSI loss helps at 1× too. It only rules out the *spatial
> super-resolution* claim of the paper.

---

## 2. Where Zarr fits — and why it is the right move for PyTorch

The PyTorch repo reads **NetCDF on the fly** (`TiggeMRMSDataset`) and the porting
roadmap explicitly says *"skip TFRecords — they were a TF perf hack."* PyTorch
`Dataset`s want random access to arrays, which is exactly what **Zarr** gives:

- **Two decoupled stores** (this is also the key to truth-swapping — see
  [`swapping_truth_imerg_rfe2_keeping_ifs.md`](swapping_truth_imerg_rfe2_keeping_ifs.md)):
  - **X store** — IFS predictors, `zarr[(date, channel, lat, lon)]`, chunked per
    `(date, channel)`. Built **once** from the Oxford NetCDF.
  - **Y store** — truth (RFE2 *or* IMERG), `zarr[(date, lat, lon)]`, same grid.
- **On-the-fly patching** in the `Dataset.__getitem__`: pick a date, crop a
  random LR patch from X and the **co-located** patch from Y. No pre-baked,
  truth-fused TFRecords; `WeightedRandomSampler` handles EP oversampling.
- **Cloud-native:** the Zarr can live on **source.coop / S3** and stream lazily
  (`xr.open_zarr` over `s3fs`), or be staged to local NVMe — the same access
  pattern the size doc recommends (see
  [`../pytorch-cgan/CGAN_STORE_TENSOR_SIZE_ESTIMATION.md`](../pytorch-cgan/CGAN_STORE_TENSOR_SIZE_ESTIMATION.md)).

**Zarr verdict:** yes — for the PyTorch direction, store IFS (and truth) as Zarr,
not TFRecords. It is the natural fit for PyTorch dataloaders, enables truth-swap,
and streams from source.coop. (TFRecords remain the right choice only for the
*TensorFlow* pipeline, which is what the `cgan_tfrecords_source_coop.py` routine
serves.)

---

## 3. How far we can go — staged roadmap

| Phase | Deliverable | Achievable with Oxford data? |
|---|---|---|
| **P1. IFS → Zarr (X store)** | Convert the 14-field Oxford NetCDF to a chunked Zarr (per date/channel), 6-hourly. | ✅ now (byte-range read already proven) |
| **P2. Truth → Zarr (Y store)** | RFE2 (24 h) or IMERG (6 h) regridded to the 0.1° IFS grid as Zarr. | ✅ once truth is downloaded/regridded |
| **P3. ICPAC dataset adapter** | Port `TiggeMRMSDataset` → reads X+Y Zarr, on-the-fly patches, `WeightedRandomSampler`. | ✅ (~code, no new data) |
| **P4. 1× calibration cGAN** | Train EP-cGAN (diff-CSI loss, 3-stage) at same-grid 1×. | ✅ the realistic target |
| **P5. 8× super-resolution** | Needs a sub-0.1° HR truth (radar/gauge/blended ~1 km). | ⛔ blocked until finer truth sourced |

**Bottom line:** with the Oxford IFS NetCDF + RFE2/IMERG we can reach **P4 — a
PyTorch EP-style extreme-precipitation *calibration* cGAN** that keeps the paper's
differentiable-CSI advantage, fed from **Zarr** stores streamed from source.coop.
**P5 (true super-resolution) is out of reach** with 0.1° truth and would require a
new high-resolution observation source. Accumulation stays 6-hourly/24 h, so the
strict ">20 mm/3 h" EP definition would be reframed to our cadence.

---

## 4. Immediate next step if we pursue PyTorch

Build the **IFS→Zarr X store** from the Oxford NetCDF (P1) — reusable by both the
PyTorch dataloader and any future truth source — using the same byte-range read
and source.coop publishing pattern already validated for the TFRecords routine.
That single artifact unblocks P2–P4 and is independent of the RFE2-vs-IMERG truth
choice.
