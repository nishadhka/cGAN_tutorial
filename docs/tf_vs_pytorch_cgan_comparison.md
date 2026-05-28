# TensorFlow cGAN tutorial vs PyTorch EP-cGAN — technical comparison

This document compares the two conditional-GAN codebases that sit side-by-side
on this machine:

- **TF version** — `/scratch/notebook/cGAN_tutorial/` (this repo). IFS → ICPAC
  downscaling tutorial, TensorFlow 2.x with the legacy-Keras flag.
- **PyTorch EP version** — `/scratch/notebook/CGAN_extreme_precipitation/`.
  Codebase from Xu et al. (2026, *Wea. Forecasting* 41, 381–401, DOI
  10.1175/WAF-D-24-0199.1) — "Postprocessing for 24-Hour Advanced Forecasting
  of Extreme Precipitation Using Deep Learning Generative Models." Adapted from
  Rasp's `nwp-downscale` framework.

Both are WGAN-GP conditional GANs for precipitation, but they differ sharply
in framework, scale, ergonomics, and how they treat extreme precipitation
(EP).

## TL;DR

| Axis | TF tutorial | PyTorch EP |
| --- | --- | --- |
| Framework | TensorFlow 2.x + legacy Keras (`TF_USE_LEGACY_KERAS=1`) | PyTorch 2.x + PyTorch Lightning 1.9 |
| Distributed training | single-GPU centric | multi-GPU DDP + FP16 mixed precision |
| Generator | Residual UNet, bilinear upsample, softplus output | Residual blocks, ConvTranspose ×8, sigmoid/ReLU output |
| Discriminator | Single-scale, separate LR and HR paths concatenated | Multi-scale (HR downsampled + LR resblocks merged), optional spectral norm |
| GAN loss | WGAN-GP, λ_gp=10 | WGAN-GP, λ_gp=10 |
| n_critic | 2 disc steps / 1 gen step | 5 disc steps / 1 gen step |
| Optimizer | Adam β=(0.0, 0.999); G lr=1e-2, D lr=1e-4 | Adam β=(0.0, 0.9); G lr=5e-5, D lr=1e-4 |
| Content loss | CRPS / CRPS_phys / ensmeanMSE (weight=1000) | Weighted L1 (ens-mean LR + HR) + differentiable CSI (paper's novelty) |
| Inputs | 14 IFS fields + constants, 6-h accum | 11 IFS fields + 768-km synoptic padding channel, 3-h accum |
| Patching | TFRecords pre-built, frequency-bin oversampling | On-the-fly 32→256 patches (8× super-resolution), WeightedRandomSampler |
| Batch size | 2 (single GPU) | 256 global (64/GPU on 4-GPU rig) |
| Eval | CRPS, RAPSD, FSS, ROC, thresholded rank histograms | FSS, CSI, BIAS, RAPSD, ROC, reliability, Brier, CRPS (via xskillscore) |

## 1. Generator architecture

### TF tutorial (`model/models.py:10–180`, `model/blocks.py:64–118`)
- Residual UNet-style.
- Input stays at LR; constants downscaled separately.
- Bilinear upsampling (configurable steps via `downscaling_steps`, typically
  5×2 = 10×).
- Optional BatchNorm (`norm="batch"`), usually off.
- Noise concatenated at the bottleneck as `noise_channels` extra channels.
- Softplus output guarantees non-negative precipitation.

### PyTorch EP (`codes/src/models_05_loglrts_val.py:35–92`, lines 639–703 for
the residual blocks)
- Corrector → super-resolver, two-branch design.
- Initial conv 64ch → two residual blocks (128 → 256 ch).
- Noise (single channel, ~N(0,1) weighted by 0.2) concatenated after the
  initial conv.
- Three 256-ch residual blocks, then bifurcates:
  - LR branch: conv with constrained sigmoid → `g(x, z)` (LR bias-corrected
    forecast at 8 km).
  - HR branch: four residual blocks (256 → 32 ch) + three bilinear ×2
    upsampling stages → `G(x, z)` (1 km HR forecast).
- Kaiming init throughout; optional spectral norm via flag.

The PyTorch architecture is more *physically motivated* (explicit LR/HR
intermediate outputs that both get loss terms) while the TF one is more of a
straight UNet with noise injected at the bottleneck.

## 2. Discriminator

### TF tutorial (`model/models.py:183–310`)
- Single-scale patch discriminator.
- Two paths: LR (condition + constants downsampled) and HR (real or fake
  precip). Concatenated before GlobalAvgPool → Dense(64, ReLU) → Dense(1).
- No spectral norm.

### PyTorch EP (`codes/src/models_05_loglrts_val.py:95–217`)
- Dual pathway:
  - HR path: conv → 3 residual blocks with stride-2 downsampling.
  - LR path: same prep but stride-1 residual blocks (keeps spatial res).
- Merged at 256/512 channels, GlobalAvgPool → Dense(256, LeakyReLU) →
  Dense(1, linear).
- Optional spectral norm on Conv2d/Linear.
- `StephanDiscriminator2` adds a true multi-scale branch.

## 3. Loss function

Both are WGAN-GP at heart. The key differences are in the *content loss* that
augments the generator's adversarial objective.

### TF tutorial — CRPS-family content loss
`model/gan.py:82–204`. Selectable via config `CL_type`:
- `CRPS` — ensemble Continuous Ranked Probability Score.
- `CRPS_phys` — CRPS with physical (mm) units.
- `ensmeanMSE` / `ensmeanMSE_phys` — MSE on ensemble mean.

Weight: `content_loss_weight=1000` (`config/config.yaml:31`). Ensemble size:
10 (`config/config.yaml:29`).

### PyTorch EP — three-stage training with differentiable CSI (paper's novelty)
Three sequential training stages (paper §2.d):

1. **Stage 1 — corrector pretraining (noise=0).** Loss = precipitation-weighted
   L1 on LR forecast + differentiable FSS (Roberts–Lean fractions skill
   score with sigmoid relaxation, threshold ≈ 0.75 after log transform).

2. **Stage 2 — generator pretraining (noise=0).** Loss = precip-weighted L1
   on LR + HR forecasts + **differentiable CSI** on both. CSI is made
   differentiable via sigmoid surrogates for the comparison operators
   (Larraondo et al. 2020):

   ```
   hits          = σ(c·(fcst−thr)) ⊙ σ(c·(obs−thr))
   misses        = σ(c·(obs−thr))  ⊙ σ(−c·(fcst−thr))
   false_alarms  = σ(−c·(obs−thr)) ⊙ σ(c·(fcst−thr))
   CSI           = hits / (hits + misses + false_alarms)        with c=10
   ```

3. **Stage 3 — full WGAN-GP with noise z~N(0,1) weighted by 0.2.** Loss is
   the standard WGAN-GP terms plus the precip-weighted L1 and differentiable
   CSI applied to the *ensemble mean* across 6 noise samples for both LR and
   HR outputs.

This is the paper's main contribution and what makes "CGAN" outperform
"CGAN-pre" on EP events (paper Fig. 5, ~20% CSI gain at 50 mm/3h threshold).

## 4. Training loop

| Item | TF tutorial | PyTorch EP |
| --- | --- | --- |
| File | `model/gan.py:207–322` | `mec-step3-train-generator.py`, `mec-train-final-step-ts.py`, `models_05_loglrts_val.py:1335–1378` |
| Lightning | none | PyTorch Lightning 1.9 `pl.Trainer(accelerator='ddp', precision=16, gpus=[0,1,2,3])` |
| n_critic | 2 | 5 |
| Adam betas | (0.0, 0.999) | (0.0, 0.9) |
| G lr / D lr | 1e-2 / 1e-4 | 5e-5 / 1e-4 |
| Batch size | 2 (with content loss) | 256 global, 64/GPU |
| Ensemble size for content loss | 10 | 6 (stage 3), 10 (val) |
| EMA on G weights | no | no |
| Mixed precision | not used | FP16 |
| Gradient penalty | custom `GradientPenalty` Keras layer | inline `torch.autograd.grad` in `training_step` |

The asymmetric G learning rate in the TF repo (10× the D rate) is unusual
for WGAN-GP and prone to instability on heavy-tail distributions. The PyTorch
repo's recipe (5e-5 vs 1e-4, n_critic=5) is closer to the standard.

## 5. Inputs / targets

### TF tutorial — IFS → ICPAC East Africa downscaling
- Predictors (LR): 14 IFS fields — `cape, cp, mcc, sp, ssr, t2m, tciw,
  tclw, tcrw, tcw, tcwv, tp, u700, v700` (`data/data.py:20–23`).
- Plus high-resolution constant fields (topography, land–sea mask).
- Target: precipitation, single channel.
- Accumulation: 6-hourly.
- Log transform: optional `log10(1 + precip)` (`data/data.py:40–44`).
- Spatial: configurable downscaling factor, typically 10× total.

### PyTorch EP — IFS → CMA observation, central/eastern China
- Predictors (LR): 11 IFS fields + 1 synoptic padding channel where the
  full-precip patch of 768 km × 768 km is downsized to the same 256 px
  centred on the HR target (paper §2.b).
- Channels typically include: `tp, lsp, cp, cape, tcwv, t2m, u, v, sp,
  msl, ...`.
- Accumulation: 3-hourly (lets you target EP defined as >20 mm / 3 h).
- Log transform applied via `log_trans(x, eps=0.01)` in the dataloader,
  then min-max scaled to [0, 1].
- Spatial: 32×32 LR (8 km) → 256×256 HR (1 km), 8× super-resolution.

The 3-hourly accumulation + 768-km synoptic context channel are paper-
specific choices that directly improve EP forecast accuracy. The TF tutorial
uses 6-hourly accumulation which dilutes peak intensities.

## 6. Data pipeline

| | TF tutorial | PyTorch EP |
| --- | --- | --- |
| File format | TFRecords (`data/tfrecords_generator.py`) | netCDF read on-the-fly by `TiggeMRMSDataset` |
| Normalization | per-variable min-max across train set | per-variable min-max from pickled stats |
| Train/val/test | year-based via config | first 3 days of each month → val, rest → train, 2022 → test |
| Extreme handling | frequency-bin oversampling weights `[0.4, 0.3, 0.2, 0.1]` | `WeightedRandomSampler` over patches |
| Patches | full LR & HR images, not patched | sliding-window 32 px LR patches with stride |
| Augmentation | none | none |

## 7. Evaluation metrics

Both repos cover the standard meteorological metrics. The TF tutorial has
slightly richer implementations (RAPSD with pySTEPS handling, thresholded
rank histograms); the PyTorch repo leans on `xskillscore` and adds
classification ROC + Brier + CSI computations:

- TF: `evaluation/crps.py`, `evaluation/fss.py`, `evaluation/rapsd.py`,
  `evaluation/run_roc.py`, `evaluation/thresholded_ranks.py`.
- PyTorch: `codes/src/evaluation_o.py` (FSS with windows 4/10/20 px, ROC at
  threshold 0.1 mm, classification metrics via sklearn).

Both metric stacks are numpy-based and framework-agnostic, so they can be
shared between the two repos with minimal porting.

## 8. Runtime requirements

| | TF tutorial | PyTorch EP |
| --- | --- | --- |
| Python | 3.10 | 3.10 |
| Framework | tensorflow 2.x, `TF_USE_LEGACY_KERAS=1` | torch 2.1, pytorch_lightning 1.9.x (NOT 2.x — uses removed `pytorch_lightning.core.lightning.LightningModule` and `pytorch_lightning.plugins.DDPPlugin` imports) |
| GPU | single GPU typical | DDP across 4 GPUs hard-coded (`CUDA_VISIBLE_DEVICES = "0,1,2,3"`) |
| Mixed precision | not used | `precision=16` in `pl.Trainer` |
| Extra deps | xarray, netCDF4, wandb | xarray, xskillscore, dask, scikit-image, scikit-learn, catalyst (only for `DistributedSamplerWrapper`) |

Common gotchas when bringing up the PyTorch env:

1. **PyTorch Lightning must be 1.6 ≤ pl < 2.0.** 2.x removes the API the
   training scripts use.
2. **catalyst** is only needed for `DistributedSamplerWrapper`. On Python
   3.10 older catalyst versions break with `module 'collections' has no
   attribute 'MutableMapping'`. Either upgrade (`pip install "catalyst>=22.04"`)
   or replace the import with
   `from torch.utils.data.distributed import DistributedSampler`.
3. **setuptools < 81** required for `pkg_resources` (Lightning 1.9 still
   imports it).
4. **`src/dataloader.py:6`** does `from utils import …` (no leading dot) —
   either add `codes/src` to `PYTHONPATH` or convert to a relative import.

## Key differences in a sentence each

1. **Loss function**: TF uses CRPS-based ensemble content loss; PyTorch
   introduces a *differentiable CSI* surrogate explicitly tuned for EP
   thresholds. This is the paper's main scientific contribution and is the
   reason "CGAN" beats "CGAN-pre" by ~48% on >50 mm/3h CSI.

2. **Training schedule**: TF jumps straight to joint training; PyTorch uses
   a 3-stage progressive recipe (corrector pretrain → generator pretrain →
   full GAN). The progressive schedule is what makes the larger 8× super-
   resolution stable.

3. **Discriminator**: TF is single-scale; PyTorch supports multi-scale +
   spectral norm. Multi-scale helps capture both rainband-scale structure
   and convective-scale intensity.

4. **Throughput**: TF is single-GPU at bs=2 (the content loss is heavy).
   PyTorch uses DDP at bs=256 with FP16, an order of magnitude faster end-
   to-end.

5. **Inputs**: TF uses 6-h accumulation with no synoptic padding; PyTorch
   uses 3-h accumulation with a 768-km synoptic context channel that gives
   the generator information about the regional flow regime around the
   target patch.

## Porting roadmap — TF tutorial → PyTorch

| TF file | PyTorch equivalent | Effort | Notes |
| --- | --- | --- | --- |
| `model/gan.py` (WGANGP) | `LightningModule.training_step` w/ manual optimizer toggle | High | GP becomes inline `torch.autograd.grad` |
| `model/models.py` generator | `nn.Module` w/ `nn.Upsample(bilinear)` + residual blocks | High | Watch noise channel order |
| `model/models.py` discriminator | `nn.Module` w/ optional `nn.utils.spectral_norm` | High | Mirror LR/HR path concat |
| `model/blocks.py` | reuse PyTorch residual blocks | Medium | `nn.ReflectionPad2d` for the custom padding |
| `model/layers.py` `GradientPenalty` | inline in `training_step()` | Medium | See `models_05_loglrts_val.py:1316–1333` |
| `model/noise.py` | `torch.randn` per step | Low | Stateless |
| `data/data.py`, `data/tfrecords_generator.py` | `torch.utils.data.Dataset` reading netCDF directly | High | Skip TFRecords — they were a TF perf hack |
| `evaluation/*` | unchanged | Low | Numpy, framework-agnostic |
| `config/config.yaml` | unchanged | Low | PyYAML loads same dict |
| `main.py` | replace `setupmodel`/`setupdata` with PyTorch equivalents; use `pl.Trainer` | Medium | Move checkpointing to `ModelCheckpoint` |
| `model/vaegantrain.py` | optional; skip for MVP | High | Complex |

Estimated effort: ~800–1200 LOC, 3–5 days for an experienced PyTorch
developer. The metric stack and config files come over essentially free.

## Which to use for East Africa EP work?

Recommendation: **adopt the PyTorch EP repo, swap in an ICPAC dataset
adapter, and port the TF evaluation suite over.**

Reasons:
- The differentiable CSI loss is precisely what's missing in the TF tutorial
  for EP-grade events.
- 3-h accumulation aligns with operational early-warning workflow over EA.
- Multi-GPU + FP16 makes per-lead-time training tractable in hours instead
  of days.
- The TF eval suite (RAPSD, thresholded ranks, CRPS) is more mature and is
  framework-agnostic — keep it as the single source of truth across both
  experiments.

See `east_africa_kenya_training_plan.md` (in this same `docs/` folder) for
GPU budgeting, dataset strategy, and the case for an East-Africa-wide
training domain vs Kenya-only.
