# East Africa / Kenya EP-cGAN training plan — GPU budget and dataset strategy

This document captures the planning work for adapting the PyTorch
`CGAN_extreme_precipitation` framework (Xu et al. 2026,
DOI 10.1175/WAF-D-24-0199.1) to extreme-precipitation forecasting over East
Africa, with a focus on Kenya.

For the framework comparison this builds on, see
`tf_vs_pytorch_cgan_comparison.md` in this same folder.

## 1. Why an East Africa training domain is needed (not Kenya alone)

Kenyan rainfall is *forced by features outside Kenya*. A model trained on a
Kenya-only conditioning window will not see the synoptic drivers that
determine where and when extreme precipitation (EP) actually falls.

### Regional drivers the model needs to see

- **Indian Ocean SST and the Indian Ocean Dipole (IOD).** The east–west
  SST gradient across the equatorial Indian Ocean modulates moisture
  transport into the Horn. A positive IOD doubles MAM and OND rainfall
  anomalies over coastal Kenya, Somalia, and Ethiopia.
- **Somali jet (cross-equatorial low-level flow).** Channels moisture from
  the southern Indian Ocean across Somalia into the East African Highlands.
  Its strength and orientation determine whether long rains are normal,
  failed, or excessive.
- **Turkana jet.** Low-level wind acceleration through the Turkana
  topographic gap (between the Ethiopian and Kenyan highlands) that
  influences nighttime convection over Lake Victoria and the western Rift.
- **Congo air boundary.** Convergence zone between westerly flow from the
  Congo Basin and easterly Indian Ocean flow. Sits over western Kenya /
  Uganda during MAM and produces persistent afternoon convection.
- **ITCZ migration.** Drives the bimodal MAM (long rains) and OND (short
  rains) seasons.
- **Tropical cyclones in the SW Indian Ocean and the Madden–Julian
  Oscillation (MJO).** Modulate sub-seasonal rainfall variability across the
  Greater Horn.
- **East African Highlands and Rift Valley topography.** The HR target
  resolves topographic features at 1 km that the 8 km LR input cannot —
  this is where the super-resolver earns its keep.

If the conditioning patch (the IFS-derived input) is cropped tightly to
Kenya, none of these drivers enter the network's receptive field. The
generator cannot learn the rainfall map from the synoptic state it cannot
see.

### Practical consequence

**Train on a wider domain, evaluate on Kenya.** Two reasonable choices:

1. **Greater Horn of Africa.** Roughly 30°E–52°E, −12°S–18°N (~22° × 30°).
   Includes Ethiopia, Somalia, Kenya, Uganda, Tanzania, South Sudan, Rwanda,
   Burundi, eastern DRC, Eritrea.
2. **Full ICPAC region.** Roughly 22°E–52°E, −12°S–23°N (~30° × 35°).

Either is acceptable. The paper's padding-channel design (768 km × 768 km
synoptic context resized to the same 256 px frame as the HR target) explicitly
preserves wide-area context — keep this design when porting; it is the
cheapest way to get synoptic information into the model without enlarging the
HR receptive field.

## 2. Why the bimodal precipitation regime matters

Kenya and most of East Africa has two rainy seasons:

| Season | Months | Drivers | EP character |
| --- | --- | --- | --- |
| Long rains (MAM) | Mar–Apr–May | ITCZ northward migration, Somali jet onset, Congo air boundary | Frontal + convective, longer-duration EP |
| Short rains (OND) | Oct–Nov–Dec | ITCZ southward migration, IOD-modulated | More convective, shorter-duration EP, stronger IOD sensitivity |
| Dry / transition | Jun–Sep, Jan–Feb | upper-tropospheric subsidence | sporadic, mostly highland-driven |

A model trained on **MAM only** will fail on OND, even though OND is
operationally just as important (especially for Somalia, NE Kenya). A model
trained on the **full year** is more general but dilutes the EP signal with
~6–9 dry-season months per year.

The paper sidesteps this for China by training on May–October (the entire
Chinese wet season as one contiguous block). For East Africa the equivalent
would be two separate wet seasons — either train one combined model on both
seasons, or train two season-specific models. The combined model is
recommended for the first pass because it doubles the data per fitted
parameter.

## 3. Dataset strategy — three options compared

For "long-term data" to train on the full annual cycle vs the focused MAM
season, here are the three configurations considered:

| Scheme | Train days | MAM seasons seen | OND seasons seen | EP density per epoch |
| --- | --- | --- | --- | --- |
| **A. MAM × 3 yr** (e.g. 2024,25,26 train + 2023 val) | 276 | **3** | 0 | high |
| **B. 12 mo × 2 yr** (2025,26 train + 2024 val) | 730 | 2 | 2 | diluted by ~600 dry days |
| **C. 12 mo × 3 yr** (e.g. 2023,24,25 train + 2026 val) | ~1095 | **3** | **3** | balanced — recommended |

### Verdict

- **Don't pick B.** 12 mo × 2 yr is the weakest of the three: short on both
  axes (fewer MAM seasons than A, fewer years than C).
- **Pick A** if the immediate operational goal is MAM long-rains EP
  forecasting only.
- **Pick C** if you want one year-round model that generalises to both
  seasons. This is the *long-term* answer.

### What the paper used (China, for reference)

- May–October × 2019–2021 → 504 training days, 54 validation days, 2022 as
  test (paper §2.b).
- 3 wet seasons of data, ~18 months total.
- Scheme C above is the closest East Africa analogue.

## 4. GPU budget and benchmarks

### Hardware established for this work

- Single NVIDIA L4 (22 GiB VRAM, sm_89, Ada generation, FP8/bf16 capable).
- `torch 2.1.2`, `pytorch_lightning 1.9.4`, CUDA 12.0, cuDNN 8.9.
- AMP FP16 confirmed working; NCCL available; DDP backend `nccl` configured
  via `PL_TORCH_DISTRIBUTED_BACKEND=nccl`.

### What the paper used (inferred from code, not stated in paper text)

- 4 GPUs (`CUDA_VISIBLE_DEVICES = "0,1,2,3"` in
  `codes/mec-step3-train-generator-ts.py:2`).
- DDP via PyTorch Lightning.
- FP16 mixed precision (`precision=16`).
- Global batch 256 → 64 per GPU.
- num_workers=32 per dataloader, pin_memory=True.
- GPU model not stated in paper.md (would be in Table 3, which is not
  rendered in the markdown). Likely V100 or A100 given CMA 2024 timing.

### Per-lead-time training schedule

From the checkpoint filenames in the training scripts:

| Stage | File | Epochs | Steps/epoch | Notes |
| --- | --- | --- | --- | --- |
| 1 corrector | `mec-step2-traincorrector.py` | ~50 | ~217 | noise=0, precip-weighted L1 + diff. FSS |
| 2 generator pretrain | `mec-step3-train-generator-ts.py` (epochs=51) | 51 | ~867 | noise=0, adds differentiable CSI |
| 3 full WGAN-GP | `mec-train-final-step-ts.py` (epochs=201) | 201 | ~867 (×5 for D) | noise weighted by 0.2; ens=6 for content loss |

### Wall-clock estimates per single lead-time model

Assumes the paper's schedule (50 + 51 + 201 epochs). FP16 mixed precision
throughout. Step time scales linearly with batch size for these workloads.

| GPU setup | Stage 1 | Stage 2 | Stage 3 | **Per lead time** | **All 8 lead times** |
| --- | --- | --- | --- | --- | --- |
| 4 × A100 80GB | ~0.6 h | ~2.5 h | ~10 h | **~13 h** | **~4–5 days** |
| 1 × A100 80GB | ~1.5 h | ~6 h | ~25 h | **~32 h** | **~16–20 days** |
| 4 × V100 32GB | ~1.5 h | ~6 h | ~24 h | **~32 h** | **~10–12 days** |
| 1 × V100 32GB | ~6 h | ~24 h | ~95 h | **~5 days** | **~6 weeks** |
| **1 × L4 22GB** | **~2 h** | **~8 h** | **~30–40 h** | **~40–50 h** | **~14–20 days** |
| 1 × T4 16GB | likely OOM at bs=64; bs=16 + grad accumulation: ×4 the L4 time | | | ~1 week | ~2 months+ |

The L4 line is the realistic local-development scenario. The 4×A100 line is
the realistic production-training scenario.

### Single-L4 memory caveat

The paper uses bs=64 per GPU on probably-V100/A100. On L4 with 22 GiB:

- Stages 1 and 2 (no ensemble) fit at bs=64.
- Stage 3 runs **ensemble=6 forward passes per training step** for the
  content loss → activation memory scales ~6×. Expect OOM at bs=64.

Mitigation (in order of preference):

1. `batch_size: 32` + `accumulate_grad_batches=8` in `pl.Trainer` to recover
   the paper's effective bs=256.
2. `batch_size: 16` + `accumulate_grad_batches=16` if 32 still OOMs.
3. Reduce ensemble for content loss from 6 to 3 (will mildly hurt CSI but
   keeps training feasible).
4. Gradient checkpointing on residual blocks — last resort, ~30% slowdown.

### Required code edits for single-GPU L4

The scripts hard-code 4 GPUs. Two lines per training file need to change:

- `os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"` → `"0"`
- `'gpus': [0,1,2,3]` in the `train_hparams` dict → `[0]`

Lightning 1.9 will then fall back to single-GPU mode and DDP becomes a no-op.

## 5. Recommended pilot-then-scale plan

### Pilot (1 lead time, single L4)

- **Lead time:** `lt18h` (IFS +18 h from 00 Z init, valid 18 Z = 21:00 EAT).
  This is the afternoon-evening convective peak over the Kenyan highlands
  and the most operationally diagnostic single lead time. `lt15h` (valid
  15 Z = 18:00 EAT) is a defensible alternative if you want to catch the
  convective onset instead.
- **Dataset:** MAM × 3 yr first (Scheme A) — minimum data needed to prove
  the pipeline. Training domain: Greater Horn (30°E–52°E, −12°N–18°N).
  Evaluation: masked to Kenya only.
- **Time budget:** ~2 days wall clock on the L4. Burn budget: ~50 GPU-hours.
- **Success criteria:**
  - Stage 1 loss decreasing monotonically over the first 20 epochs.
  - Stage 2 differentiable CSI on validation > 0.1 at threshold 20 mm/3h.
  - Stage 3 ensemble-mean CSI on validation > IFS baseline.
  - RAPSD curves visually plausible (no high-frequency noise spike).
- **Outputs:** a single checkpoint, validation metrics, RAPSD plots for
  Kenya.

### Iteration 2 (same lead time, full annual cycle)

- Retrain `lt18h` on Scheme C (12 mo × 3 yr).
- Same hardware. ~3 days wall clock (slightly more data, similar epoch
  count).
- **Decision point:** does year-round training degrade MAM skill?
  - If skill is preserved or improved → Scheme C is the path forward.
  - If MAM skill degrades > 10% on CSI → fall back to two season-specific
    models (MAM, OND).

### Production (all 8 lead times)

- Hardware: 4 × A100 (cloud or shared cluster). The L4 cannot do this in
  reasonable time.
- Dataset: whichever of Scheme A vs C iteration 2 validated.
- **Time budget:** ~5 days continuous on 4×A100 → ~480 GPU-hours.
- Cloud cost ballpark (May 2026 spot prices, ~$1.50/hr per A100 spot, 4
  GPUs × 5 days): **~$700–$900 per full training run**. Use the existing
  `env/coiled_register.py` workflow if available.

## 6. Benchmarks summary table

Everything in one place for the steering doc:

| Item | Value | Source |
| --- | --- | --- |
| Paper training framework | PyTorch + Lightning 1.9.x + DDP + FP16 | code |
| Paper GPU count | 4 | `mec-step3-train-generator-ts.py:2` |
| Paper batch size (global / per GPU) | 256 / 64 | line 79 |
| Paper training data | May–Oct 2019–21, ~504 d train, ~54 d val | paper §2.b |
| Paper EP threshold | 20 mm / 3 h | paper §2.a |
| Paper key novelty | Differentiable CSI loss | paper §2.d |
| Paper CSI gain vs ECMWF (≥20 mm/3h) | +30.0% | paper §3.a |
| Paper CSI gain vs CGAN-pre (≥50 mm/3h) | +48% | paper §3.a |
| EA equivalent domain | Greater Horn 30°E–52°E, −12°N–18°N (~22°×30°) | this plan |
| Recommended EA dataset | 12 mo × 3 yr (Scheme C) | this plan |
| Recommended pilot lead time | lt18h (00 Z init, valid 18 Z) | this plan |
| EA evaluation domain | Kenya only (masked) | this plan |
| Single L4 22GB available | yes (confirmed) | `check_env.py` output |
| Single-L4 single-lead-time wall-clock | ~40–50 h | this plan |
| Single-L4 8-lead-time wall-clock | ~14–20 days | this plan |
| 4×A100 8-lead-time wall-clock | ~4–5 days | this plan |
| 4×A100 cost estimate (spot) | ~$700–$900 per full run | this plan |

## 7. Inference-time data availability — AWS S3 ECMWF Open Data

This section is operational and decides whether the channel set chosen at
training time will actually be reachable at inference time over East Africa.
The reference is the existing ICPAC GIK pipeline in
`example_notebooks/cgan_ecmwf/` (`README.md` line 257, table of "Variables
Extracted", and `stream_cgan_variables.py` line 99 `CGAN_SURFACE_VARS`).

### What AWS S3 ECMWF Open Data ENS actually provides

- **Surface fields available:** `10u, 10v, 2t, 2d, msl, sp, skt, tcw, tcwv,
  tp, ssr, ssrd, sf, ro, tcc` (and a handful of accumulated/instantaneous
  variants).
- **Pressure-level fields available:** `gh, t, u, v, w, q` at 13 levels
  (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa).
- **Static:** `lsm` (land-sea mask).
- **NOT in the ENS open data product:** `cp` (convective precip), `lsp`
  (large-scale precip), `cape`, `tciw`, `tclw`, `tcrw`, `mcc`, `lcc`, `hcc`.

The current `stream_cgan_variables.py` extracts only 12 of the available
fields (the subset the TF tutorial needs). Adding more is a one-line edit
per field — the data is in the feed, the script just doesn't currently
fetch it.

### Mapping the PyTorch EP 11-channel set to AWS S3 Open Data

Paper channel set (filename `det_tp_pad_pw_cape_cp_msl_sp_v_u_gh_vb_ub_lsp`):

| Channel | What it is | AWS S3 | Status |
| --- | --- | --- | --- |
| `tp` | total precipitation (3-h accum) | ✅ direct | already in `stream_cgan_variables.py` |
| `pad` | tp resized 768 km → 256 km (synoptic context) | ✅ derived from `tp` | a post-processing op, free |
| `pw` | precipitable water | ✅ direct (= `tcwv`) | already in script (as `tcwv`) |
| `msl` | mean sea-level pressure | ✅ direct | needs adding to `CGAN_SURFACE_VARS` |
| `sp` | surface pressure | ✅ direct | already in script |
| `u` | low-tropospheric u-wind | ✅ direct at any of 13 pressure levels | already in script at 700 hPa |
| `v` | low-tropospheric v-wind | ✅ direct at any of 13 pressure levels | already in script at 700 hPa |
| `ub` | u-wind at 2nd level (e.g. 925 hPa) | ✅ direct | needs adding to `CGAN_PRESSURE_VARS` |
| `vb` | v-wind at 2nd level (e.g. 925 hPa) | ✅ direct | needs adding to `CGAN_PRESSURE_VARS` |
| `gh` | geopotential height (e.g. 500 hPa) | ✅ direct | needs adding to `CGAN_PRESSURE_VARS` |
| `cape` | CAPE | ⚠️ proxy via `mucape` | same workaround as TF inference uses |
| **`cp`** | convective precipitation | ❌ not in ENS open data | TF script uses `sf` (snowfall) as proxy |
| **`lsp`** | large-scale precipitation | ❌ not in ENS open data | no clean derivation |

### Net result

- **9 channels directly extractable** from AWS S3 ECMWF Open Data
  (`tp, pad, pw, msl, sp, u, v, ub, vb, gh`). Six of these are already in
  `stream_cgan_variables.py`; four need adding to the extraction dicts.
- **1 channel via proxy** (`mucape` → `cape`). Same workaround the TF
  inference pipeline already uses.
- **2 channels genuinely unavailable** (`cp`, `lsp`).

### Comparison with the TF tutorial set

For reference (see `tf_vs_pytorch_cgan_comparison.md` §5):

| Set | Direct from open data | Proxy needed | Genuinely missing |
| --- | --- | --- | --- |
| TF tutorial (14 ch) | 8 | 3 (`mucape→cape`, `sf→cp`, `tcc→mcc`) | 3 (`tciw`, `tclw`, `tcrw`) |
| **PyTorch EP (11 ch)** | **9** | **1 (`mucape→cape`)** | **2 (`cp`, `lsp`)** |

The PyTorch channel set is **operationally more portable** to AWS S3 ECMWF
Open Data than the TF set, even after accounting for the `cp`/`lsp` gap.

### How to handle the gaps in the EA port

1. **Drop `lsp`.** It is largely redundant with `tp − cp` and is not in the
   ENS open data feed. Most generative downscaling work (Harris et al. 2022;
   corrector-GAN) does not treat `lsp` as a separate channel. Dropping it
   takes the model to a 10-channel set.

2. **Replace `cp` with `sf` proxy.** Follow the existing TF inference
   pattern (`stream_cgan_variables.py` already extracts `sf` "for cp
   estimation"). Train *and* infer on this proxy — no train/inference
   distribution mismatch.

3. **Use `mucape` for `cape`.** Same convention as the TF inference
   pipeline. Acceptable for EP detection since MU-CAPE is more relevant for
   the highest-intensity convection anyway.

4. **Extend `stream_cgan_variables.py`** to also pull `msl` (surface) and
   `gh@500`, `u@925`, `v@925` (pressure-level). These exist in the open
   data feed and only need to be added to `CGAN_SURFACE_VARS` /
   `CGAN_PRESSURE_VARS` (script lines 101 and 118).

### Operational implication

There is **no fundamental blocker** to running the PyTorch EP model
operationally over East Africa from the AWS S3 ECMWF Open Data feed. The
two unavailable channels (`cp`, `lsp`) are handled the same way the TF
pipeline already handles its own missing channels. This removes a class of
risk that the TF tutorial set carries (3 unhandled hydrometeor channels
with no clean proxy).

For the pilot training in this plan, use the **10-channel reduced set** —
drop `lsp`, use `sf` as the `cp` proxy at both training and inference —
and confirm CSI/FSS skill before considering adding any further fields.

## 8. Open questions / known unknowns

- **Padding-channel scaling for EA.** The paper uses 768 km × 768 km
  synoptic context. For East Africa the relevant synoptic features (Somali
  jet, IOD signature) extend further — may want to test 1024 km or 1536 km
  for the padding channel.
- **High-resolution observational target.** Which 1 km gridded product to
  use over EA: CHIRPS (5 km native), IMERG-Final (10 km), TAMSAT, or a
  fused product? Paper uses CMPAS (1 km, China only). This is the single
  biggest data-engineering decision and is *outside* this plan.
- **Lead-time-specific vs single-model approach.** Paper trains 8 separate
  models. A single model conditioned on lead time as an extra input would
  cut training time ~8× but is unproven at this resolution.
- **ENSO/IOD stratification.** Should validation hold out a strong-IOD
  year specifically? 2023 was a positive-IOD year and a useful holdout in
  that respect.

## 9. References

- Xu, J., Dai, K., Ma, J., Zhang, Q., Chen, Y., Zhang, F., Ng, C.-P.
  (2026). Postprocessing for 24-hour advanced forecasting of extreme
  precipitation using deep learning generative models. *Wea. Forecasting*,
  41, 381–401. DOI 10.1175/WAF-D-24-0199.1.
- Ravuri et al. (2021), DGMR for radar nowcasting — origin of the
  precipitation-weighted loss.
- Price & Rasp (2022), Corrector-GAN — origin of the corrector + super-
  resolver dual-output architecture.
- Larraondo et al. (2020), differentiable CSI / categorical skill scores.
- Harris et al. (2022), GAN/VAE-GAN for ECMWF IFS downscaling — closely
  related EA work (used IFS predictors, generated 1 km probabilistic
  precipitation).
- Rasp's `nwp-downscale` framework on GitHub —
  https://github.com/raspstephan/nwp-downscale.
