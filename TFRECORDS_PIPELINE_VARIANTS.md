# TFRecords pipeline variants — RFE2 vs cGAN_tutorial (verified)

There are **two distinct cGAN codebases** in this workspace that both turn the
same Oxford IFS NetCDF into training TFRecords, but with very different scope —
which is why the published TFRecord size differs by ~5–7×. Verified by reading
the actual `write_data` / `data.py` in each repo (2026-06-30).

| | **A. SEWAA-forecasts-RFE2** | **B. cGAN_tutorial** (this repo, `snath-xoc`) |
|---|---|---|
| Code path | `SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan/` | `tensorflow-dev-test/data/` |
| Truth / cadence | RFE2 daily, `HOURS = 24` | IMERG, `HOURS = 6` (6-hourly) |
| Lead times written | **1** — `for time_idx in range(1,2)` | **4** — `np.arange(30,54,6)` → 30/36/42/48 h |
| Channels per field | **2** (mean+sd, single window) | **4** (mean+sd at *both* interval ends) |
| Total input channels | **28** = `2×14` | **56** = `4×14` |
| `DEFAULT_FCST_SHAPE` | `(128,128,28)` | `(128,128,56)` |
| Classes / rain bins | 4 — `[0.0059,0.0362,0.0761]` mm/hr | 4 — `[0.2,0.3,0.45]` mm/hr |
| TFRecord filename | `2018_1.<class>.tfrecords` | `2018_<30\|36\|42\|48>.<class>.tfrecords` |
| Files per year | 4 (1 lead × 4 class) | 16 (4 lead × 4 class) |

Both use the same **14 forecast fields** (`cape,cp,mcc,sp,ssr,t2m,tciw,tclw,tcrw,tcw,tcwv,tp,u700,v700`).

## Why the sizes differ — and the key point

The raw archive is ~388 GB (14 fields × 4 years × ~29 valid-times × mean+sd).
**Both pipelines keep only a *subset* of the lead-time dimension** — neither
writes the whole 388 GB. The difference is *how big a subset*:

| Pipeline | Lead times kept | Channels | TFRecords (GZIP) | Fraction of raw |
|---|---|---|---|---|
| **A. RFE2** | 1 of ~28 | 28 | **~14–20 GB** | ~4–5% |
| **B. cGAN_tutorial** | 4 of ~28 | 56 | **~100 GB** *(see caveat)* | ~25% |
| *(hypothetical "all lead times")* | 28 of 28 | 56 | **~700 GB+** | >100% of raw (patch overlap) |

Per-patch bytes: A = `128²×(28+2+1)×4 ≈ 1.94 MB`; B = `128²×(56+2+1)×4 ≈ 3.87 MB`.

> **So "the entire 388 GB into TFRecords" is neither needed nor what either
> pipeline does.** Training *streams* random patches; it samples across dates and
> the chosen lead window. Converting *all* lead times would produce **more** bytes
> than the raw archive (because of patch overlap), for no training benefit.

### ⚠️ Caveat on the cGAN_tutorial ~100 GB figure
The `write_data` in `tensorflow-dev-test/data/tfrecords_generator.py` currently
loops `for batch in range(nsamples)` (= 8) instead of `range(len(dgc))` (all
dates) — so **as written it samples only the first ~8 dates per lead time**
(it lives under `tensorflow-dev-test/`). That makes the *current* output only a
few GB. The ~100 GB estimate assumes the production fix (iterate **all** dates,
like variant A does). Confirm which behaviour you want before a full run.

## Switching the routine between A and B

The driver (`prep_year.sh`) and transfer routine are parameterised by
`REPO_DIR` — point it at whichever code path you want:

```bash
# Variant A (RFE2, 1 lead, 28 ch, ~14–20 GB)
export REPO_DIR=/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan

# Variant B (cGAN_tutorial, 4 leads, 56 ch, ~100 GB)
export REPO_DIR=/scratch/notebook/cGAN_tutorial/tensorflow-dev-test/data
```

**Lead-time coverage is a code constant, not a CLI flag** in either repo — to
change how many lead times are written, edit the loop in that repo's
`tfrecords_generator.py` / `data.py`:
- A: `for time_idx in range(1,2)`  → widen to e.g. `range(1,5)`
- B: `np.arange(30,54,6)`          → e.g. `np.arange(0,168,6)` for all 28

Channels (28 vs 56) follow automatically from each repo's `load_fcst` and
`DEFAULT_FCST_SHAPE` — no manual edit needed.

> Plan accordingly: variant B's ~100 GB still fits the disk-frugal per-year loop
> (~25 GB/yr of TFRecords + ~97 GB raw → keep peak ≈ ~125 GB), but the source.coop
> upload and GPU-side download are ~5–7× larger than the variant-A numbers in
> `RFE2_cGAN_RUNBOOK.md`.
