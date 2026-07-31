# ECMWF Icechunk → East Africa cGAN: what to pull, per era, on a Dask cluster

**Which variables and which vertical levels have to come out of the published
ECMWF IFS ensemble Icechunk store for each of its three schema eras, whether
the additional predictors proposed by the members exist in ECMWF open data,
and the concrete Dask-cluster recipe to materialise the realised subset.**

_Written 2026-07-30. Every availability claim below was verified by opening the
live store anonymously and reading the field — not inferred from the era
inventory alone. Probe dates: 2024-05-15 and 2025-05-15, +24 h, member 0,
East Africa box._

Sibling docs:
- `east_africa_cgan_variable_selection_rationale.md` — *why* each channel earns
  its place (driver → field → variable chain).
- `east_africa_kenya_training_plan.md` — domain options, dataset schemes, GPU
  budget.
- `grib-index-kerchunk/ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md` —
  the per-era `.index` inventory this cross-checks against.

Runnable companion: `pytorch-cgan/materialize_ea_icechunk_dask.py`.

---

## 0. Answers up front

| Question | Answer |
|---|---|
| Is the current cGAN conditioned on **700 hPa U and V**? | **Yes — and 700 hPa is the *only* vertical level in the set that is training today.** §1 |
| Do the store's levels support it? | Yes. 700 hPa is present in **all three eras**. §2 |
| **RH at 800 hPa?** | ❌ **800 hPa does not exist in any era.** Nearest published level is **850 hPa**. §4 |
| **SST?** | ❌ not published in ECMWF open data. Use **`skt`** (skin temperature) masked to water — over open ocean IFS skin temperature *is* the SST field. §4 |
| **OLR?** | ✅ as **`ttr`** (top net thermal radiation, accumulated J m⁻²). `OLR = −ttr/Δt`. Verified: −2.32×10⁷ J m⁻² at +24 h → **268 W m⁻²**. §4 |
| **K-index?** | ❌ not published, ✅ **fully derivable** from `t`+`r` at 850/700/500 — all present in every era. §4 |
| **Wet-bulb potential temperature θw?** | ❌ not published, ✅ **derivable** from `t`+`r` at 850 hPa. §4 |
| **U/V at 200, 500, 850 hPa? RH at 700 hPa?** | ✅ all present in **all three eras**. §3 |
| **Vertical velocity `w` at 925/850/700/500?** | ✅ in 49r1 and 50r1 — ❌ **`w` does not exist at all in the 0p4 era**. §3 |
| Net effect on the training window | **Adopting the proposed set rules out MAM 2023.** The usable window becomes **2024-02-29 → present** (49r1 + 50r1). §5.1 |
| Biggest trap | The `49r1/00z` group is a **union of two sub-eras**. `cape`, `mucape`, `tcw`, `tprate`, `ptype` all read back **silently all-NaN** outside their sub-window. §2.4 |
| Second-biggest trap | The **`pytorch-cgan` extraction box (20–53 °E) does not cover the grid the model actually trains on** (19.15–54.25 °E). §5.2 |

---

## 1. Confirming the vertical level in the current training

**Confirmed: 700 hPa, `u700` + `v700`, and it is the only pressure level anywhere
in the live configuration.**

| Where | Evidence | Level |
|---|---|---|
| **Live training field set** — `SEWAA-forecasts-RFE2/SEWAA-forecasts/24h_accumulations/cGAN/dsrnngan/data.py:37` | `_F13 = ['cp','mcc','sp','ssr','t2m','tciw','tclw','tcrw','tcw','tcwv','tp','u700','v700']`; `_F14 = ['cape'] + _F13`. Default `FIELD_SET = 'run11'` → `_F13`. | **700 hPa only** |
| Tutorial pipeline — `tensorflow-dev-test/data/data.py:20` | same 14-field list, `u700, v700` | 700 hPa only |
| TFRecords writer — `make_local_tfrecords.py:28` | same list | 700 hPa only |
| Inference streamer — `gefs-gik-data-prepration/cgan_ecmwf/stream_cgan_variables_coiled.py:80` | `{'u': 'u700', 'v': 'v700'}`, `TARGET_PRESSURE_LEVEL = 700` | 700 hPa only |
| GEFS cross-map — `2026-03-01-cgan-normalization-investigation.md:103` | `u700 → ugrd`, flagged "**700 hPa wind, NOT 10 m**" | 700 hPa |

Two caveats worth stating plainly, because they change what "current" means:

1. **RFE2 runs 04/05 trained on a 4-field surface-only set** (`tp, t2m, tcwv,
   sp`, `data.py` `'run05'`) — *no pressure level at all*. Runs 06–11 restored
   the 13/14-field set, so `u700/v700` are in the current champion. See
   `RFE2_cGAN_CRPS_AND_DATASET.md:55`.
2. **The PyTorch EP plan wants three levels — 700, 925, 500 —** (`u/v@700`,
   `ub/vb@925`, `gh@500`), but
   `pytorch-cgan/ingest_ecmwf_pytorch_cgan_variables.py:258` still has
   `PRESSURE_VARS = {}`: the pressure-level channels are disabled pending the
   GIK parquet per-level-key fix. **The Icechunk store makes that fix moot** —
   `isobaricInhPa` is a real dimension there, so `.sel(isobaricInhPa=700)` is
   unambiguous. See §5.5.

---

## 2. The store as it actually is

Opened anonymously at `https://data.source.coop`, bucket `e4drr-project`,
prefix `forecasts/ecmwf_ifs_ens_aws_s3_icechunk_vd`. Arrays live under
`{era}/00z`, never at the root.

### 2.1 Eras

| Group | Grid | Window (00z) | dates | pl levels | pl vars | sfc vars | total |
|---|---|---|---|---:|---:|---:|---:|
| `0p4/00z`  | 451 × 900  | 2023-01-18 … 2024-02-28 | 401 | **9**  | 8  | 11 | 19 |
| `49r1/00z` | 721 × 1440 | 2024-02-29 … 2026-05-12 | 794 | **13** | 9  | 50 | 59 |
| `50r1/00z` | 721 × 1440 | 2026-05-13 … present    | 51  | **14** | 10 | 44 | 54 |

All three: `number = 51`, `step = 85`.

### 2.2 Vertical levels, per era

```
0p4  ( 9):                        50, 200, 250, 300,      500,      700, 850, 925, 1000
49r1 (13):      50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
50r1 (14): 10,  50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
```

**There is no 800 hPa level in any era.** There is also no 925→850 intermediate,
no 750, no 775. The proposed "RH at 800 hPa" must become **RH at 850 hPa**
(present everywhere), optionally blended with 700 hPa.

Every level the proposed channel set needs — **200, 500, 700, 850, 925** — is
present in **all three eras**. The level choice is therefore era-safe; only the
*variables* break across eras.

### 2.3 Pressure-level variables, per era

| era | pl vars |
|---|---|
| `0p4`  | `d, gh, q, r, t, u, v, vo` — ⚠️ **no `w`** |
| `49r1` | `d, gh, q, r, t, u, v, vo, w` |
| `50r1` | `d, gh, q, r, t, u, v, vo, w, z` |

### 2.4 The `49r1` union trap — verified, and the single most important gotcha

The `49r1/00z` group spans **two** schema sub-eras (9-level until 2025-01-14,
13-level after). The store declares the **union** of both variable sets and both
level sets, so a variable that belongs to only one sub-era **still opens, still
has the right shape, and reads back all-NaN** in the other. No exception is
raised. Measured, EA box, member 0, +24 h:

| field | 2024-05-15 (9-level sub-era) | 2025-05-15 (13-level sub-era) |
|---|---|---|
| `cape`   | ✅ finite 1.00, 0–1076 J kg⁻¹ | ❌ **all-NaN** |
| `mucape` | ❌ **all-NaN** | ✅ finite 1.00, 0–2500 J kg⁻¹ |
| `tcw`    | ❌ **all-NaN** | ✅ 9.3–86.3 kg m⁻² |
| `tprate` | ❌ **all-NaN** | ✅ finite |
| `ptype`  | ❌ **all-NaN** | ✅ finite |
| `tcc`, `sf` | ❌ all-NaN | ❌ **all-NaN** (declared in the group, never populated — they only exist from 50r1) |
| `ttr`, `skt`, `ssr` | ✅ | ✅ |
| `w`@700  | ✅ −2.17…1.03 Pa s⁻¹ | ✅ −3.50…2.22 Pa s⁻¹ |
| `r`@925/850/700/500/200 | ✅ | ✅ |
| `r`@**600/400/150/100** | ❌ **all-NaN** | ✅ |

> This is *exactly* the failure mode that produced the all-NaN/all-zero
> `cp_proxy` channel in the earlier surface-only build (rationale §5.2). It is
> not a bug in the store — it is the honest representation of a mid-era schema
> change. **Every extraction must assert a non-zero finite fraction per
> (channel, date-block) and fail loudly**, which is what `--check` and the
> `finite == 0.0` guard in the companion script do.

### 2.5 Coordinates, steps, members

- `latitude` **descends** 90 → −90 (so `slice(25.25, -15.25)`, high first).
- `longitude` is **0 … 359.75** (0-360 convention). East Africa 18.5–55 °E needs
  no wrap or roll. 0p4 is 0.4°/451×900; 49r1 and 50r1 are 0.25°/721×1440.
- `step`: 85 values — **3-hourly 0…144 h, then 6-hourly 150…360 h**.
- `number`: 0…50 (0 = control). In 50r1 the control moved to the `oper/fc`
  stream upstream; the store still exposes `number=0` and it decodes.

### 2.6 The RAM constraint that dictates the whole Dask design

Resolving **any** chunk of an array makes icechunk load that array's **entire
manifest** — ~200 bytes per chunk-ref, refs = `dates × members × steps × levels`.
Subsetting time/level/member does **not** reduce it.

| array | chunk-refs | peak RSS |
|---|---:|---:|
| `50r1/t2m` | 221 K | 0.25 GB |
| `49r1/t2m` (794 dates, 2-D) | 3.44 M | 0.80 GB |
| `49r1/t` (794 dates × 13 levels) | **44.7 M** | **~9.1 GB** |

So: **a worker holds about one 49r1 pressure-level manifest at a time.** Tasks
must be big enough to amortise that 9 GB load — one task = *(one store variable,
all its levels, one year)*, never one task per (variable, date).

*Upstream fix (out of scope here):* rebuild the store with
`icechunk.ManifestSplittingConfig` split along `time`, so a read loads one shard
instead of the whole array manifest.

---

## 3. Part A — per-era download matrix

✅ = published and verified finite · ❌ = not published (or declared but never
populated) · the 49r1 column is split at the 2025-01-14 sub-era break.

### 3.1 Surface / single-level

| Out channel | Store var | 0p4 (2023) | 49r1 ≤2025-01-13 | 49r1 ≥2025-01-14 | 50r1 | Notes |
|---|---|:--:|:--:|:--:|:--:|---|
| `tp` | `tp` | ✅ | ✅ | ✅ | ✅ | accumulated from step 0 — **difference consecutive steps** |
| `pw` | `tcwv` | ✅ | ✅ | ✅ | ✅ | precipitable water |
| `sp` | `sp` | ✅ | ✅ | ✅ | ✅ | |
| `msl` | `msl` | ✅ | ✅ | ✅ | ✅ | |
| `t2m` | `t2m` | ✅ | ✅ | ✅ | ✅ | store name is `t2m` in all three eras |
| `skt` → **SST** | `skt` | ✅ | ✅ | ✅ | ✅ | mask with `lsm < 0.5`; **the only SST-like field** |
| `ssr` | `ssr` | ❌ | ✅ | ✅ | ✅ | accumulated |
| `ttr` → **OLR** | `ttr` | ❌ | ✅ | ✅ | ✅ | accumulated; `OLR = −ttr/Δt` |
| `tcw` | `tcw` | ❌ | ❌ | ✅ | ✅ | |
| `cape` | `cape` | ❌ | ✅ | ❌ | ❌ | surface-parcel CAPE |
| `mucape` | `mucape` | ❌ | ❌ | ✅ | ✅ | most-unstable CAPE |
| `ptype` | `ptype` | ❌ | ❌ | ✅ | ✅ | optional convective/stratiform flag |
| `tcc` | `tcc` | ❌ | ❌ | ❌ | ✅ | the only `mcc` substitute; 50r1 only |
| `lsm` | `lsm` | ✅ | ✅ | ✅ | ✅ | **static — read once, not per date** |
| `cp`, `lsp`, `tciw`, `tclw`, `tcrw`, `mcc`, `sst` | — | ❌ | ❌ | ❌ | ❌ | **never in open data, any era** |

### 3.2 Pressure-level (`isobaricInhPa`)

| Out channel | Store var | Level | 0p4 | 49r1 | 50r1 | Notes |
|---|---|---:|:--:|:--:|:--:|---|
| `u925`, `v925` | `u`,`v` | 925 | ✅ | ✅ | ✅ | Somali + Turkana jet core (EP `ub`/`vb`) |
| `u850`, `v850` | `u`,`v` | 850 | ✅ | ✅ | ✅ | classic low-level jet level |
| `u700`, `v700` | `u`,`v` | 700 | ✅ | ✅ | ✅ | **the level training today** |
| `u500`, `v500` | `u`,`v` | 500 | ✅ | ✅ | ✅ | mid-level steering |
| `u200`, `v200` | `u`,`v` | 200 | ✅ | ✅ | ✅ | upper-level divergence / TEJ |
| `gh500` | `gh` | 500 | ✅ | ✅ | ✅ | ITCZ / mass field |
| `w925`,`w850`,`w700`,`w500` | `w` | 925/850/700/500 | ❌ **no `w` in 0p4** | ✅ | ✅ | large-scale ascent |
| `r850` | `r` | 850 | ✅ | ✅ | ✅ | **stands in for the requested "RH 800"** |
| `r700` | `r` | 700 | ✅ | ✅ | ✅ | |
| `t850`,`t700`,`t500` | `t` | 850/700/500 | ✅ | ✅ | ✅ | inputs to K-index / θw |
| `q` (alt. to `r`) | `q` | any | ✅ | ✅ | ✅ | specific humidity, if preferred |
| RH @ **800** | — | 800 | ❌ | ❌ | ❌ | **level does not exist** |

---

## 4. Part B — reflection on the proposed variables

| Proposed | In open data? | Verdict | What to actually pull |
|---|---|---|---|
| **SST** | ❌ no `sst` field in any era | **proxy** | `skt` masked to water (`lsm < 0.5`). Over open ocean IFS skin temperature *is* the prescribed SST. |
| **OLR** | ✅ as `ttr` | **keep** | `ttr`, convert `OLR = −ttr/Δt`. **Not in 0p4.** |
| **U, V @ 500 hPa** | ✅ all eras | keep (Tier 2) | `u`,`v` @ 500 |
| **RH @ 800 hPa** | ❌ level absent | **substitute** | `r` @ **850** (and `r` @ 700 for the mid-level) |
| **K-index** | ❌ not published | **derive** | `t`@850/700/500 + `r`@850/700 → `K = (T₈₅₀−T₅₀₀) + Td₈₅₀ − (T₇₀₀−Td₇₀₀)` |
| **`w` @ 925/850/700/500** | ✅ 49r1, 50r1 · ❌ 0p4 | keep | `w` at four levels — **kills MAM 2023** |
| **U, V @ 850 hPa** | ✅ all eras | keep (Tier 2) | `u`,`v` @ 850 |
| **U, V @ 200 hPa** | ✅ all eras | **keep (Tier 1)** | `u`,`v` @ 200 — the most *independent* of the wind additions |
| **RH @ 700 hPa** | ✅ all eras | keep (Tier 1) | `r` @ 700 |
| **Wet-bulb potential temperature** | ❌ not published | **derive** | `t`@850 + `r`@850 → θe (Bolton 1980 eq. 43) → θw (Davies-Jones 2008) |

### 4.1 SST — the honest version

There is no `sst` in ECMWF open data (`enfo/ef` or `oper/fc`), in any era. The
ocean-adjacent fields the store *does* carry are `skt`, `zos` (sea-surface
height), `sithick`, `asn`. **`skt` over water is the usable proxy** and it is
available in all three eras.

But be clear about what it buys. The rationale doc's position (§3, §6) is that
IOD/ENSO are captured **implicitly, by domain width** — they modulate the wind,
moisture and pressure fields the model already ingests. Adding `skt` inside a
20–55 °E box does *not* change that, because **the IOD east pole sits at
90–110 °E, far outside the conditioning window.** An SST channel cropped to East
Africa carries the *local* west-pole gradient and the coastal upwelling signal —
useful, but it is not "the IOD".

If the intent is genuinely to condition on the basin mode, do it deliberately:

- **Option A (recommended, cheap):** extract `skt` and `ttr` on a **wide, coarse
  box** — 30–120 °E, 25 °S–25 °N, coarsened 0.25° → 2° — as *extra* low-resolution
  channels or reduced to indices (DMI from the two IOD poles; an MJO-phase proxy
  from OLR). Read cost is control-member-only and negligible.
- **Option B:** skip SST entirely and keep relying on domain width, per the
  existing rationale.

Do **not** add a Kenya-box SST channel and describe it as an IOD predictor.

### 4.2 OLR — the cleanest win in the list

`ttr` is top net thermal radiation, accumulated in J m⁻² since step 0.
Verified at +24 h over the EA box: −2.32×10⁷ J m⁻² → **−2.32×10⁷ / 86 400 =
−268 W m⁻²**, i.e. OLR ≈ 268 W m⁻², a textbook tropical value. So the
conversion is confirmed, not assumed.

OLR is the standard tropical-convection proxy and the standard MJO variable. It
is *cheap* (one surface field), *era-stable* across 49r1+50r1, and it partially
substitutes for the permanently-missing `mcc`/`tcc` cloud channels. **Of every
variable proposed, this is the one to add first.**

### 4.3 K-index and θw — derive them, don't look for them

Neither is a published IFS open-data field. Both are cheap functions of fields
that *are* published at every level needed, in **every era including 0p4**:

- **K-index** `= (T₈₅₀ − T₅₀₀) + Td₈₅₀ − (T₇₀₀ − Td₇₀₀)`. Needs `t`@850/700/500
  and `r`@850/700 (dewpoint via Magnus). Five reads, one derived channel.
- **θw@850** — via Bolton (1980) θe then the Davies-Jones (2008) inversion.
  Needs `t`@850 + `r`@850. Zero extra reads once `t850`/`r850` are pulled.

Both are computed on the worker after the read, so they cost CPU, not S3.

A caveat on K-index specifically: it was designed for **midlatitude airmass
thunderstorms**. Over equatorial East Africa, where the mid-troposphere is
persistently moist during MAM, K saturates and discriminates poorly. `mucape`
+ `w` is the better-conditioned instability pair for this region. Keep K, but
put it in Tier 3 and ablate it — do not assume it adds skill because it is a
recognised index.

### 4.4 On channel count — the one place to push back

The proposed set takes the model from **11 predictor channels to ~26** (plus
`pad` and the static constants). Two concerns worth weighing before committing
the full extraction:

1. **Vertical redundancy.** `u`/`v` at 925, 850, 700, 500 and 200 are strongly
   correlated in the vertical. 925 and 850 in particular sample the same
   low-level jet. Expect diminishing returns per added level, while read cost
   scales *linearly*. Same for `w` at four levels.
2. **Sample size.** MAM × 3 years is ~276 forecast days. Doubling the input
   channels against a fixed, small training set is exactly the regime where
   extra predictors cost more in variance than they return in signal. The
   RFE2 sweep already measured field-set as the dominant lever (~14% CRPS,
   `CHIRPS_cGAN_RUNBOOK.md:140`) — which cuts both ways: it is worth testing,
   and it is worth testing *properly*.

**Recommendation — extract the superset once, train in tiers.** Extraction is
the expensive, hard-to-repeat step; training is cheap to repeat. So pull
everything in §5.3, then ablate:

| Tier | Channels | Rationale |
|---|---|---|
| **1 — core** | `tp, pw, msl, sp, t2m, mucape\|cape, w700, u/v@925, u/v@700, u/v@200, gh500, r700, r850, olr` | the EP set + the three highest-information additions (upper winds, RH, OLR) |
| **2 — add** | `u/v@850, u/v@500, w925, w850, w500, sst(skt), ssr, tcw` | the vertically-redundant and secondary fields |
| **3 — test** | `kindex, thetaw850, ptype` | derived indices with region-specific caveats |

Before committing to Tier 2, run the correlation check in §5.7 — it costs one
season of data and can retire half the wind channels on evidence.

---

## 5. Part C — final Dask-cluster extraction spec

### 5.1 Era / training-window decision

| If you want… | Usable eras | Window | Cost |
|---|---|---|---|
| `w`, OLR, CAPE-family — i.e. the proposed set | `49r1` + `50r1` | **2024-02-29 → present** | **MAM 2023 is unusable** (0p4 has no `w`, no `ttr`, no CAPE) |
| To include MAM 2023 | + `0p4` | 2023-01-18 → | drop `w`, `olr`, `cape`, `ssr`, `tcw`; 0.4° grid needs separate regridding |

**Recommendation: 49r1 + 50r1 only.** Take MAM 2024 + MAM 2025 + MAM 2026 (276
forecast days), and handle the `cape` → `mucape` break at 2025-01-14 by
harmonising on `mucape` + `w`@700 with **per-sub-era normalisation** (rationale
§7.5, option 1). MAM 2024 falls entirely in the `cape` sub-era; MAM 2025 and
MAM 2026 in `mucape`.

### 5.2 Extent — resolve the box mismatch first

Three different boxes exist in this repo, and they do not agree:

| Source | Box | Grid |
|---|---|---|
| **What actually trains** — `IFS_training/<yr>/<field>.nc` | −13.65…24.65 N, **19.15…54.25 E** | **0.1°, 384 × 352** |
| `pytorch-cgan/ingest_ecmwf_pytorch_cgan_variables.py:202` | −15…25 N, **20…53 E** | 0.25°, 161 × 133 |
| `stream_cgan_variables_coiled.py:92` (inference) | −14…25 N, 19…55 E | 0.25°, 157 × 145 |

Two things fall out of this that must not be missed:

1. **The `pytorch-cgan` box does not cover the training grid.** It stops at
   53 °E; the TF frame runs to 54.25 °E. Anything built on that box cannot be
   interpolated onto the frame the model trains on without extrapolating.
2. **The training grid is 0.1°, the store is 0.25°.** A store-based rebuild is
   an *upsampling* onto the 384 × 352 frame — which is what the Oxford IFS files
   already are. Reproduce that frame exactly, or regrid truth, constants
   (`elev.nc`, `lsm.nc`) and `FCSTNorm2018.pkl` together. Do not mix.

**Extract one 0.25° superset box with a 2-cell halo, and crop per consumer:**

```
latitude   25.25 N  →  −15.25 N     (163 points, store order is descending)
longitude  18.5 E   →   55.0 E      (147 points)
                                     163 × 147 = 23,961 cells @ 0.25°
```

This strictly contains all three boxes plus enough halo that bilinear
0.25° → 0.1° interpolation never extrapolates. It is 12% more cells than the
current `pytorch-cgan` box — negligible insurance.

Plus, if §4.1 Option A is taken, a second coarse basin box for `skt`/`ttr`
only: **30–120 °E, 25 °S–25 °N, coarsened ×8 → 2°**, control member only.

### 5.3 The fetch list — 30 reads per (date, member, step)

**Surface (10):** `tp`, `tcwv`, `sp`, `msl`, `t2m`, `skt`, `ssr`, `ttr`, `tcw`,
`cape`\|`mucape`
**Pressure-level (20):**

| var | levels | count |
|---|---|---:|
| `u` | 200, 500, 700, 850, 925 | 5 |
| `v` | 200, 500, 700, 850, 925 | 5 |
| `w` | 500, 700, 850, 925 | 4 |
| `r` | 700, 850 | 2 |
| `t` | 500, 700, 850 | 3 |
| `gh` | 500 | 1 |

**Static (1, read once):** `lsm`
**Derived on the worker (0 reads):** `olr` ← `ttr` · `sst` ← `skt`+`lsm` ·
`kindex` ← `t850/700/500`+`r850/700` · `thetaw850` ← `t850`+`r850` · `pad` ←
`tp` (at training time, not stored)

### 5.4 Steps and members

- **Steps: 24 … 54 h at 3 h (11 values).** Covers variant A's single 24 h
  accumulation window and variant B's 30/36/42/48 h set, and includes the
  step *before* each window start, which accumulated fields (`tp`, `ssr`,
  `ttr`, `str`, `ro`) need in order to be differenced.
- **Members: all 51**, reduced **on the worker** to `mean` + `sd` — the two
  channels per field the cGAN actually consumes (`load_fcst` reads
  `{field}_mean`/`{field}_sd`). The reduction shrinks the *write* side 25×;
  it does **not** shrink the read side.
- **Cheap mode: control + every 5th perturbed member (11 total).** Cuts read
  cost 4.6× for a mean/sd estimate that is close for smooth predictor fields.
  Recommended for the first full pass.

### 5.5 Why the Icechunk store, not the GIK parquet

`ingest_ecmwf_pytorch_cgan_variables.py` disabled the pressure-level channels
because the parquet exposes one `pl` reference per `(variable, step)` while the
GRIB encodes a *different* hPa level at each lead time — storing that would
silently mix levels along the lead-time axis
(`GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`).

**The Icechunk store does not have this problem.** `isobaricInhPa` is a real,
labelled dimension, so `.sel(isobaricInhPa=700)` is exact and level-mixing is
structurally impossible. Moving the extraction to the store **unblocks the wind
channels today**, without waiting for the parquet fix.

### 5.6 Cluster shape and task decomposition

Driven entirely by §2.6 (one ~9 GB pl manifest per worker) and the fact that
**cropping to East Africa does not reduce bytes read** — a virtual chunk is one
whole *global* GRIB message, so the subset only shrinks what you write.

| Setting | Value | Why |
|---|---|---|
| **Region** | **`eu-central-1`** | where `s3://ecmwf-forecasts/` actually lives (verified via `LocationConstraint`). ⚠️ `stream_cgan_variables_coiled.py:94` defaults to `eu-west-1` "close to ECMWF S3" — that is **cross-region**: slower and billed. Fix it. |
| Worker VM | `r7i.xlarge` (4 vCPU / 32 GiB) | one 9 GB pl manifest + decode buffers, with headroom |
| `nthreads` | 2 | 2 concurrent decodes; more risks a second manifest on the same worker |
| Workers | **24** | one per heavy pressure-level task (6 pl vars × 4 era-years) |
| Task unit | **(era, store variable, all its levels, one year)** | amortises the manifest load; ~64 tasks total (24 heavy pl, ~40 light sfc) |
| Manifest preload | **off** (`max_total_refs=0, max_arrays_to_scan=0`) | source.coop returns sporadic HTTP 500s; eager preload turns that into a failed open |
| Credentials | **none** | anonymous on both source.coop and `ecmwf-forecasts`; run with `AWS_*` unset and `from_env=False` |

`--cluster local` will **OOM on any 49r1 pressure-level variable** on a machine
under ~16 GB. Use it only for surface fields or a 50r1 smoke test.

### 5.7 Cost model

Grid 163 × 147 = 23,961 cells → **95.8 KB** per 2-D float32 field.
MAM × 3 = 276 dates · 11 steps · 30 fields.

| | 51 members | 11 members (cheap) |
|---|---:|---:|
| GRIB messages decoded | 4.65 M | 1.00 M |
| **Bytes read from S3** (0.8–2 MB/message) | **~4–9 TB** | ~0.8–2 TB |
| Egress cost | **$0** in-region | $0 |
| **Bytes written** (mean+sd, float32) | **~17 GB** | ~17 GB |
| Wall clock @ 48 concurrent decodes | 21 h @ 0.8 s/field · 5 h @ 0.2 s | 4.6 h @ 0.8 s · 1.2 h @ 0.2 s |

The 0.8 s/field figure is **measured from a non-AWS host with a warm manifest**
and is network-dominated; in `eu-central-1` expect materially better.
**Calibrate with the one-day dry run before sizing the cluster** — do not take
either bracket on faith.

Two knobs, in order of leverage: **members** (linear, 4.6× available) and
**levels/channels** (linear — this is what the Tier-2 correlation check buys).

### 5.8 Output layout

One NetCDF per **(channel, year)**, matching the existing
`IFS_training/<year>/<field>.nc` convention so the TF loader needs no change
beyond the frame:

```
ea_out/<year>/<channel>_<year>.nc
    dims: (time, step, stat, latitude, longitude)
    stat: ["mean", "sd"]          # the 2 channels per field the cGAN consumes
    float32, zlib level 4
```

Then, downstream and separately: difference the accumulated fields, crop/
interpolate onto the consumer frame (0.1° 384 × 352 for TF; 0.25° crop for the
PyTorch EP ingest), and rebuild `FCSTNorm*.pkl` — **normalisation constants are
per-field and must be regenerated whenever the field set changes**.

### 5.9 Running it

```bash
cd ~/cGAN_tutorial/pytorch-cgan
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE

# 1. availability check — one date per year, no cluster, ~2 min
uv run materialize_ea_icechunk_dask.py check --years 2024 2025 2026

# 2. one-day dry run — calibrates seconds/field before you size the cluster
uv run materialize_ea_icechunk_dask.py run \
    --years 2024 --dates 2024-03-01 --cheap-members \
    --cluster local --out ./ea_out_dryrun

# 3. full extraction, eu-central-1
uv run materialize_ea_icechunk_dask.py run \
    --years 2024 2025 2026 --months 3 4 5 \
    --cheap-members --basin \
    --cluster coiled --workers 24 --out ./ea_out
```

### 5.10 Acceptance checks — run every one before training

1. **No all-NaN channel.** Finite fraction > 0.99 for every
   `(channel, year)` file. The script refuses to write a 0.00-finite channel;
   verify none were silently *skipped*.
2. **The sub-era break lands where expected.** `cape_2024.nc` exists,
   `cape_2025.nc` does not; `mucape_2025.nc` and `mucape_2026.nc` exist,
   `mucape_2024.nc` does not.
3. **Physical sanity, EA MAM.** `olr` ≈ 200–300 W m⁻² · `sst` ≈ 297–303 K over
   the western Indian Ocean · `w700` within ±3 Pa s⁻¹ · `r850` in 10–100% ·
   `u925` shows the cross-equatorial southerly Somali-jet signature.
4. **Level identity.** `gh500` ≈ 5850–5900 m over the tropics — the single
   cheapest proof that level selection is not silently off.
5. **Grid registration.** Extracted lat/lon strictly bracket
   −13.65…24.65 N / 19.15…54.25 E on all four sides, so interpolation onto the
   0.1° frame never extrapolates.
6. **Vertical redundancy check** (before committing Tier 2): correlate
   `u925`/`u850`/`u700`/`u500`/`u200` pairwise over one season. Any pair above
   ~0.9 makes the second level a candidate to drop.

---

## 6. Open decisions for the team

1. **SST scope** — basin box (§4.1 Option A) or drop SST and keep relying on
   domain width? This is the only proposed variable whose value depends
   entirely on a domain choice the current box does not support.
2. **Which frame is canonical** — reproduce the 0.1° 384 × 352 Oxford frame, or
   move everything (truth, constants, norm) to the store's native 0.25°? The
   second is cleaner and cheaper but invalidates existing checkpoints.
3. **MAM 2023** — accept the loss (recommended), or keep 0p4 with a reduced
   channel set and a second regridding path?
4. **Members** — 11 or 51 for the production pass? 11 first is the low-risk
   route; 51 only if the mean/sd difference proves to matter.
5. **Tier 2 and 3 channels** — commit now, or gate on the §5.10 check 6
   correlation result and the Tier-1 ablation?

## 7. References

- Gist: *Opening the published ECMWF IFS ensemble virtual Icechunk store* —
  <https://gist.github.com/nishadhka/3917c3d1b5391bb97c65fd98b06f6ca7>
  (anonymous access recipe, three schema eras, manifest-RAM gotcha).
- `grib-index-kerchunk/ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md`
  — per-era `.index` inventory.
- `grib-index-kerchunk/icechunk-dask/OPENING_PUBLISHED_ECMWF_ICECHUNK.md`,
  `smoke_test_published_ecmwf.py` — minimal open snippet.
- `east_africa_cgan_variable_selection_rationale.md` — driver → field → variable
  chain; §7 on the convective-precipitation gap.
- `pytorch-cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md` — the parquet limitation
  the store makes moot (§5.5).
- Bolton, D. (1980), *Mon. Wea. Rev.* 108, 1046–1053 — θe.
- Davies-Jones, R. (2008), *Mon. Wea. Rev.* 136, 2764–2785 — θe → θw inversion.
- George, J. J. (1960), *Weather Forecasting for Aeronautics* — K-index.
