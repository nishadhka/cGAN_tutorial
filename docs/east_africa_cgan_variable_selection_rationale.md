# East Africa cGAN — domain & variable-selection rationale

**Why the Kenya EP-cGAN is conditioned on a large multi-country domain, what
each large-scale rainfall driver demands of the input fields, and how that
demand is reconciled with the variables that actually exist in the ECMWF Open
Data feed on AWS S3.**

This document supplies the *connective rationale* that the two sibling docs
leave implicit:

- `east_africa_kenya_training_plan.md` §1 lists the regional drivers and §7
  lists what AWS S3 ECMWF Open Data provides — but never links a specific
  driver to the specific field needed to represent it.
- `tf_vs_pytorch_cgan_comparison.md` compares the two channel sets.

Here the chain is made explicit end to end:

> **rainfall driver → the field that diagnoses it → the ECMWF Open Data
> variable/level that carries that field → keep / proxy / drop.**

It is the design record for the channel list hard-coded in
`pytorch-cgan/ingest_ecmwf_pytorch_cgan_variables.py` (the
`SURFACE_VARS` / `PRESSURE_VARS` dicts and the `LAT_MIN…LON_MAX` box).

_Last updated 2026-06-02 — added the per-era convective-precipitation analysis
(§7) reconciling the channel set against the ECMWF era inventory; corrected the
`sf`-as-`cp`-proxy and "CAPE absent" claims._

---

## 1. The premise: Kenyan rainfall is forced from outside Kenya

A super-resolution cGAN learns `HR precip (Y) = G(LR predictors (X))`. The
generator can only condition on what is inside its receptive field. If the LR
predictor patch is cropped to Kenya, the synoptic features that *decide where
and when* East African extreme precipitation falls are simply absent from the
input — the network is asked to predict a rainfall map from a state it cannot
see.

Every one of the dominant East African rainfall controls is a **large-scale,
trans-boundary** feature whose core sits well outside Kenya:

- the **Somali (East African) low-level jet** — core over the western Indian
  Ocean / Somali coast;
- the **Turkana jet** — channelled through the Ethiopia–Kenya topographic gap;
- the **Congo air boundary** — a convergence line entering from the west;
- the **ITCZ** — migrating across the whole Horn seasonally;
- the **Indian Ocean Dipole (IOD)** signature — an ocean-basin-scale gradient;
- the **MJO** and **SW Indian Ocean tropical cyclones** — sub-seasonal,
  ocean-basin features.

The conclusion is structural, not a tuning choice: **train on a wide domain,
evaluate masked to Kenya.**

---

## 2. The conditioning domain actually used

The ingest store is built on:

| | Min | Max | Span | Grid points @0.25° |
|---|---|---|---|---|
| Latitude | **−15° S** | **+25° N** | 40° | **161** |
| Longitude | **+20° E** | **+53° E** | 33° | **133** |

(`ingest_ecmwf_pytorch_cgan_variables.py`, `LAT_MIN..LON_MAX`; 161×133 grid
confirmed against the materialised stores.) This is the wider "Full ICPAC
region" end of the two options in the training plan §1 — chosen so the
conditioning window fully *contains* the jets rather than clipping them.

### Each boundary earns its place

| Boundary | What it captures | Driver it serves |
|---|---|---|
| **South to −15° S** | SW Indian Ocean, northern Mozambique Channel, the inflow latitude of the cross-equatorial flow and SWIO cyclone tracks | Somali jet *source*, MJO/TC inflow |
| **North to +25° N** | Ethiopian highlands, Red Sea, southern Arabian Peninsula — the jet exit/recurvature region and the northern ITCZ excursion | Somali jet *exit*, ITCZ, Turkana gap northern wall |
| **East to +53° E** | Open western Indian Ocean and Somali coast — where the low-level jet is strongest and the IOD east pole expresses | Somali jet core, IOD signature, moisture source |
| **West to +20° E** | Eastern Congo Basin and the South Sudan / western Rift | Congo air boundary, westerly moisture inflow |

A Kenya-only box (roughly 34–42° E, −5–5° N) would amputate the jet cores, the
Congo inflow, and the IOD east pole — i.e. every driver in §1.

> **Padding channel.** The paper's `pad` channel (a 768 km synoptic crop
> resized to the 256 px frame) is the *cheap* way to widen synoptic context
> without enlarging the HR target. It is computed at training time from `tp`,
> not stored. For EA the relevant features (Somali jet, IOD) extend further than
> China's 768 km — testing a 1024–1536 km pad is an open item (plan §8).

---

## 3. Driver → field → ECMWF Open Data variable (the core mapping)

This is the table the other docs do not provide: for each driver, the physical
field that reveals it, and the concrete AWS S3 ECMWF Open Data variable (and
pressure level) that carries that field.

| Rainfall driver | Field that diagnoses it | ECMWF Open Data variable @ level | In ENS S3 feed? | Channel decision |
|---|---|---|---|---|
| **Somali low-level jet** | low-level wind speed/direction (cross-equatorial southerlies) | `u`,`v` @ **925 hPa** → `ub`,`vb`; `u`,`v` @ **700 hPa** → `u`,`v` | ✅ `u`,`v` at all 13 levels | **keep** (pressure-level set) |
| **Turkana jet** | low-level southeasterly acceleration through the gap | `u`,`v` @ **925/850 hPa** → `ub`,`vb` | ✅ | **keep** (same wind channels) |
| **Moisture transport / convergence** | column water content | `tcwv` → `pw` | ✅ | **keep** (surface) |
| **Congo air boundary** | mid-level flow convergence + moisture gradient | `u`,`v` @ **700 hPa** + `pw` | ✅ | **keep** |
| **ITCZ position** | mass / pressure field, mid-level height | `msl`; `gh` @ **500 hPa** | ✅ | **keep** |
| **Synoptic mass field / steering** | sea-level & surface pressure, geopotential | `msl`, `sp`, `gh@500` | ✅ | **keep** |
| **Convective instability** | CAPE | `cape` (2024 era) / `mucape` (2025–26 era) | ✅ **era-dependent** (see §7) | **keep** — primary convective channel |
| **Convective uplift** | mid-level vertical velocity | `w` @ **500/700 hPa** | ✅ all 0.25° eras | **keep** — era-stable convective proxy (see §7) |
| **Convective vs stratiform split** | convective precipitation `cp` | `cp` | ❌ **never** in open data (any era) | unavailable — substitute with CAPE + `w`, **not** `sf` (see §7) |
| **Total rainfall (the predictand context)** | total precipitation | `tp` | ✅ | **keep** (surface) |
| **Topographic forcing (Rift, highlands)** | resolved by the HR target, not LR | — (HR target store) | n/a | HR side, separate store |
| **IOD / ENSO / MJO** | *implicit* — expressed through the above fields over the wide domain | (the domain itself) | n/a | captured by domain width, not a channel |

**Reading the table:** the jets — the features the user flagged as the reason a
larger region is needed — are represented *not* by adding exotic variables but
by the **pressure-level wind channels `u, v, ub, vb`** read over the wide
domain. The domain width is what lets those same wind fields actually contain
the jet cores. IOD/ENSO/MJO need no dedicated channel: they modulate the wind,
moisture and pressure fields the model already ingests, *provided the domain is
wide enough to see the gradient.* This is the crux — **basin-scale drivers are
captured by domain extent, jets by the wind channels, and both are
free-of-charge in the open-data feed.**

---

## 4. What the AWS S3 ECMWF Open Data ENS feed actually offers

(Grounded in the per-era inventory
`grib-index-kerchunk/ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md`,
which supersedes the looser snapshot in training plan §7. **Availability is
era-dependent** — see §7 for the full per-era table.)

- **Surface fields (broadly present):** `10u, 10v, 2t, 2d, msl, sp, skt, tcw,
  tcwv, tp, ssr, ssrd, ro, ttr`; plus **`mucape`** (13-level/50r1 eras) or
  **`cape`** (49r1 9-level era); `ptype, tprate` (13-level/50r1); `sf, tcc`
  (50r1 only).
- **Pressure-level fields present:** `d, gh, q, r, t, u, v, vo, w` at 9–14
  levels depending on era (13 levels in the 2025/26 window: 1000…50 hPa).
  Note **`w` (vertical velocity) and `r` (relative humidity) are present on
  pl in every 0.25° era** — directly useful for convection.
- **Static:** `lsm` (land–sea mask).
- **Genuinely absent from open data in *every* era:** `cp` (convective
  precip), `lsp` (large-scale precip), `tciw`, `tclw`, `tcrw`, `mcc`, `lcc`,
  `hcc`. **CAPE is *not* in this list** — a CAPE-family field is published in
  every cGAN-relevant era (it just changes name across the schema break).

The selection rule for this project is therefore strict and deliberate:

> **A channel is admissible only if it can be produced at *inference* time from
> the AWS S3 ECMWF Open Data feed** — directly, or via a proxy that is applied
> *identically at training and inference* (so there is no train/serve
> distribution shift). Anything else is dropped.

This rule is what keeps the EA model operationally runnable: the very feed the
model trains on is the feed it forecasts from.

---

## 5. The selected channel set

### 5.1 Full PyTorch EP target set (11 channels)

| # | Channel | Meaning | Source variable @ level | Driver served (§3) |
|---|---|---|---|---|
| 1 | `tp` | total precipitation (3-h accum diff) | `tp` (surface) | rainfall context |
| 2 | `pad` | `tp` resized 768 km → 256 km | derived from `tp` (not stored) | synoptic context |
| 3 | `pw` | precipitable water | `tcwv` (surface) | moisture transport |
| 4 | `msl` | mean sea-level pressure | `msl` (surface) | ITCZ / mass field |
| 5 | `sp` | surface pressure | `sp` (surface) | mass field / terrain |
| 6 | `cape`/`w` | convective environment (instability + uplift) | `mucape`/`cape` (sfc) + `w`@700 (pl) | convective signal — **replaces the unavailable `cp`; `sf` is *not* used (§7)** |
| 7 | `u` | low-trop u-wind | `u` @ 700 hPa | Congo boundary, jets |
| 8 | `v` | low-trop v-wind | `v` @ 700 hPa | Congo boundary, jets |
| 9 | `ub` | 2nd-level u-wind | `u` @ 925 hPa | **Somali + Turkana jet** |
| 10 | `vb` | 2nd-level v-wind | `v` @ 925 hPa | **Somali + Turkana jet** |
| 11 | `gh` | geopotential height | `gh` @ 500 hPa | ITCZ / steering |

All 11 are available from AWS S3 (channel 2 is derived, channel 6 is a proxy).
**Net: 9 direct + 1 derived + 1 proxy, zero genuinely-missing fields.**

### 5.2 Current pilot store (5 surface channels)

What is *built today* is the surface-only subset — `tp, pw, msl, sp, cp_proxy`
— because the pressure-level winds (`u, v, ub, vb, gh`) are temporarily
**disabled at the ingest layer**, not because of any open-data gap.

> 🛑 **`cp_proxy` (= `sf` snowfall) is deprecated and should be dropped.** The
> per-era inventory (§7) shows `sf` is **not even published** in the eras
> covering MAM 2024 (49r1 9-level) and MAM 2025 + most of MAM 2026 (49r1
> 13-level) — it first appears only in 50r1 (from 2026-05-12). And where it
> *is* published it is physically ~zero over tropical East Africa. This is the
> exact cause of the empirically-observed all-NaN/all-zero `cp_proxy` in the
> built stores. Replace it with the CAPE + `w` convective channel (§7).

> ⚠️ **The jets are not yet in the store.** Channels 7–11 — the *only* channels
> that actually carry the Somali and Turkana jet signal — are disabled pending
> a GIK parquet fix: the current parquet exposes a single `pl` reference per
> `(variable, step)` while the underlying GRIB encodes a *different* hPa level
> at each lead time (e.g. `u/pl` cycles 250→500→500→… hPa across steps). Storing
> that would silently mix levels along the lead-time axis. See
> `pytorch-cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`. This is an **upstream
> parquet-construction issue, not an ECMWF Open Data limitation** — every level
> the model wants is present in the S3 feed.

So the surface pilot can prove the pipeline, but **the meteorological case in
§3 is only fully satisfied once the per-level wind channels are re-enabled.**
That re-enablement is the single highest-value data task for the EA model's
skill on jet-driven rainfall.

---

## 6. What is deliberately excluded, and why

| Field | Why excluded | Consequence |
|---|---|---|
| `cp` (convective precip) | **never** in open data (any era) | substitute the convective *environment* — CAPE/`mucape` + mid-level `w` — **not** `sf` (see §7); the HR target teaches convective rainfall structure on the output side |
| `lsp` (large-scale precip) | not in open data; ~redundant with `tp − cp` | dropped; takes the paper's set 11→10, consistent with Harris et al. 2022 |
| `sf` (snowfall) | absent in 2024/25 eras, ~zero over EA | **removed as a `cp` proxy** — invalid for this region (§7) |
| `cape`/`mucape` | **present** (era-dependent name) | **kept** as the primary convective channel (§7) — not "future", available now |
| `tciw`/`tclw`/`tcrw` (hydrometeors) | not in ENS, no clean proxy | dropped — this is the risk class the TF 14-ch set carries and the EP set avoids |
| explicit IOD/ENSO/MJO indices | not gridded fields | **captured implicitly** by the wide domain (§3) — they modulate the kept wind/moisture/pressure fields |
| SST | not in this ENS predictor feed | IOD/ENSO effect enters via the atmospheric response already in the domain |

The guiding principle: **prefer a field the open-data feed can deliver
operationally over a "physically ideal" field it cannot.** A channel that
exists only in reanalysis but not in the real-time S3 feed would make the model
untrainable-for-deployment — every excluded field above either has an
operational proxy or is captured implicitly by domain width.

---

## 7. Convective precipitation — the real gap and the era-aware options

This section answers the central question: **if ECMWF open data never carries
convective precipitation (`cp`), is it still worth using — and how do we give
the cGAN the convective signal it needs?** It is grounded in the per-era
inventory
(`grib-index-kerchunk/ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md`).

### 7.1 The TF tutorial genuinely depends on `cp`

The TF cGAN's input set (`tensorflow-dev-test/data/data.py`) is **14 fields**:

```
cape, cp, mcc, sp, ssr, t2m, tciw, tclw, tcrw, tcw, tcwv, tp, u700, v700
```

with `cp` in `accumulated_fields` alongside `tp`. So `cp` is a *first-class*
predictor in the original design — losing it is a real change, not a cosmetic
one. The question is what replaces it.

### 7.2 `sf` (snowfall) is not a valid `cp` proxy for East Africa — twice over

The ingest script (and the earlier rationale) used `sf` as a `cp` proxy. The
inventory shows this fails on two independent grounds:

1. **`sf` is not even published in the training-window eras.** It is absent
   from the **49r1 9-level** (MAM 2024) and **49r1 13-level** (MAM 2025 and
   most of MAM 2026) surface sets; it first appears only in **50r1**
   (from 2026-05-12). So for ~all of the 2024–2026 MAM data there is *no `sf`
   field to read*.
2. **Even where present, `sf` ≈ 0 over tropical EA.** Snowfall over the Horn is
   confined to a few highland pixels; the field carries essentially no
   convective information.

Together these *exactly explain* the empirical finding that `cp_proxy` came
back **all-NaN (2024/2025) and all-zero (2026)** in the built stores. `sf` is
therefore dropped, not demoted.

### 7.3 What the open data *does* offer for convection, per era

| Convective signal | 0p4-beta (pre-2024-02) | 49r1 9-level (MAM 2024) | 49r1 13-level (MAM 2025, MAM 2026≤05-12) | 50r1 (MAM 2026≥05-12) |
|---|---|---|---|---|
| `cape` (surface-parcel CAPE) | ❌ | ✅ | ❌ (replaced) | ❌ (replaced) |
| `mucape` (most-unstable CAPE) | ❌ | ❌ | ✅ | ✅ |
| `w` vertical velocity (pl, 700/500) | ❌ (no `w`) | ✅ | ✅ | ✅ |
| `r` relative humidity (pl) | ✅ | ✅ | ✅ | ✅ |
| `q` specific humidity (pl) | ✅ | ✅ | ✅ | ✅ |
| `ptype` precipitation type | ❌ | ❌ | ✅ | ✅ |
| `tprate` precip rate | ❌ | ❌ | ✅ | ✅ |
| `sf` snowfall | ❌ | ❌ | ❌ | ✅ (but ≈0 over EA) |
| `cp` convective precip | ❌ | ❌ | ❌ | ❌ |

### 7.4 The recommended substitution: condition on the convective *environment*

The cGAN does **not** need `cp` as an input to learn convective rainfall. Its
whole purpose is to *correct* the NWP's poorly-parameterised convection, so the
model's own `cp` is a biased teacher anyway. The convective rainfall *structure*
is taught on the **output side** by the HR observation target (IMERG/CHIRPS),
while the **input side** only needs to localise *where the convective
environment is*. The open data carries exactly those environment fields:

- **CAPE family — the primary convective channel.** Use `mucape` for MAM 2025
  & 2026 and `cape` for MAM 2024. This is precisely the approach of Paper 2
  (the EP-cGAN), which conditions on CAPE rather than `cp`.
- **Mid-level vertical velocity `w` @ 700/500 hPa — the era-stable proxy.**
  Present in *every* 0.25° era (2024/25/26), so it is the one convective
  predictor with no schema discontinuity. `w` is a direct dynamical signature
  of ascent/convection.
- **Moisture stack `tcwv` + `r`/`q` on pl** — instability needs moisture;
  these are available in all eras.
- **(Optional) `ptype`** — a categorical convective/stratiform flag, available
  2025 onward, usable as an auxiliary channel if the 2024 gap is acceptable.

### 7.5 The era-consistency caveat (important for a 3-season train set)

The convective channel is **not schema-stable across the MAM 2024/25/26
window**:

| Season | Era | CAPE field | `w` | `ptype` |
|---|---|---|---|---|
| MAM 2024 | 49r1 9-level | `cape` | ✅ | ❌ |
| MAM 2025 | 49r1 13-level | `mucape` | ✅ | ✅ |
| MAM 2026 | 13-level → 50r1 | `mucape` | ✅ | ✅ (+`sf`,`tcc` after 05-12) |

Three ways to handle the `cape` (2024) vs `mucape` (2025/26) break:

1. **Harmonise on `mucape` + `w`** and treat 2024's `cape` as an approximate
   stand-in (document the discontinuity). MU-CAPE ⊇ surface CAPE, so the
   distributions are close but not identical — normalise per-era.
2. **Drop 2024, train on 13-level-era data only** (2025-01-14 onward) for a
   fully consistent `mucape`-based set. Costs one MAM season (back to Scheme B
   territory — see training plan §3).
3. **Use only `w` + moisture** as the convective channels (no CAPE), which is
   the single era-stable intersection across all three seasons. Cleanest but
   drops the strongest instability predictor.

**Recommendation:** option 1 — `mucape` + `w`@700 as the convective pair, with
per-era normalisation, keeping all three MAM seasons.

### 7.6 So is ECMWF open data still worth it without `cp`? — Yes

Tangible benefits that survive the `cp` gap:

1. **Train = infer.** It *is* the operational feed (free, real-time, global,
   no licence). Training on it eliminates the domain shift that an ERA5- or
   reanalysis-trained model would hit operationally (the core argument of
   `faq-cgan-training-gefs.md`).
2. **The convective environment is fully represented** — CAPE/`mucape`, `w`,
   moisture, low-level convergence — which is what the network actually needs
   to localise convection. `cp` itself adds a *parameterised, biased* field on
   top of these; its absence is a modest loss, not a blocker.
3. **The HR target carries the convective rainfall morphology** — the cGAN
   learns convective structure from observations, not from the NWP's `cp`.
4. **Ensemble spread** (50–51 members) gives calibrated uncertainty for free.

The honest residual cost: no explicit convective/stratiform partition on the
input. That is mitigated by CAPE + `w` (+ optional `ptype` from 2025), and by
letting the observation target do the convective-structure learning.

---

## 8. Reflection vs the TensorFlow tutorial set

The legacy TF tutorial conditions on a larger, less-portable 14-field list
(`cape, cp, mcc, sp, ssr, t2m, tciw, tclw, tcrw, tcw, tcwv, tp, u700, v700`).
Mapped onto the AWS S3 feed:

| Set | Direct from open data | Proxy / substitute | Genuinely missing |
|---|---|---|---|
| TF tutorial (14 ch) | 7 (`sp, ssr, t2m, tcw, tcwv, tp, u700/v700`) | 2 (`cape`/`mucape` direct; `mcc`←`tcc`) | **4** (`cp`, `tciw`, `tclw`, `tcrw`) |
| **PyTorch EP (11 ch)** | **9** | **1** (CAPE direct; `cp`→CAPE+`w`) | **2** (`cp`, `lsp`) |

The PyTorch EP set is the **more operationally portable** choice: it avoids the
three unhandled hydrometeor channels (`tciw/tclw/tcrw`) the TF set carries with
no clean proxy, and it handles the `cp` gap honestly via the convective
environment rather than a degenerate `sf` proxy. For a model that must run from
the same open-data feed it trains on, that portability is the deciding factor —
and it is *why the EP channel list is the basis for the EA build* rather than
the heavier TF set.

---

## 9. Summary

1. **Domain is dictated by physics, not preference.** Kenyan EP is forced by
   trans-boundary jets and basin-scale modes; the 161×133 / −15–25 N, 20–53 E
   box is sized to *contain* the Somali jet, Turkana jet, Congo boundary, ITCZ
   and IOD pole. Train wide, evaluate on Kenya.
2. **The jets are represented by pressure-level winds (`u, v, ub, vb`) read
   over that wide domain** — no exotic variable needed; the domain width is
   what gives those wind fields the jet cores.
3. **Basin-scale modes (IOD/ENSO/MJO) need no dedicated channel** — they
   modulate the kept fields and are captured implicitly by domain extent.
4. **Convective precip (`cp`) is never in the open data — but that is not a
   blocker.** Condition on the convective *environment* instead: CAPE/`mucape`
   (era-dependent) + mid-level vertical velocity `w`@700 (era-stable) +
   moisture. The HR observation target teaches the convective rainfall
   structure on the output side. See §7.
5. **`sf` (snowfall) is dropped as a `cp` proxy** — it is absent from the
   2024/2025 eras and ≈0 over EA, which is exactly why the built `cp_proxy`
   channel came back all-NaN/all-zero.
6. **The CAPE field changes name across the schema break** — `cape` (MAM 2024,
   49r1 9-level) vs `mucape` (MAM 2025/26, 13-level/50r1). Harmonise on
   `mucape` + `w` with per-era normalisation to keep all three MAM seasons (§7.5).
7. **Every kept channel is reachable from the AWS S3 ECMWF Open Data feed.**
   The remaining genuinely-missing fields (`cp`, `lsp`, hydrometeors) are
   handled by substitution or dropped — this is *why open data is still the
   right feed* (train = infer; §7.6).
8. **The current store is surface-only.** The jet-carrying wind channels are
   blocked by a GIK parquet per-level-key issue, *not* by data availability —
   re-enabling them (see `GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`), and adding the
   CAPE+`w` convective channel while dropping `cp_proxy`, are the gating data
   tasks for the model to exploit the larger region it was built for.

## 10. References

- `grib-index-kerchunk/ecmwf/docs/2026-06-02-ecmwf-era-variable-inventory.md`
  — authoritative per-era variable & pressure-level inventory of ECMWF open
  data; the source for §4 and §7 (CAPE/`sf`/`w`/`ptype` availability per era).
- `tensorflow-dev-test/data/data.py` — the TF tutorial's 14-field input set
  (`all_fcst_fields`), including `cp`, referenced in §7.1 and §8.
- `east_africa_kenya_training_plan.md` — domain options (§1), AWS S3
  availability (§7), GPU/dataset strategy.
- `tf_vs_pytorch_cgan_comparison.md` — TF vs PyTorch channel comparison.
- `pytorch-cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md` — the per-level-key fix
  that re-enables the wind channels.
- `pytorch-cgan/ingest_ecmwf_pytorch_cgan_variables.py` — the implemented
  domain box and channel dicts this rationale documents.
- Xu et al. (2026), *Wea. Forecasting* 41, 381–401, DOI
  10.1175/WAF-D-24-0199.1 — the EP-cGAN method and its channel set.
