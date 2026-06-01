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

_Last updated 2026-06-01._

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
| **Convective instability** | CAPE | `cape` | ❌ (not in ENS) → **MU-CAPE proxy** | proxy (`cape_proxy`, future) |
| **Convective vs stratiform split** | convective precipitation `cp` | `cp` | ❌ (not in ENS) → **snowfall `sf` proxy** | proxy (`cp_proxy`) |
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

(Grounded in `cgan_ecmwf/stream_cgan_variables.py` and training plan §7.)

- **Surface fields present:** `10u, 10v, 2t, 2d, msl, sp, skt, tcw, tcwv, tp,
  ssr, ssrd, sf, ro, tcc`.
- **Pressure-level fields present:** `gh, t, u, v, w, q` at **13 levels**
  (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa).
- **Static:** `lsm` (land–sea mask).
- **Genuinely absent from the ENS product:** `cp`, `lsp`, `cape`, `tciw`,
  `tclw`, `tcrw`, `mcc`, `lcc`, `hcc`.

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
| 6 | `cp_proxy` | convective-precip proxy | `sf` snowfall (surface) | convective signal |
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
| `cp` (convective precip) | not in ENS open data | use `sf` snowfall proxy at train+infer (no shift); → `cp_proxy` |
| `lsp` (large-scale precip) | not in ENS; ~redundant with `tp − cp` | dropped; takes the paper's set 11→10, consistent with Harris et al. 2022 |
| `cape` | not in ENS | MU-CAPE proxy (`cape_proxy`), future channel; same convention as TF inference |
| `tciw`/`tclw`/`tcrw` (hydrometeors) | not in ENS, no clean proxy | dropped — this is the risk class the TF 14-ch set carries and the EP set avoids |
| explicit IOD/ENSO/MJO indices | not gridded fields | **captured implicitly** by the wide domain (§3) — they modulate the kept wind/moisture/pressure fields |
| SST | not in this ENS predictor feed | IOD/ENSO effect enters via the atmospheric response already in the domain |

The guiding principle: **prefer a field the open-data feed can deliver
operationally over a "physically ideal" field it cannot.** A channel that
exists only in reanalysis but not in the real-time S3 feed would make the model
untrainable-for-deployment — every excluded field above either has an
operational proxy or is captured implicitly by domain width.

---

## 7. Reflection vs the TensorFlow tutorial set

The legacy TF tutorial conditions on a larger, less-portable channel list (≈14
fields incl. `2t, ssr, ssrd, tcw, tcc, ro`). Mapped onto the AWS S3 feed:

| Set | Direct from S3 | Proxy | Genuinely missing |
|---|---|---|---|
| TF tutorial (~14 ch) | 8 | 3 (`mucape→cape`, `sf→cp`, `tcc→mcc`) | 3 (`tciw`, `tclw`, `tcrw`) |
| **PyTorch EP (11 ch)** | **9** | **1 (`mucape→cape`)** | **2 (`cp`, `lsp`)** |

The PyTorch EP set is the **more operationally portable** choice: fewer
proxies, and — critically — **no unhandled hydrometeor channels**. For a model
that must run from the same open-data feed it trains on, that portability is the
deciding factor, and it is *why the EP channel list is the basis for the EA
build* rather than the heavier TF set.

---

## 8. Summary

1. **Domain is dictated by physics, not preference.** Kenyan EP is forced by
   trans-boundary jets and basin-scale modes; the 161×133 / −15–25 N, 20–53 E
   box is sized to *contain* the Somali jet, Turkana jet, Congo boundary, ITCZ
   and IOD pole. Train wide, evaluate on Kenya.
2. **The jets are represented by pressure-level winds (`u, v, ub, vb`) read
   over that wide domain** — no exotic variable needed; the domain width is
   what gives those wind fields the jet cores.
3. **Basin-scale modes (IOD/ENSO/MJO) need no dedicated channel** — they
   modulate the kept fields and are captured implicitly by domain extent.
4. **Every kept channel is reachable from the AWS S3 ECMWF Open Data feed** —
   9 direct, 1 derived (`pad`), 1 proxy (`cp_proxy`); `cape` via MU-CAPE later.
   `cp`, `lsp`, hydrometeors are dropped because the feed cannot serve them.
5. **The current store is surface-only.** The jet-carrying wind channels are
   blocked by a GIK parquet per-level-key issue, *not* by data availability —
   re-enabling them (see `GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`) is the gating
   task for the model to actually exploit the larger region it was built for.

## 9. References

- `east_africa_kenya_training_plan.md` — domain options (§1), AWS S3
  availability (§7), GPU/dataset strategy.
- `tf_vs_pytorch_cgan_comparison.md` — TF vs PyTorch channel comparison.
- `pytorch-cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md` — the per-level-key fix
  that re-enables the wind channels.
- `pytorch-cgan/ingest_ecmwf_pytorch_cgan_variables.py` — the implemented
  domain box and channel dicts this rationale documents.
- Xu et al. (2026), *Wea. Forecasting* 41, 381–401, DOI
  10.1175/WAF-D-24-0199.1 — the EP-cGAN method and its channel set.
