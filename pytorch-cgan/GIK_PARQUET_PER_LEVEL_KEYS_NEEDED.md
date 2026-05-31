# GIK parquet must expose per-pressure-level keys — required upstream fix

**Status:** Workaround in place (surface-only pilot). Upstream fix needed
before pressure-level inputs can be used for cGAN training.

**Upstream repo:** https://github.com/icpac-igad/grib-index-kerchunk
**Affected dataset:** `E4DRR/gik-ecmwf-par` on Hugging Face
**Downstream blocked work:** PyTorch EP-cGAN port for East Africa
(`example_notebooks/pytorch_cgan/ingest_ecmwf_pytorch_cgan_variables.py`)

## TL;DR

The current GIK parquet stores **one `pl` reference per `(variable, step)`**
where the GRIB byte-range pointed at happens to be the *first* GRIB message
of that variable in the source `.grib2` file. ECMWF varies the level
ordering across steps, so under the *same key* the encoded pressure level
changes from step to step. A consumer that fetches
`step_009/u/pl/control/0.0.0` and `step_024/u/pl/control/0.0.0` gets u-wind
at two different levels — physically meaningless when stacked along a
`lead_time` axis.

The pressure-level channels needed by the PyTorch EP-cGAN (u/v/gh, paper
§2.b) are therefore unusable until the parquet exposes **per-level keys**.

## Evidence — probed on 20260301 (control member)

Same parquet key, same date, decoded GRIB header reports a different
`isobaricInhPa` for every step:

| Lead step (h) | `gh/pl` level (hPa) | `u/pl` level (hPa) | `v/pl` level (hPa) |
| ---: | ---: | ---: | ---: |
|  6 | **400** | **250** | **250** |
|  9 | 300 | 500 | 500 |
| 12 | 300 | 500 | 500 |
| 15 | **400** | 500 | 500 |
| 18 | 300 | **250** | **250** |
| 21 | **400** | **250** | **250** |
| 24 | **1000** | **250** | **250** |
| 27 | **1000** | **250** | **250** |
| 30 | **1000** | 500 | 500 |

The shift to `gh @ 1000 hPa` on steps 24/27/30 is the smoking gun — the
height field at 1000 hPa is the orographic-surface height, with values
near 0 m and negative over below-sea-level cells. Stacking that with
300/400 hPa heights (~9,000 m) produces a `gh` array with `min=59 gpm`,
`max=9,724 gpm`, `mean=5,781 gpm` — looks plausible at first glance but
encodes three different physical quantities.

Surface (`sfc`) keys are not affected by this issue.

## Why this matters for cGAN training

The Xu et al. 2026 EP-cGAN (the framework this port targets, DOI
10.1175/WAF-D-24-0199.1) uses **5 pressure-level channels** out of 11
inputs total:

- `u`, `v` at low-troposphere (paper used a single level set)
- `ub`, `vb` at the boundary layer (a second wind level)
- `gh` at mid-troposphere

These channels carry information that the surface variables do not:

- **Somali jet diagnostics** (the 850–925 hPa cross-equatorial flow that
  determines East Africa long-rains moisture transport)
- **Turkana jet** (low-level wind acceleration through the Ethiopian–
  Kenyan highland gap, controls Lake Victoria–region convection)
- **500/300 hPa trough/ridge identification** for synoptic-scale
  precipitation organisation
- **Vertical wind shear** between the two wind levels (key for EP
  organisation — squall lines, deep convection)

Surface-only models can learn rainfall climatology and basic bias
correction but lose the synoptic context that distinguishes a typical
afternoon thunderstorm from a flood-producing mesoscale convective
system. The paper's CSI improvements at the >50 mm/3 h threshold come
substantially from these pressure-level signals.

## Concrete fix — per-level keys in the GIK parquet

The parquet schema needs to emit one reference per `(var, step, level)`
tuple. The proposed key convention:

```
step_NNN/{var}/pl/{level_hPa}/{member}/0.0.0
```

For example:

```
step_009/u/pl/250/control/0.0.0
step_009/u/pl/500/control/0.0.0
step_009/u/pl/700/control/0.0.0
step_009/u/pl/850/control/0.0.0
step_009/u/pl/925/control/0.0.0
step_009/u/pl/1000/control/0.0.0
step_009/v/pl/250/control/0.0.0
...
step_009/gh/pl/300/control/0.0.0
step_009/gh/pl/500/control/0.0.0
...
```

ECMWF's ENS Open Data feed provides u, v, gh (and t, q, w) at all 13
isobaric levels per step:

```
1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa
```

so the keys naturally enumerate 13× per `(var, step)`.

### Where in the GIK code

The relevant scanning logic lives in `grib-index-kerchunk` — the function
that parses `.index` sidecars (or GRIB headers via `gribberish`) and
generates zarr-store reference keys for each message. Currently that loop
emits one key per `(var, step)` using the *first* level encountered; it
needs to include the level value in the key (or in a separate level
coordinate).

Minimum acceptable change for cGAN port:

1. Detect the `isobaricInhPa` (or equivalent `level`) value in each pl
   GRIB message header.
2. Append it to the key as a literal hPa integer (e.g. `pl/700/`).
3. Update any downstream consumers that depend on the old key shape.

Pull-through is straightforward — the byte offsets already point at the
right messages; only the key naming changes.

## What's running on the surface-only workaround

While the upstream fix is pending, the ingest script
(`ingest_ecmwf_pytorch_cgan_variables.py`) has `PRESSURE_VARS = {}` and
writes 5 channels (surface only):

| Channel | Source | Notes |
| --- | --- | --- |
| `tp` | `tp` | total precipitation (3-h accum diff) |
| `pw` | `tcwv` | precipitable water |
| `msl` | `msl` | mean sea-level pressure |
| `sp` | `sp` | surface pressure |
| `cp_proxy` | `sf` | snowfall as convective-precip proxy |

This is enough to:

- Validate the full pipeline end-to-end (parquet → S3 GRIB → Icechunk).
- Begin a *baseline* cGAN training pass on surface inputs only — a useful
  scientific data point in its own right (how much of the paper's skill
  is recoverable without the upper-air channels?).
- Quantify the cost / wall-clock characteristics for the multi-month
  fetch (per `docs/east_africa_kenya_training_plan.md` §7).

It is **not** enough to faithfully reproduce the Xu et al. paper, nor to
get the paper's CSI improvements at the >50 mm/3 h extreme threshold over
East Africa.

## Two interim fallbacks if the upstream fix is delayed

### Fallback A — bypass the parquet for pressure-level vars
Rather than fetching one byte-range per parquet key, do a single full
`.grib2` `byte-range` per step (or a small set of byte-ranges per step
guided by the index file) that captures *all* pl messages for `u, v, gh`.
Decode them, extract the levels we care about, write them out. Costs
~10–13× more bytes per pl-var per step (we read several levels we
discard) but gets us all-level access without an upstream change.

### Fallback B — write extra per-level entries client-side
Re-process the existing parquet client-side: for each `(var, step)` pl
key, peek the GRIB header to learn the level, then store the byte-range
under a `(var, step, level)`-suffixed key in a *new* parquet. Workers
still fetch one byte-range each but the level is now explicit. Pure
metadata transform — same bytes, better keying.

Both fallbacks are 1–2 days of work; the upstream fix is preferred
because it benefits every downstream user, not just the cGAN port.

## Recommended path

1. **Immediate** (today): Run the surface-only pilot through the full
   training pipeline. Document baseline skill vs IFS. (Workaround in
   place.)
2. **Within 1–2 weeks**: Upstream PR against
   `icpac-igad/grib-index-kerchunk` adding per-level keys. Coordinate
   with whoever maintains the `E4DRR/gik-ecmwf-par` regeneration.
3. **Once per-level keys land**: Restore `PRESSURE_VARS` in the ingest
   script to:

   ```python
   PRESSURE_VARS = {
       ("u",  700): "u",
       ("v",  700): "v",
       ("u",  925): "ub",
       ("v",  925): "vb",
       ("gh", 500): "gh",
   }
   ```

   Re-run the ingest, retrain.
4. **Optionally**: extend with `("t", L)`, `("q", L)`, `("w", L)` for
   future model iterations (paper variants and Harris et al. 2022 use
   these too).

## References

- Probe data captured on 2026-05-28 against the 20260301 00z parquet.
- Xu et al. (2026), *Wea. Forecasting* 41:381–401 (DOI 10.1175/WAF-D-24-0199.1)
- See `docs/east_africa_kenya_training_plan.md` §7 for the
  AWS-S3-availability framing this builds on.
- See `docs/tf_vs_pytorch_cgan_comparison.md` for why the PyTorch
  channel set is preferred over the TF tutorial set.
