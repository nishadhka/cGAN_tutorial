# Request to GIK maintainers — per-level pressure keys + parquet back-fill

Two related asks from the East Africa cGAN training work, both blocking
or constraining the same downstream pipeline.

- **Upstream repo:** https://github.com/icpac-igad/grib-index-kerchunk
- **Upstream dataset:** `E4DRR/gik-ecmwf-par` on Hugging Face
- **Downstream:** `cGAN_tutorial/example_notebooks/pytorch_cgan/` —
  ECMWF IFS ENS inputs for the PyTorch EP-cGAN (Xu et al. 2026, DOI
  10.1175/WAF-D-24-0199.1) training on the extended East Africa domain
  (20–53°E, 15°S–25°N).

---

## Request 1 — per-pressure-level keys for `u/v/gh` (and ideally `t/q/w`)

### Symptom
The parquet currently emits **one `pl` reference per `(variable, step)`**
keyed as `step_NNN/{var}/pl/{member}/0.0.0`. The underlying byte-range
points at whatever GRIB message is *first* in the source `.grib2` file
for that variable at that step — which is a different isobaric level
depending on step.

### Evidence (probed against `20260301-00z-control.parquet`)

Same parquet key, decoded GRIB header reports a different `isobaricInhPa`
for each lead-time step:

| Lead step (h) | `gh/pl` level | `u/pl` level | `v/pl` level |
| ---: | ---: | ---: | ---: |
|  6 | **400** | **250** | **250** |
|  9 | 300 | 500 | 500 |
| 12 | 300 | 500 | 500 |
| 15 | **400** | 500 | 500 |
| 18 | 300 | **250** | **250** |
| 21 | **400** | **250** | **250** |
| 24 | **1000** | **250** | **250** |
| 27 | **1000** | **250** | **250** |
| 30 | **1000** | **250** | **250** |

Surface (`sfc`) keys are unaffected.

### Why it matters

A downstream consumer that stacks `step_006/u/pl/.../0.0.0` …
`step_030/u/pl/.../0.0.0` along a `lead_time` axis gets a "u-wind" array
that mixes 250 / 500 hPa — physically meaningless. The same is true for
`gh` (which spans 300 / 400 / 1000 hPa, where 1000 hPa is effectively
the orographic-surface height).

For the cGAN: the Xu et al. paper's input set uses **5 pressure-level
channels out of 11 inputs total** (`u`, `v` at low-trop and again at
boundary-layer, plus `gh`). Without per-level keys we cannot reproduce
the paper's channel set. The Somali jet (850–925 hPa cross-equatorial
flow controlling East-Africa long-rains moisture), Turkana jet (Ethio-
pian–Kenyan highland gap, controls Lake-Victoria-region convection),
vertical wind shear, and 500-hPa trough/ridge organisation all live in
those pressure-level channels.

### Proposed fix

Include the level value in the key, e.g.:

```
step_NNN/{var}/pl/{level_hPa}/{member}/0.0.0
```

Concrete examples:

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

ECMWF ENS Open Data provides `u, v, gh, t, q, w` at all 13 isobaric
levels per step (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150,
100, 50 hPa). So the keys naturally enumerate 13× per `(var, step)`.

### Where to change

The scanner that walks the `.index` sidecars (or GRIB headers via
gribberish) and generates zarr reference keys for each message. The byte
offsets already point at the right messages — only the key naming changes.

Minimum acceptable change:
1. Read `isobaricInhPa` (or equivalent `level`) from each pl GRIB header.
2. Append it to the key as a literal hPa integer (`pl/700/`).
3. Optionally also expose `t/pl`, `q/pl`, `w/pl` per-level — same
   variables are already in the source `.grib2`.

### Backwards compatibility note

Downstream consumers currently fetching `step_NNN/{var}/pl/{member}/0.0.0`
would need updates. Since those keys are physically incoherent
already, breakage is the correct outcome.

Long-form analysis in `cGAN_tutorial/example_notebooks/pytorch_cgan/GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md`.

---

## Request 2 — back-fill the HF parquets for 2026-04-08 → 2026-05-27

### Symptom
HuggingFace dataset `E4DRR/gik-ecmwf-par` is missing daily parquets from
**2026-04-08 onward**. URLs like

```
https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/run_par_ecmwf/2026/04/20260408/00z/2026040800z-control.parquet
```

return HTTP 404, while the equivalent for 20260407 and earlier resolves
normally. The underlying ECMWF `.grib2` data is available on
`s3://ecmwf-forecasts/` for all those dates (today is 2026-05-28).

### Evidence
Running our `ingest_ecmwf_pytorch_cgan_variables.py fill` over
2026-03-01 → 2026-05-31 produced 38 successful dates (Mar 1 → Apr 7)
and 54 dates failed with HTTP 404 on parquet read. Failed list
contiguous from 20260408 to 20260531.

### Why it matters
We are training the cGAN on **MAM 2024, 2025, 2026** — losing 54 days
of MAM 2026 is losing the entirety of the peak long-rains period over
East Africa (mid-April to late-May). 38 days is enough for a pilot but
not for the production training pass.

### Ask
Run whatever pipeline produces the HF parquets against 20260408 →
present, and put it on a daily cron so this gap doesn't recur. If the
pipeline is non-trivial to back-fill, even just rolling forward from
today and accepting the historical gap would be useful.

### Bonus ask
A status endpoint or manifest file (`E4DRR/gik-ecmwf-par/.../available_dates.json`)
listing which dates have parquets uploaded would let downstream
consumers fail fast and skip cleanly, rather than hitting 404s per
member fetch.

---

## Sign-off

Happy to help with either request — can supply test fixtures, run the
new keys through our cGAN ingest pipeline end-to-end to validate, or
take a first pass at the per-level-key PR against
`icpac-igad/grib-index-kerchunk` if pointed at the relevant module.

— ICPAC GIK / PyTorch EP-cGAN port
