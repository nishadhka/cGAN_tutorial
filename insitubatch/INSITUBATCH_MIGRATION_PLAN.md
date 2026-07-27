# Migrating the cGAN training pipeline to insitubatch + Zarr — direction doc

**Verdict: yes, feasible.** The current pipeline (netCDF sources -> cropped,
class-balanced, gzip `.tfrecords` -> `tf.data.TFRecordDataset`) maps cleanly
onto [insitubatch](https://github.com/emfdavid/insitubatch)'s model of
several co-batched variables sharing one sample axis, chunk-aligned splits,
and cheap per-batch transforms. Two real gaps need a deliberate workaround
(rain-class-weighted sampling, static constants with no sample axis) — not
blockers, just decisions to make up front. Nothing below has been built yet;
this is the plan.

Source inspected: `github.com/emfdavid/insitubatch` (cloned and read locally —
README, DESIGN.md, `src/insitubatch/{source,store,split,types}.py`, and the
`examples/advection` + `examples/microscopy` reference pipelines).

---

## 0. What insitubatch actually is

A data-loader that reads training samples **directly out of a Zarr store**
(local `file://` or cloud `s3://`/`gs://`) with one async event loop instead
of PyTorch-style worker processes. It reads each *stored chunk* once,
decodes it into a shared `ChunkPool` (cache + assembly buffer), and streams
shuffled, split-aware batches off it. Key primitives:

| Primitive | What it does |
|---|---|
| `obstore_store(url)` | Opens a Zarr v3 `Store` for a `file://`/`s3://`/`gs://` URL |
| `open_geometries(store, variables=[...])` | Reads shape/chunks/dtype per array from Zarr metadata -> `{name: ArrayGeometry}` |
| `split_by_chunk(geom, fractions=(.8,.1,.1), contiguous=True)` | Chunk-aligned train/val/test `SplitManifest` (contiguous blocks for time series, to avoid leakage) |
| `InSituDataset(store, manifest, geometries=..., batch_size=...)` | The dataset; iterate `.train` / `.val` / `.test` / `.all` |
| `chunk_transform` | Runs once per decoded chunk, **cached** — scaling, unit conversion, dtype cast |
| `batch_transform` | Runs once per assembled batch, **uncached** — cross-variable derived fields, per-sample random augmentation |
| `as_tf_dataset(ds.view)` / `as_torch(...)` / `to_jax(...)` | Framework handoff (TF takes one CPU copy; torch/JAX are zero-copy DLPack) |

Variables sharing a dataset only need the **same sample-axis length** — they
can be chunked completely differently (e.g. microscopy example: `raw` is
Z-chunk 1, its `mask` label is Z-chunk 30) and carry different channel
counts. That's exactly our shape: forecast stack, truth, and (optionally)
per-day climatology all indexed by the same **date** axis, with different
channel depths.

Requires Python >=3.12 — the GPU server's `cgan_env` is already 3.12, so no
interpreter change needed. Core install is `numpy`/`zarr>=3`/`xarray`/`obstore`;
add `insitubatch[tf]` for the TensorFlow adapter, since `main.py`/`train.py`
are TF-based.

---

## 1. Current pipeline, for contrast

| Stage | File | What happens |
|---|---|---|
| Truth / forecast / constants read | `data.py` | Per-day netCDF: `~/RFE/{yr}/{date}.nc` (`precipitation`), `~/IFS_training/{yr}/{field}.nc` (`{field}_mean`/`{field}_sd`), `cGAN_data/elev.nc`, `lsm.nc`, `RFE_climatology_meansd_doy.nc` |
| Patch + class write | `tfrecords_generator.write_data()` | For each day: crops random 128x128 patches, sorts into 4 rain-intensity bins, writes GZIP `.tfrecords` — one file per `(year, class)` |
| Read + mix | `tfrecords_generator.create_mixed_dataset()` | `tf.data.TFRecordDataset(..., compression_type="GZIP")` per class, `Dataset.sample_from_datasets(datasets, weights=[.4,.3,.2,.1])`, `.batch()` |

The crop and the class split are **baked in at write time** — changing the
crop size, the class weights' underlying pool, or the field set means
regenerating the tfrecords from scratch (see `RUN08_NOCAPE_STEPS.md` for how
disruptive that already is for a field-set change alone).

---

## 2. Target shape with insitubatch

One Zarr v3 store at `/tank/projects/cGAN/zarr/run11_clim_meansd/` (`/tank/projects`
has 2.2T free vs. 45G on `/home/ezra` — store datasets there, not under `$HOME`), with:

| Array | Shape | Notes |
|---|---|---|
| `fcst` | `(n_days, 384, 352, 26)` f32 | mean+sd stack for the 13 IFS fields (today's `_F13`), same normalisation as `load_fcst`/`FCSTNorm2018.pkl` |
| `truth` | `(n_days, 384, 352, 1)` f32 | RFE2 daily accumulation, same `log10(1+x)` transform as `load_truth_and_mask` |
| `rain_class` | `(n_days,)` i8 | Per-day rain-intensity bin (quantile-derived, ~balanced by construction); read as an ordinary co-batched variable and consumed by a Phase 3 `batch_transform`, **not** used to split the store (see below) |

Chunked along the day axis (e.g. 32 days/chunk) — analogous to today's
`shuffle_size=64` in spirit, tune once real IO is measured.

**Kept outside the Zarr store, not on the sample axis:** `elev.nc`, `lsm.nc`,
and the climatology mean/sd. These are static or day-of-year-indexed, not
per-training-sample — load them once at process start exactly as
`load_hires_constants()`/`_load_climatology()` do today, and inject them via
a `batch_transform` (concatenate onto the forecast stack, gather climatology
by the batch's day-of-year). This avoids replicating a static 2-channel
array `n_days` times just to satisfy a "shared sample axis" requirement it
doesn't actually need.

**Cropping becomes a `batch_transform`, not a write-time step.** Store full
`(384, 352)` images; take a fresh random 128x128 crop per sample **per
epoch** in the transform. This is a strict improvement over today's fixed,
pre-cropped patch set — the model sees new crops every epoch instead of a
frozen sample of them — and removes the "regenerate tfrecords to change crop
size" pain point.

**Rain-class weighting has no built-in equivalent** to
`Dataset.sample_from_datasets(weights=...)`, and — a real finding from
building Phase 2, not just a theoretical gap — **it cannot be a per-class
`SplitManifest`.** `SplitManifest`/`split_by_chunk` assign **whole chunks** to
a split, but `rain_class` is a **per-day** label; a 32-day chunk mixes all 4
classes roughly evenly (they're close to i.i.d. day-to-day), so there is no
clean way to bucket whole chunks by dominant class without either shrinking
`day_chunk` to 1 (killing the chunk-amortization benefit insitubatch is built
around) or physically re-sorting the store by class (losing chronological
order, complicating the climatology-by-day-of-year gather). Revised plan:
carry `rain_class` as an ordinary co-batched Zarr variable (sample-axis
aligned, like `fcst`/`truth`), request an **oversampled** batch from the
loader (e.g. `batch_size=32` for a target of 8), and let a `batch_transform`
do weighted-without-replacement selection down to the target class mixture
`[.4,.3,.2,.1]` using each batch's own `rain_class` values. Moved to Phase 3
(it's a transform, not a split); Phase 2 only builds the train/val/test
manifest below.

**Framework handoff:** use `as_tf_dataset(ds.train)` to keep
`setupmodel.py`/`train.py` otherwise unchanged — note in the README that
TF's DLPack path is unreliable, so `as_tf_dataset` takes one CPU copy
(torch/JAX would be zero-copy instead). Acceptable since this project is
TF-only; worth remembering if throughput doesn't meet expectations.

---

## 3. Migration phases

### Phase 0 — Spike (no GPU time spent) — **done**
- Installed core `insitubatch` (no `[tf]` extra — `cgan_env` already has TF
  2.21.0; re-downloading it into a scratch venv would have been wasteful)
  into a scratch venv first, confirmed it imports cleanly.
- Ran insitubatch's own `examples/advection` end to end against a synthetic
  store — `InSituDataset`/`split_by_chunk` iterate `.train`/`.val` batches
  correctly, `target` is the shifted view as documented.

### Phase 1 — Converter script — **done**
- `dsrnngan/write_zarr.py` writes `fcst`/`truth`/`mask`/`rain_class` full
  `(384, 352, C)` images per day (no crop, no class split) to Zarr, reusing
  `data.py`/`data_generator.py` unchanged — only the sink changed, from
  `TFRecordWriter` to Zarr array assignment.
- Installed core `insitubatch`+`zarr`+`obstore` into `cgan_env` (reused its
  existing numpy/xarray — no duplicate TF download).
- Ran the **full 4-year conversion** (`train_years=[2018..2021]`, 1,461
  days) to `/tank/projects/cGAN/zarr/run11_clim_meansd/`: 19G on disk, ~2.4
  days/s, 607.5s total. Validated: shapes/chunks read back correctly via
  `open_geometries`, 10 random days spot-checked all-finite, and a direct
  parity check against `data_generator.DataGenerator` matched bit-for-bit
  (`np.allclose` True on both `fcst` and `truth`).
- **Fix applied post-hoc:** the first `rain_class` pass reused
  `write_data()`'s per-*patch*-tuned bins `[0.0059, 0.0362, 0.0761]` applied
  to a whole-image mean instead — skewed to 0/125/612/724 across classes
  0-3 (a full-domain mean is far less likely to sit near zero than a small
  patch is). Replaced with quantile edges derived from the store's own
  daily-mean distribution (`_quantile_classes` in `write_zarr.py`); rebinned
  the already-written store in place via `write_zarr.py --rebin-only`
  (reads `truth`, no netCDF re-read) — now 365/365/365/366, balanced by
  construction. `write_zarr()` itself now does this in a deferred second
  pass so any future full rebuild gets it right the first time.

### Phase 2 — Splits + geometry — **done**
- `dsrnngan/zarr_splits.py`: `build_year_split_manifest()` builds a
  chunk-aligned `SplitManifest` from named years instead of
  `split_by_chunk`'s automatic fractions — a chunk goes to a split if *any*
  date inside it falls in that split's years (same overlap-inclusive
  convention insitubatch's own `sample_range` uses for boundary chunks).
  Verified against the real store: `train` = all 46 chunks / 1,461 samples
  (`train_years` is the whole store), `val` = chunks 22-34 / 416 samples
  (overlap-covers `val_years=[2020]` plus its boundary spillover into
  2019/2021), `test` = empty (no test year configured today) — this
  reproduces the tfrecords pipeline's existing non-held-out validation
  exactly, not a redesign.
- **Real finding, not just a theoretical gap:** the original plan for
  rain-class weighting (one `SplitManifest` per class, filtered by
  `rain_class`) doesn't work — `SplitManifest` assigns **whole chunks**, but
  `rain_class` is a **per-day** label, and a 32-day chunk mixes all 4
  classes roughly evenly. Redirected to a Phase 3 `batch_transform` instead
  (oversample + weighted-without-replacement subselect) — see Section 2.

### Phase 3 — Transforms — **done**
- `dsrnngan/zarr_transforms.py`: `CropConstantsClassBalance`, one
  `batch_transform` doing all three jobs from Section 2: random 128x128 crop
  (same window across `fcst`/`truth`/`mask`/constants), static
  elev+lsm injection (loaded once via `load_hires_constants()`), and the
  oversample + weighted-without-replacement rain-class resample (request
  `batch_size=32` upstream for `target_batch_size=8`, matching
  `config.yaml`'s `training_weights=[.4,.3,.2,.1]`). Output keys
  (`lo_res_inputs`/`hi_res_inputs`/`output`/`mask`) match
  `data_generator.DataGenerator`'s nested dict exactly, flattened.
- **No climatology gather needed here** — corrected from the original draft
  of this section: Phase 1 reused `data_generator.DataGenerator` unchanged,
  and `load_fcst_stack` already appends the climatology mean/sd channels per
  date at *write* time (that's why `fcst` has 28 channels, not 26). This
  transform only handles what genuinely isn't in the store: crop, static
  constants, class resample.
- Validated against the real store: output shapes correct
  (`(8,128,128,28)`/`(8,128,128,2)`/`(8,128,128,1)`/`(8,128,128)`); observed
  class shares over 100 batches `[.372,.253,.25,.125]` vs. target
  `[.4,.3,.2,.1]` (reasonable given the oversample-then-backfill mechanics);
  a `sliding_window_view` search across 20 batches / 160 output rows
  confirmed **every** row is a genuine crop of its claimed day, cross-checked
  against `fcst` channel 0 and the static constants at the same window
  (`0 mismatches`, clean exit) — an earlier partial run flagged 1/160 as a
  false lead (a coincidental duplicate window match in a flat/dry region,
  where the verification search itself, not the transform, was ambiguous);
  the full rerun cleared it.

### Phase 4 — Parity check — **done**
- `dsrnngan/parity_check.py`: rather than regenerating tfrecords, checked
  against the **real, already-written run11 production tfrecords**
  (`/home/ezra/rfe_tfrecords/run11_clim_meansd/`, 19G — what run11 actually
  trained on). A tfrecords patch carries no date/offset metadata
  (`write_data()` picked random, unseeded crop locations), so each patch's
  origin was *located* by content: an exact-value prefilter across every day
  of its year narrows candidates to (usually) one, confirmed with a full
  `np.array_equal`; then `fcst` and constants are cross-checked at that same
  located window.
- Checked 3 patches from each of the 4 class files for 2018 (12 total): 11/12
  fully verified (`fcst_match=True`, `const_match=True` at the located
  window). The one exception (`2018_1.0.tfrecords` patch 2, class 0 = the
  driest bin) is not a pipeline discrepancy — that patch is **completely
  flat** (`min=max=0.0`, `std=0.0`), and an all-zero 128x128 window occurs at
  **300,666 different locations** across 2018 alone; content-based location
  can't disambiguate an all-zero patch from any other, so `fcst` correctly
  fails to match at whatever arbitrary all-zero window the search happened to
  land on. Class 0's other two checked patches (both non-degenerate,
  `std>0`) matched cleanly, so every rain class has at least one genuine,
  unambiguous, fully-verified match — this is a limitation of a
  content-search verification method against literally-constant data, not a
  gap in the new pipeline.

### Phase 5 — Wire into `setupdata.py` behind a switch — **done**
- `setupdata.py`: `CGAN_DATA_BACKEND=zarr|tfrecords` env var (default
  `tfrecords`, mirroring `data.py`'s `CGAN_FIELD_SET` pattern) picks
  `setup_batch_gen_zarr` or the untouched `setup_batch_gen` inside
  `setup_data()`. **No tfrecords code was modified** —
  `tfrecords_generator.py`/`write_data()`/`create_mixed_dataset()` are
  completely unchanged; this only adds a new branch. Validation
  (`setup_full_image_dataset`) needed no change either — it already reads
  netCDF directly via `data_generator.DataGenerator`, never tfrecords.
- **Real integration snag found, not a theoretical one:** insitubatch's own
  `as_tf_dataset` infers its `output_signature` from the *source*
  geometries (`fcst`/`truth`/`mask`/`rain_class` at full 384x352
  resolution) — but `CropConstantsClassBalance` renames and reshapes
  everything (crop to 128x128, `fcst`→`lo_res_inputs`, drops `rain_class`
  after consuming it), so `as_tf_dataset`'s inferred signature would be
  wrong. Wrote `zarr_tf_dataset.py`'s `as_cgan_tf_dataset` instead — a
  direct ~10-line `tf.data.Dataset.from_generator` call declaring the
  *transformed* signature — rather than patching insitubatch's inference to
  see through an arbitrary renaming `batch_transform`.
- Validated exactly as `train.py` consumes it
  (`batch_gen_train.take(1).as_numpy_iterator()`): yields a genuine
  `tf.data.Dataset` (`_PrefetchDataset`), `inputs={lo_res_inputs (8,128,128,28)
  f32, hi_res_inputs (8,128,128,2) f32}`, `outputs={output (8,128,128,1) f32,
  mask (8,128,128) bool}` — the exact nested shape/key structure
  `DataGenerator` already produces. Confirmed the default (no env var) path
  is byte-identical to before Phase 5: `CGAN_DATA_BACKEND` defaults to
  `"tfrecords"`, and that branch calls the original, untouched
  `setup_batch_gen`.
- `autocoarsen` is intentionally **not** supported on the zarr backend
  (raises `NotImplementedError` rather than silently ignoring it) — it was
  untested/unused on the tfrecords path too
  (`data_generator.DataGenerator` asserts `autocoarsen is False`), so this
  isn't a regression, just an explicit scope boundary.

### Phase 6 — Benchmark before committing
- Compare wall-clock time/checkpoint between Zarr+insitubatch and the
  existing GZIP tfrecords pipeline **on the actual RTX 5000 box**.
  insitubatch's headline win (~8x, per its README) is measured against cloud
  object storage; on local NVMe with TFRecords already prefetching in
  parallel, the win may be smaller or a wash. Measure, don't assume.

### Phase 7 — Cut over
- Once a full `num_samples=204800` run on the new path matches run06/run11
  CRPS, retire the tfrecords path (or keep it as a documented fallback).

---

## 4. Open risks to flag explicitly

- No built-in weighted multi-source sampling, **and it can't be a per-class
  manifest** (splits are whole-chunk, `rain_class` is per-day) — needs the
  oversample + `batch_transform` subselect workaround (Section 2), still to
  be implemented in Phase 3.
- Static constants have no natural sample axis — handled outside the Zarr
  store via `batch_transform`, not as a stored array.
- `as_tf_dataset` takes one CPU copy (TF's DLPack path is unreliable per
  insitubatch's own docs) — some throughput cost vs. the zero-copy
  torch/JAX adapters; this project is TF-only, so it's the only option, but
  worth knowing if the benchmark in Phase 6 disappoints.
- `split_by_chunk`'s fractional, chunk-granular split doesn't natively
  express "these specific years are validation" — plan to hand-build the
  manifest or use `sample_range` rather than fight the default splitter.
- insitubatch is alpha (README: "alpha, but validated on real cloud IO") —
  pin a version and re-read the CHANGELOG before upgrading mid-project.
