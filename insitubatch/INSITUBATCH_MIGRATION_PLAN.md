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
| `rain_class` | `(n_days,)` i8 *(optional)* | Precomputed dominant rain bin per day, to drive class-weighted sampling without physically splitting the store |

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
`Dataset.sample_from_datasets(weights=...)`. Plan: build one `SplitManifest`
per class (filtering days by the precomputed `rain_class` array) -> one
`InSituDataset` view per class -> interleave batches in the Python training
loop with `np.random.choice(4, p=[.4,.3,.2,.1])` per step. Light, no
insitubatch changes needed.

**Framework handoff:** use `as_tf_dataset(ds.train)` to keep
`setupmodel.py`/`train.py` otherwise unchanged — note in the README that
TF's DLPack path is unreliable, so `as_tf_dataset` takes one CPU copy
(torch/JAX would be zero-copy instead). Acceptable since this project is
TF-only; worth remembering if throughput doesn't meet expectations.

---

## 3. Migration phases

### Phase 0 — Spike (no GPU time spent)
- `uv sync --extra tf` (or `pip install "insitubatch[tf]"`) into a **scratch
  venv** first, not `cgan_env` — confirm it installs cleanly (obstore pulls
  Rust wheels; should be fine on Linux x86_64, but verify before touching
  the real env).
- Run insitubatch's own `examples/advection` end to end against a synthetic
  store to get a feel for `InSituDataset`/`split_by_chunk`/transforms before
  touching real data.

### Phase 1 — Converter script
- New `dsrnngan/write_zarr.py`, mirroring `write_data()`'s date iteration in
  `tfrecords_generator.py`, but writing full `(384, 352, C)` arrays into the
  two Zarr arrays instead of cropped per-class tfrecords.
- Reuse `load_fcst`/`load_truth_and_mask`/`FCSTNorm2018.pkl` unchanged — only
  the sink changes (Zarr array assignment instead of `TFRecordWriter`).

### Phase 2 — Splits + geometry
- `open_geometries()` per array.
- `split_by_chunk`'s automatic 0.8/0.1/0.1 fractional split does **not**
  match this project's semantics of explicit `train_years`/`val_years` lists
  — likely need a hand-built `SplitManifest` (or `sample_range`) carving out
  day-index ranges that correspond to the named years, rather than relying
  on the automatic fraction split.

### Phase 3 — Transforms
- Implement the `batch_transform`: random 128x128 crop + constants
  injection + climatology gather-by-day-of-year.
- Validate with `insitubatch-check-transform` against one real chunk
  **before** trusting it in training (checks GIL release + declared output
  shape).

### Phase 4 — Parity check
- For a handful of matching dates, compare a batch from the new pipeline
  against the old tfrecords pipeline (fixed crop offset, same date) to catch
  normalisation/ordering bugs before spending any GPU time.

### Phase 5 — Wire into `setupdata.py` behind a switch
- Add a `CGAN_DATA_BACKEND=zarr|tfrecords` env var, mirroring the existing
  `CGAN_FIELD_SET` pattern in `data.py`, so `setup_batch_gen` can build
  either the current `DataGenerator` or `InSituDataset` + `as_tf_dataset`.
  Keep the tfrecords path alive as a fallback until a full run on the new
  path reproduces run11's CRPS.

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

- No built-in weighted multi-source sampling — needs the per-class-manifest
  interleave workaround (Section 2).
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
