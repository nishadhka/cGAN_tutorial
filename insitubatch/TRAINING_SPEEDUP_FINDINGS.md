# Making cGAN training faster — what worked, what didn't, and why

**Bottom line:** a full 204,800-sample run takes **24.4 h** today. Halving the
discriminator and enabling XLA takes it to **10.2 h** — a **2.4x speedup,
~14 hours saved per run**. The Zarr + insitubatch migration, which this
investigation started as, contributes **nothing** to that. The two things are
independent, and conflating them was the main analytical error along the way.

The CRPS of the faster configuration is **not yet verified**. See
[§6 Status](#6-status-and-what-is-still-unproven).

---

## 1. Where the idea started, and why it was wrong

The premise was: *"use Zarr + insitubatch to flood the GPU with data properly,
instead of converting netCDF into tfrecords."*

That premise assumed the GPU was **data-starved** — waiting on the loader. It
is not, and never was. Measured:

| | samples/s |
|---|---|
| what training actually consumes | **6.6** |
| what the tfrecords loader supplies | **820.7** |

The existing GZIP tfrecords pipeline over-supplies the GPU by **124x**. There
was no starvation to fix. Every second of a training run is spent in the
model's forward/backward passes, not waiting for data.

This is worth stating plainly because it invalidates the *performance*
rationale for the migration entirely — while leaving its *maintainability*
rationale intact (see [§3](#3-so-is-the-zarr-work-worthless)).

---

## 2. The Zarr + insitubatch attempt, in full

[insitubatch](https://github.com/emfdavid/insitubatch) is a data loader that
streams batches straight out of a Zarr store with one async event loop, instead
of PyTorch/TF-style worker processes. The plan was to replace the
netCDF → cropped, class-binned, GZIP `.tfrecords` → `tf.data.TFRecordDataset`
pipeline with netCDF → Zarr → insitubatch.

### 2.1 What was built (all working, all committed)

| file | role |
|---|---|
| `dsrnngan/write_zarr.py` | netCDF → Zarr converter. Writes full uncropped daily images. |
| `dsrnngan/zarr_splits.py` | Chunk-aligned train/val manifest from named years. |
| `dsrnngan/zarr_transforms.py` | The batch transform: crop, inject constants, rain-class resample. |
| `dsrnngan/zarr_tf_dataset.py` | Wraps the loader as a `tf.data.Dataset` in the shape `train.py` expects. |
| `dsrnngan/parity_check.py` | Verifies output against the real production run11 tfrecords. |
| `dsrnngan/bench_loaders.py` | Loader throughput benchmark. |
| `setupdata.py` | `CGAN_DATA_BACKEND=zarr\|tfrecords` switch (default `tfrecords`). |

The store: `/tank/projects/cGAN/zarr/run11_clim_meansd/`, **19 GB**, 1,461 days
(2018–2021), arrays `fcst (1461,384,352,28)`, `truth`, `mask`, `rain_class`.

**No existing tfrecords code was modified.** `tfrecords_generator.py`,
`write_data()` and `create_mixed_dataset()` are untouched; the zarr path is a
new branch behind an env var that defaults to the old behaviour.

### 2.2 Correctness: it does work, and it is verified

Not the problem. Evidence:

- Converted data matches a direct `data_generator.DataGenerator` load
  **bit-for-bit** (`np.allclose` on both `fcst` and `truth`).
- Against the **real run11 production tfrecords**, 11 of 12 sampled patches
  verified byte-identical at their located crop window. (The 12th was an
  all-zero patch that matches **300,666** different windows in 2018 alone, so
  content-based location cannot disambiguate it — a limitation of the
  verification method against constant data, not a pipeline defect.)
- The delivered training distribution matches the tfrecords mixture at
  **0.99x median / 0.95x mean**, with class-conditional means at
  0.96x / 0.98x / 1.00x for classes 0–2.

### 2.3 Three real bugs found and fixed along the way

Worth recording, because two of them would have silently corrupted results.

**(a) Dataset exhaustion — would have crashed training immediately.**
`gan.py` calls `iter(batch_gen)` *once* and then `.get_next()` about
`steps_per_checkpoint x (training_ratio + 1)` ≈ **384 times per checkpoint**.
The tfrecords path is infinite (`create_mixed_dataset` ends in `ds.repeat()`).
The zarr path was finite: it raised `OutOfRangeError` after exactly **46
batches**, inside the *first* checkpoint. The Phase 5 validation had used
`.take(1)`, which structurally cannot detect this. Fixed by looping epochs
forever with a `set_epoch()` reshuffle per pass.

**(b) Wrong classification primitive — would have confounded every comparison.**
`write_data()` bins rain intensity **per patch**: every 128x128 crop gets its
own class, so the `[.4,.3,.2,.1]` weights select genuinely dry *patches*. The
first implementation binned **per day** and then cropped randomly — not
equivalent, because a random crop from the driest *day* is far wetter than the
driest *crop*. Measured effect: delivered patches were **1.86x wetter (median)**
than the real tfrecords. Fixed by cropping first, then classifying each crop
with `write_data()`'s own per-patch bins `(0.0059, 0.0362, 0.0761)`.

**(c) Deterministic quota rounding — a subtler version of the same bias.**
`np.round([3.2, 2.4, 1.6, 0.8])` = `[3,2,2,1]` pins the effective weights at
`[.375,.25,.25,.125]` on *every* batch, over-weighting the two wettest bins by
~19%. `sample_from_datasets` draws each element *independently* with
`p=weights`, so the exact equivalent is `rng.multinomial(batch_size, weights)`.

### 2.4 Why it does not make training faster

Measured loader throughput, batch size 8:

| backend | batches/s | samples/s | per batch |
|---|---|---|---|
| tfrecords (GZIP) | **102.6** | 820.7 | 9.7 ms |
| zarr, 1 crop per image | 1.29 | 10.3 | 773 ms |

**~80x slower.** The cause is **read amplification**, confirmed rather than
assumed:

- A stored Zarr sample is a **full 384x352x28 image = 15.8 MB**.
- Training uses a **128x128 crop = 1.9 MB — 12% of it**.
- At 4x oversampling (needed to have candidates for class balancing), the
  loader reads **506 MB per batch to deliver 15.3 MB — 33x amplification**.
- tfrecords stores **pre-cropped patches**, so it reads ~exactly what it uses.

Cutting oversampling 4→1 reduced amplification 33x→8x and raised throughput
1.29→5.16 batches/s: a 4.0x gain for a 4.1x amplification cut, i.e. **almost
exactly linear**, which pins the bottleneck on bytes read rather than compute.

**This is architectural, not a tuning miss.** insitubatch's unit of work is the
*sample* — `ArrayGeometry.slot_shape` assembles the entire `inner_shape` — so
it has **no spatial-subsetting path**. Chunking the store spatially would not
let it read only the crop; it would still materialise the whole image.
Reaching 1x would mean storing pre-cropped patches as the samples, which is
tfrecords in a Zarr container, and forfeits the fresh-crop-per-epoch benefit
that was the main reason to migrate.

Two secondary inefficiencies in the store, for the record:
- **Zstd earns only 1.16x** (22.1 GB raw → 19 GB on disk), so decompressing
  411 MB chunks is close to pure CPU overhead.
- **`day_chunk=32` produces 411 MB chunks**, which hurts random-access latency
  and pool memory (though not bytes-per-epoch, since every day is read once).

### 2.5 The partial rescue, and why it changes nothing

Taking **several crops from each loaded image** decouples "images read" from
"candidate crops" and recovers most of the waste:

| config | images read / batch | batches/s |
|---|---|---|
| 1 crop per image (original) | 32 | 1.29 |
| 8 crops per image, 4 images | 4 | **7.25** |
| 8 crops per image, 8 images (default) | 8 | **3.48** |

A **5.6x** loader speedup for a one-line change, and it *also* improved
distribution parity (0.83x → 0.99x median) because a larger candidate pool
means fewer class-quota shortfalls. The default keeps 8 distinct images per
batch, trading some throughput for day-diversity.

**And it makes no difference to training**, because the loader was never the
bottleneck. End-to-end, on the GPU, with the real model:

| backend | s per training iteration |
|---|---|
| tfrecords | **3.62** |
| zarr | **3.64** |

**0.6% apart — indistinguishable.**

---

## 3. So is the Zarr work worthless?

No — but its value is **maintainability, not speed**, and it should be adopted
or rejected on that basis alone:

- **No tfrecords regeneration** when the field set or crop size changes. Today
  that is a documented multi-step ordeal (see `RUN08_NOCAPE_STEPS.md`: edit
  `data.py`, rebuild `FCSTNorm2018.pkl`, regenerate ~19 GB of tfrecords, then
  rebuild validation records — all of which must stay mutually consistent).
- **Fresh random crops every epoch** instead of a frozen set of patches baked
  in at write time. The model sees more of the domain over a run.
- **Cloud-native store** (`s3://`/`gs://` work with the same code path).

Against that: it is ~4x slower at loading even after the multi-crop fix
(4.2x headroom over demand vs tfrecords' 124x), and insitubatch is alpha
software. It is currently wired in behind `CGAN_DATA_BACKEND=zarr` with
tfrecords as the default, which is the right place to leave it.

---

## 4. Where the time actually goes

Sweep at batch 8, measured per training iteration:

| change | s/iter | speedup | free? |
|---|---|---|---|
| baseline (gen 64 / disc 256, ens 8, ratio 2) | 3.63 | — | |
| `filters_disc` 256 → **64** | **1.44** | **2.52x** | no — capacity change |
| XLA + `filters_disc` 128 | 1.60 | 2.27x | no |
| `filters_disc` 256 → **128** | **2.04** | **1.78x** | no — capacity change |
| `training_ratio` 2 → 1 | 2.41 | 1.51x | no — changes WGAN dynamics |
| **XLA JIT alone** | **2.85** | **1.27x** | ~yes (numerics only) |
| `filters_gen` 64 → 32 | 3.28 | 1.11x | no |
| `ensemble_size` 8 → 2 | 3.31 | 1.10x | no |
| `ensemble_size` 8 → 4 | 3.45 | 1.05x | no |
| `batch_size` 16 or 32 | **OOM** | — | |

**The discriminator dominates — not the generator ensemble.** This contradicted
the obvious guess. Cutting the ensemble 8→2 (4x less generator work) buys only
**1.10x**, and halving the generator's filters only **1.11x**, while halving the
discriminator's filters buys **1.78x**.

**Mixed precision (float16) does not currently work.** It dies with a dtype
`TypeError` in the WGAN-GP path: `layers.py:RandomWeightedAverage` draws float32
interpolation weights and multiplies them by float16 activations. Making that
one layer dtype-agnostic is **not sufficient** — a second mismatch follows in
the same graph. Enabling AMP needs a small dtype audit of `gan.py` / `layers.py`,
and GANs with gradient penalties are known to be fp16-fragile, so it should be
treated as an experiment with a loss-curve check, not a flag flip. Potentially
the largest remaining win (Ada tensor cores).

---

## 5. Background: the two changes that actually work

### 5.1 Why halving the discriminator helps so much

The model is a **WGAN-GP** (Wasserstein GAN with gradient penalty). One
training iteration at `training_ratio=2` runs:

```
2 x discriminator step:
     generator forward  (make fakes)
   + discriminator on real
   + discriminator on fake
   + gradient penalty  <-- interpolate real/fake, then take the gradient
                           of D w.r.t. that input, which requires a SECOND
                           backward pass through the discriminator
1 x generator step:
     8 x generator forward   (the ensemble for the ensmeanMSE content loss)
   + discriminator forward
   + backward
```

Two things make the discriminator, not the generator, the cost centre:

1. **It is 4x wider.** `filters_disc: 256` against `filters_gen: 64`. Convolution
   cost scales roughly with the product of input and output channel counts, so a
   4x wider network is far more than 4x the arithmetic of the generator per pass.
2. **The gradient penalty makes it run twice per pass.** The penalty term needs
   ∇D(x̂) where x̂ is a random interpolation between real and fake samples
   (`layers.py:RandomWeightedAverage`). Differentiating that gradient during
   optimisation means back-propagating *through a backward pass* —
   double-backprop — so every discriminator step costs roughly twice a plain
   forward+backward.

Combined, the discriminator is touched 2 x (real + fake + double-backprop GP)
per iteration versus the generator's mostly-forward-only ensemble work. That is
why `ensemble_size` — which looks expensive at 8 forward passes — barely
registers, while `filters_disc` dominates.

**Halving `filters_disc` 256 → 128 gives 1.78x.** The cost is model capacity:
a weaker critic may provide a weaker training signal to the generator. That is
an empirical question, which is what run12 tests.

Is 256 actually needed? The existing architecture sweep does **not** establish
that it is:

| run | gen/disc | best CRPS | checkpoints |
|---|---|---|---|
| run11 | 64/256 | **0.0671** | 30 |
| run06 | 64/256 | 0.0719 | 200 |
| arch_128_512 | 128/512 | 0.0750 | 80 |
| arch_8_32 | 8/32 | 0.0783 | 80 |
| arch_32_128 | 32/128 | 0.0804 | 80 |
| arch_16_64 | 16/64 | 0.0909 | 80 |

The `arch_*` runs are short screening runs on a different schedule, so they are
not directly comparable to run11's 0.0671. But *among themselves* the ordering
is noisy — 8/32 beats both 32/128 and 16/64 — which means capacity was not
cleanly determining CRPS at that budget. And **64/128 has never been tested**.
Given it is 1.78x faster, that is the highest-value experiment available.

### 5.2 What XLA fusion does

XLA (Accelerated Linear Algebra) is a compiler built into TensorFlow. Without
it, TF executes the graph **op by op**: each convolution, bias-add, activation
and elementwise multiply is a separate GPU kernel launch, and each one writes
its full output tensor to GPU memory before the next kernel reads it back.

For a network of *many small* operations — exactly this generator and
discriminator, which are stacks of modest convolutions and residual blocks at
128x128 — that pattern is dominated by two overheads:

- **Kernel launch latency**: thousands of tiny launches per iteration, each with
  fixed CPU-side cost.
- **Memory bandwidth**: intermediate tensors round-trip to GPU DRAM between
  every op, even though the next op consumes them immediately.

XLA compiles clusters of adjacent ops into a **single fused kernel**
("autoclustering"). A `conv → bias → activation` chain becomes one kernel that
keeps the intermediates in registers/shared memory and writes DRAM once. Fewer
launches, far less bandwidth.

That is why it earns **1.27x here for no change in model capacity** — it is
purely a change in how the same arithmetic is scheduled.

The caveat: fusion **re-orders floating-point operations**, so a JIT run is not
bit-identical to a non-JIT one. Results are equivalent within float tolerance,
not reproducible bit-for-bit against previous runs. That is why it is exposed as
an opt-in flag rather than switched on by default:

```yaml
# local_config.yaml
use_xla: True
```

which `read_config.set_gpu_mode()` turns into `tf.config.optimizer.set_jit(True)`,
alongside the existing `gpu_mem_incr` / `disable_tf32` options. It defaults to
`False`, so no existing run changes behaviour unless it opts in.

---

## 6. Status and what is still unproven

### Measured, from real checkpoint timestamps

| run | config | per checkpoint | full 200-checkpoint run |
|---|---|---|---|
| run06 | disc 256, no XLA | 441 s | 24.5 h |
| **run11** | disc 256, no XLA | 440 s | **24.4 h** (current baseline) |
| **run12** | **disc 128 + XLA** | **183 s** | **10.2 h** |

**2.40x faster, ~14 hours saved per run.** run06 and run11 agreeing to within
1 s/checkpoint confirms the measurement method.

### The gap

**run12's CRPS is unknown.** The run was stopped at **106,496 / 204,800 samples
(52%, checkpoint 105/200)** when the GPU was needed for another job, before any
evaluation. It ran clean to that point — no NaN, no errors, rate stable — so the
*speed* claim is validated over 3.5 h of real training. The *quality* claim is
not tested at all.

A 2.4x speedup is worthless if skill degrades. Until that eval runs:

- **XLA alone is low-risk** — 1.27x (24.4 h → ~19 h), no capacity change,
  independent of the discriminator question. Reasonable to adopt now.
- **disc 128 should not be adopted** until its CRPS is compared against run11's
  0.0671.

### Resuming run12

Weights and optimizer states are intact in `logs_RFE2_run12_disc128/`.
About 5 h of training remain, then the evaluation.

```bash
cd <cGAN>/dsrnngan
export LD_LIBRARY_PATH=$(ls -d /home/ezra/cgan_env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
setsid nohup env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /home/ezra/cgan_env/bin/python main.py \
  --config config_run12_disc128.yaml --restart \
  > ~/train_run12_disc128.log 2>&1 < /dev/null & disown
```

`--restart` is **mandatory**. Without it `main.py` starts from sample 0 and
overwrites run12's weights and `log.txt`.

Then evaluate and compare against run11's 0.0671:

```bash
env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" /home/ezra/cgan_env/bin/python main.py \
    --config config_run12_disc128.yaml --no_train --evaluate --eval_full \
    > ~/eval_run12_disc128.log 2>&1
```

### If more speed is wanted after that

1. **Fix mixed precision** — dtype audit of `gan.py` / `layers.py`. Largest
   remaining lever (tensor cores), highest uncertainty (fp16 + gradient penalty).
2. **`filters_disc` 64** — 2.52x, but a bigger capacity cut than 128; only worth
   testing if 128 holds CRPS.
3. **`training_ratio` 1** — 1.51x, but changes WGAN convergence behaviour, so it
   is a modelling decision rather than an optimisation.
4. **Larger batch** — currently blocked (16 and 32 both OOM at disc 256). Worth
   re-testing at disc 128, though note that TF's incremental allocator grew run12
   to the same ~28.7 GB as run11 in steady state, so the headroom may not be real.

---

## Appendix: reproducing the measurements

```bash
cd <cGAN>/dsrnngan
export LD_LIBRARY_PATH=$(ls -d /home/ezra/cgan_env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')

# loader throughput, one backend per process (CUDA disabled, safe on a busy GPU)
CGAN_DATA_BACKEND=tfrecords python bench_loaders.py --n 300
CGAN_DATA_BACKEND=zarr      python bench_loaders.py --n 300

# end-to-end training iteration time (needs the GPU)
CGAN_DATA_BACKEND=tfrecords python bench_train_step.py --label baseline --steps 6
python bench_train_step.py --label xla      --steps 6 --xla
python bench_train_step.py --label disc128  --steps 6 --filters-disc 128

# real per-checkpoint wall-clock from any completed run
ls -l --time-style=+%s logs_RFE2_run11/models/gen_weights-*.h5 \
  | awk '{print $6}' | sort -n | awk 'NR>1{print $1-p} {p=$1}' | sort -n
```
