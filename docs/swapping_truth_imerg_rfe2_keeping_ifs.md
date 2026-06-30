# Swapping the truth (IMERG ↔ RFE2) while keeping the IFS work — is it possible?

**Question:** can we keep the IFS TFRecords intact and change *only* the truth
(IMERG ↔ RFE2), so we don't redo the expensive IFS pipeline each time?

**Short answer:** **Not at the TFRecord level** — the truth is *fused into every
TFRecord example*, so changing it means regenerating the TFRecords. **But the
expensive part (the ~388 GB IFS download + normalisation) is fully reusable**, so
a truth swap costs only a cheap re-run of `write_data`, not a re-download. And if
you want truth that is *genuinely hot-swappable*, the fix is to **decouple X and Y
into separate Zarr stores** (the PyTorch direction).

---

## 1. Why the current TFRecords are NOT truth-swappable

Each TFRecord `Example` (see `tfrecords_generator.py` / `data.py`) bundles three
things **for the same random crop of the same date**:

```
Example = {
    generator_input  : forecast patch  (128×128×C, the IFS fields)   ← X
    constants        : elev/lsm patch  (128×128×2)
    generator_output : truth patch     (128×128×1, IMERG or RFE2)    ← Y  ← fused in
}
```

Three things make Y inseparable from X once written:

1. **Co-located random crop.** The patch is cut at `idh,idw = random.randint(...)`,
   and **those crop coordinates are never stored.** You cannot re-align a new
   truth to the existing input patches.
2. **Truth-dependent binning.** Which class file (`…​.0/.1/.2/.3.tfrecords`) a
   patch lands in is decided from `denormalise(truth).mean()` vs the rain bins.
   A different truth → different bin membership → different files.
3. **Baked transforms/units.** `log10(1+y)`, mm/day→mm/hr, mask — all applied at
   write time.

So **swapping IMERG↔RFE2 requires regenerating the TFRecords.** There is no way to
patch the new truth into existing `.tfrecords`.

---

## 2. What *is* reusable (the important, practical part)

The cost of a truth swap is **not** the IFS data — only the final write:

| Pipeline stage | Reused on truth swap? | Cost |
|---|---|---|
| IFS download (~388 GB, 14 fields × 4 yr) | ✅ **reused** | $0, already on disk/source.coop |
| `FCSTNorm*.pkl` (forecast normalisation) | ✅ **reused** (depends only on IFS) | $0 |
| Constants (elev/lsm) | ✅ reused | $0 |
| `write_data(year)` (patch + bin + serialize) | ♻️ **re-run** with new truth | cheap, CPU-only, ~minutes–hours |
| TFRecords (~14–20 GB) | ♻️ regenerated + re-uploaded | one upload |

So in practice: **point `TRUTH_PATH` at the new truth, re-run `write_data` for
each year, re-upload.** The 388 GB IFS download and the normalisation are
untouched. That is the "keep the IFS work intact" guarantee at the *raw* level —
just not at the *TFRecord* level.

**Requirement for a clean swap:** both truth sources must be on the **same
384×352 IFS grid**. RFE2 is already regridded to it; IMERG must be regridded the
same way so X/Y stay registered cell-for-cell.

---

## 3. How to make the truth genuinely hot-swappable (decouple X and Y)

If the goal is to swap truth **without regenerating** anything, stop fusing Y into
the TFRecords. Three designs, easiest/best last:

### Option A — store crop coordinates, truth as full-grid
Write IFS-only TFRecords that **also record `(date, idh, idw)`** per patch; keep
truth as full-grid NetCDF/Zarr per date. At load, crop the truth at the stored
coords. Swap = swap the truth store. *Cost:* must recompute bins per truth (or
bin on the fly / drop oversampling).

### Option B — parallel, co-indexed truth TFRecords
One IFS-only TFRecord set + **one truth-only set per source**, written with the
**same deterministic crops/index**; pair by index at load. *Cost:* brittle — any
change to the crop RNG desyncs them.

### Option C — full on-the-fly from Zarr ✅ recommended
Don't patch at write time at all:
- **X store:** IFS predictors as Zarr `[(date, channel, lat, lon)]` — built once.
- **Y store:** truth as Zarr `[(date, lat, lon)]`, **one per source** (RFE2, IMERG).
- Patch **on the fly** in the dataloader; crop X and the co-located Y together.

Swapping truth = open a different Y Zarr. **X is never touched.** This is exactly
the PyTorch dataloader pattern in
[`pytorch_cgan_direction_oxford_ifs.md`](pytorch_cgan_direction_oxford_ifs.md) —
i.e. **the truth-swap question and the PyTorch/Zarr direction are the same
architectural move.**

---

## 4. Recommendation

| If you are on… | To swap IMERG ↔ RFE2… |
|---|---|
| **TensorFlow + TFRecords** (current variant A routine) | Re-run `write_data` with the new `TRUTH_PATH`; IFS download + `FCSTNorm` are reused. Regen is cheap (~14–20 GB). **No hot-swap.** |
| **PyTorch + Zarr** (the direction) | Keep one **X Zarr** + per-source **Y Zarr**; point the dataloader at the desired Y. **True hot-swap, X untouched.** |

**So: yes, there is a possibility** — but only by decoupling X and Y (Option C /
Zarr). With the fused TFRecords you can't hot-swap; you re-generate, which is
cheap because the costly IFS half is reused. If truth-swapping is a first-class
requirement, that is itself a strong argument for the **Zarr-based PyTorch
pipeline** over fused TFRecords.
