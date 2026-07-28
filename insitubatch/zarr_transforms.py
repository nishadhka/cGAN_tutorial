"""Phase 3 of the insitubatch migration: the batch_transform (see
INSITUBATCH_MIGRATION_PLAN.md, Phase 3 + Section 2).

One batch_transform replaces everything ``write_data()`` used to bake into each
``.tfrecords`` example at write time, now recomputed fresh on every draw instead
of frozen once into the store:

1. **Random 128x128 crop** of ``fcst``/``truth``/``mask`` -- same window for all
   three (``downscaling_factor`` is 1, so no separate hi/lo-res scaling is needed,
   see ``downscaling_factor.yaml``).
2. **Constants injection** -- elevation + land-sea mask are static (not a stored
   Zarr variable, see plan doc Section 2), loaded once and concatenated onto the
   cropped forecast stack, cropped to the same window.
3. **Rain-class-weighted resample** -- Phase 2 found this *cannot* be a
   per-class ``SplitManifest`` (splits are whole-chunk, ``rain_class`` is
   per-day). Instead: request an OVERSAMPLED batch from the loader (e.g. 32 rows
   for a target of 8), and weighted-without-replacement subselect down to the
   target batch size using each row's own ``rain_class``, matching
   ``config.yaml``'s ``TRAIN.training_weights = [0.4, 0.3, 0.2, 0.1]``.

Climatology is NOT gathered here. Phase 1 reused ``data_generator.DataGenerator``
unchanged, and ``load_fcst_stack`` already appends the climatology mean/sd
channels per date at *write* time -- they are baked into the stored ``fcst``
array's 28 channels (that's why it's 28, not 26). The plan doc's original draft
assumed this transform would gather climatology; corrected once Phase 1 actually
ran.

Output keys match ``data_generator.DataGenerator``'s nested dict exactly
(flattened -- a ``Batch`` is one flat dict): ``lo_res_inputs`` (fcst),
``hi_res_inputs`` (constants), ``output`` (truth), ``mask``. Phase 5 wires this
into the ``(inputs, outputs)`` tuple ``setupmodel.py``/``train.py`` expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data import denormalise, load_hires_constants
from insitubatch.types import Batch

IMG_H, IMG_W = 384, 352

# `write_data()`'s original per-PATCH bins (data.py's RFE2 2018 patch
# distribution, mm/hr). These are applied to each *crop's* own mean here --
# which is exactly what they were tuned on -- so the class-weighted draw
# reproduces `create_mixed_dataset`'s semantics. See `_classify_crops`.
PATCH_RAIN_BINS = (0.0059, 0.0362, 0.0761)


@dataclass
class CropConstantsClassBalance:
    """The one batch_transform for the run11 zarr pipeline.

    ``target_batch_size`` rows are returned; request a larger ``batch_size``
    from ``InSituDataset`` (the oversample factor) so each of the 4 rain
    classes usually has enough rows present to fill its share -- e.g.
    ``batch_size=32`` upstream for ``target_batch_size=8`` (4x oversample).
    """

    crop: int = 128
    class_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    target_batch_size: int = 8
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _constants: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _classify_crops(self, cropped_truth: np.ndarray) -> np.ndarray:
        """Rain class (0-3) of each *cropped* patch, from its own mean rainfall.

        This is the parity-critical step. `write_data()` classified **per
        patch** -- every 128x128 crop got its own bin -- and
        `create_mixed_dataset`'s weighted draw then selected genuinely dry
        *patches* for class 0. Classifying per *day* instead (the stored
        `rain_class` array) and cropping randomly is **not** equivalent: a
        random crop from the driest day is far wetter than the driest crop, so
        the delivered patches skewed wet (measured: median 1.86x, mean 1.25x
        wetter than the real run11 tfrecords). Cropping first and classifying
        each crop with `write_data()`'s own per-patch bins restores parity.
        """
        means = denormalise(cropped_truth).mean(axis=(1, 2, 3))
        return np.digitize(means, PATCH_RAIN_BINS).astype(int)

    def _select_indices(self, rain_class: np.ndarray) -> np.ndarray:
        """Weighted-without-replacement row selection matching ``class_weights``.

        Fills each class's quota from rows actually present in this
        (oversampled) batch; a shortfall (a class under-represented in this
        particular draw) backfills from whatever's left over, so the output
        is always exactly ``target_batch_size`` rows -- the weights are a
        target, not a hard guarantee.
        """
        # Multinomial, NOT deterministic rounding. `sample_from_datasets` draws
        # each element independently with p=weights, so the per-batch class
        # counts are Multinomial(batch_size, weights) -- this reproduces that
        # exactly. Deterministic `np.round` would fix the counts at
        # round([3.2,2.4,1.6,0.8]) = [3,2,2,1] every batch, i.e. effective
        # weights [.375,.25,.25,.125] instead of [.4,.3,.2,.1] -- which
        # over-weights the two wettest bins and measurably skewed the delivered
        # patches ~1.19x wet (predicted from the class means, and observed).
        n_classes = len(self.class_weights)
        weights = np.array(self.class_weights, dtype=float)
        quota = self._rng.multinomial(self.target_batch_size, weights / weights.sum())

        chosen: list[int] = []
        leftover: list[int] = []
        for c in range(n_classes):
            idx_c = np.flatnonzero(rain_class == c)
            self._rng.shuffle(idx_c)
            take = min(quota[c], len(idx_c))
            chosen.extend(idx_c[:take].tolist())
            leftover.extend(idx_c[take:].tolist())

        shortfall = self.target_batch_size - len(chosen)
        if shortfall > 0:
            leftover_arr = np.array(leftover, dtype=int)
            self._rng.shuffle(leftover_arr)
            chosen.extend(leftover_arr[:shortfall].tolist())

        chosen_arr = np.array(chosen[: self.target_batch_size], dtype=int)
        self._rng.shuffle(chosen_arr)
        return chosen_arr

    def __call__(self, batch: Batch) -> Batch:
        if self._constants is None:
            # (H, W, 2) -- elev + lsm, static across the whole store; loaded once.
            self._constants = load_hires_constants(batch_size=1)[0]

        # Crop EVERY oversampled row first, then classify each crop, then select
        # -- the order matters for parity; see `_classify_crops`. The stored
        # per-day `rain_class` variable is deliberately unused for weighting
        # (it is still co-batched, and remains useful for coarse day-level
        # filtering, but it cannot reproduce per-patch class semantics).
        fcst, truth, mask = batch.arrays["fcst"], batch.arrays["truth"], batch.arrays["mask"]
        n_all = fcst.shape[0]
        crop = self.crop

        all_fcst = np.empty((n_all, crop, crop, fcst.shape[-1]), dtype=fcst.dtype)
        all_truth = np.empty((n_all, crop, crop, 1), dtype=truth.dtype)
        all_mask = np.empty((n_all, crop, crop), dtype=mask.dtype)
        all_const = np.empty((n_all, crop, crop, self._constants.shape[-1]),
                             dtype=self._constants.dtype)

        idh = self._rng.integers(0, IMG_H - crop + 1, size=n_all)
        idw = self._rng.integers(0, IMG_W - crop + 1, size=n_all)
        for i in range(n_all):
            h0, w0 = int(idh[i]), int(idw[i])
            all_fcst[i] = fcst[i, h0:h0 + crop, w0:w0 + crop, :]
            all_truth[i] = truth[i, h0:h0 + crop, w0:w0 + crop, :]
            all_mask[i] = mask[i, h0:h0 + crop, w0:w0 + crop]
            all_const[i] = self._constants[h0:h0 + crop, w0:w0 + crop, :]

        idx = self._select_indices(self._classify_crops(all_truth))

        return Batch(
            arrays={
                "lo_res_inputs": all_fcst[idx],
                "hi_res_inputs": all_const[idx],
                "output": all_truth[idx],
                "mask": all_mask[idx],
            },
            sample_indices=batch.sample_indices[idx],
        )
