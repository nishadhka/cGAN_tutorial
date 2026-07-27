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

from data import load_hires_constants
from insitubatch.types import Batch

IMG_H, IMG_W = 384, 352


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

    def _select_indices(self, rain_class: np.ndarray) -> np.ndarray:
        """Weighted-without-replacement row selection matching ``class_weights``.

        Fills each class's quota from rows actually present in this
        (oversampled) batch; a shortfall (a class under-represented in this
        particular draw) backfills from whatever's left over, so the output
        is always exactly ``target_batch_size`` rows -- the weights are a
        target, not a hard guarantee, but a 4x oversample of 4
        roughly-equal-frequency classes (Phase 2's rebalanced ``rain_class``)
        makes a shortfall rare.
        """
        n_classes = len(self.class_weights)
        quota = np.round(np.array(self.class_weights) * self.target_batch_size).astype(int)
        quota[np.argmax(quota)] += self.target_batch_size - quota.sum()  # fix rounding drift

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

        idx = self._select_indices(batch.arrays["rain_class"])
        n = len(idx)

        fcst = batch.arrays["fcst"][idx]    # (n, 384, 352, 28)
        truth = batch.arrays["truth"][idx]  # (n, 384, 352, 1)
        mask = batch.arrays["mask"][idx]    # (n, 384, 352)

        crop = self.crop
        out_fcst = np.empty((n, crop, crop, fcst.shape[-1]), dtype=fcst.dtype)
        out_truth = np.empty((n, crop, crop, 1), dtype=truth.dtype)
        out_mask = np.empty((n, crop, crop), dtype=mask.dtype)
        out_const = np.empty((n, crop, crop, self._constants.shape[-1]), dtype=self._constants.dtype)

        idh = self._rng.integers(0, IMG_H - crop + 1, size=n)
        idw = self._rng.integers(0, IMG_W - crop + 1, size=n)
        for i in range(n):
            h0, w0 = int(idh[i]), int(idw[i])
            out_fcst[i] = fcst[i, h0:h0 + crop, w0:w0 + crop, :]
            out_truth[i] = truth[i, h0:h0 + crop, w0:w0 + crop, :]
            out_mask[i] = mask[i, h0:h0 + crop, w0:w0 + crop]
            out_const[i] = self._constants[h0:h0 + crop, w0:w0 + crop, :]

        return Batch(
            arrays={
                "lo_res_inputs": out_fcst,
                "hi_res_inputs": out_const,
                "output": out_truth,
                "mask": out_mask,
            },
            sample_indices=batch.sample_indices[idx],
        )
