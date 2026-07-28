"""Phase 5 helper: wrap an insitubatch dataset as a `tf.data.Dataset` yielding
the exact ``(inputs, outputs)`` nested-dict shape `train.py`/`model.train()` expect
(see `data_generator.DataGenerator.__getitem__` / `tfrecords_generator._parse_batch`):
``({"lo_res_inputs": ..., "hi_res_inputs": ...}, {"output": ..., "mask": ...})``.

Not a use of insitubatch's own `as_tf_dataset`: that helper infers its
`output_signature` from the *source* geometries (`fcst`/`truth`/`mask`/`rain_class`,
at full 384x352 resolution), but `zarr_transforms.CropConstantsClassBalance`
renames and reshapes everything (crop to 128x128, `fcst`->`lo_res_inputs`, drops
`rain_class` after consuming it). Declaring a signature from the *transformed*
keys/shapes instead is a ~10-line `tf.data.Dataset.from_generator` call -- simpler
than patching insitubatch's inference to see through a renaming batch_transform.

**Infinite by construction**, matching the tfrecords path. `create_mixed_dataset()`
ends in `ds.repeat()`, and `gan.py`'s `train()` does `iter(batch_gen)` once then
calls `.get_next()` ~`steps_per_checkpoint * (training_ratio + 1)` times per
checkpoint (~384 for run11's config) -- across a full run, 25600+ generator
batches. One epoch of this store is only ~46 batches, so a finite dataset raises
`OutOfRangeError` inside the *first* checkpoint. The generator therefore loops
epochs forever, calling `set_epoch()` each pass so every epoch reshuffles
(the tfrecords path only ever reshuffled within a fixed 64-sample buffer).
"""

from __future__ import annotations


def as_cgan_tf_dataset(dataset, crop: int, n_fcst_channels: int,
                       split: str = "train", prefetch: int = 2, repeat: bool = True):
    """``dataset`` is an `insitubatch.source.InSituDataset` whose `batch_transforms`
    include `CropConstantsClassBalance` (or anything else producing these same four
    keys/shapes). ``split`` picks which view to iterate ('train'/'val'/'test'/'all').

    ``repeat=True`` (the default, and what training requires) iterates epochs
    endlessly with a reshuffle per epoch; ``repeat=False`` yields exactly one
    epoch, for evaluation or for counting a single pass.
    """
    import tensorflow as tf

    output_signature = (
        {
            "lo_res_inputs": tf.TensorSpec(shape=(None, crop, crop, n_fcst_channels), dtype=tf.float32),
            "hi_res_inputs": tf.TensorSpec(shape=(None, crop, crop, 2), dtype=tf.float32),
        },
        {
            "output": tf.TensorSpec(shape=(None, crop, crop, 1), dtype=tf.float32),
            "mask": tf.TensorSpec(shape=(None, crop, crop), dtype=tf.bool),
        },
    )

    def _emit(batch):
        arrs = batch.arrays
        return (
            {"lo_res_inputs": arrs["lo_res_inputs"], "hi_res_inputs": arrs["hi_res_inputs"]},
            {"output": arrs["output"], "mask": arrs["mask"]},
        )

    def gen():
        epoch = 0
        while True:
            dataset.set_epoch(epoch)
            for batch in getattr(dataset, split):
                yield _emit(batch)
            if not repeat:
                return
            epoch += 1

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.prefetch(prefetch) if prefetch else ds
