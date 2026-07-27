"""Phase 5 helper: wrap an insitubatch split view as a `tf.data.Dataset` yielding
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
"""

from __future__ import annotations


def as_cgan_tf_dataset(view, crop: int, n_fcst_channels: int, prefetch: int = 2):
    """``view`` is an insitubatch split view (e.g. ``ds.train``) whose
    `batch_transforms` include `CropConstantsClassBalance` (or anything else
    producing these same four keys/shapes)."""
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

    def gen():
        for batch in view:
            arrs = batch.arrays
            yield (
                {"lo_res_inputs": arrs["lo_res_inputs"], "hi_res_inputs": arrs["hi_res_inputs"]},
                {"output": arrs["output"], "mask": arrs["mask"]},
            )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.prefetch(prefetch) if prefetch else ds
