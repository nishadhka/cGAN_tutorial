import gc
import os

from data import all_fcst_fields
from tfrecords_generator import DataGenerator

# insitubatch/Zarr backend switch (see INSITUBATCH_MIGRATION_PLAN.md, Phase 5).
# Mirrors data.py's CGAN_FIELD_SET env-var pattern. Default ("tfrecords") is
# today's unchanged behaviour; nothing about the tfrecords path below is
# altered by this switch existing.
CGAN_DATA_BACKEND = os.environ.get("CGAN_DATA_BACKEND", "tfrecords")
if CGAN_DATA_BACKEND not in ("tfrecords", "zarr"):
    raise ValueError(f"CGAN_DATA_BACKEND={CGAN_DATA_BACKEND!r} unknown; pick 'tfrecords' or 'zarr'")

# Zarr backend defaults -- only read when CGAN_DATA_BACKEND=zarr is actually selected.
CGAN_ZARR_URL = os.environ.get("CGAN_ZARR_URL", "file:///tank/projects/cGAN/zarr/run11_clim_meansd/")
# Default 8 (not 4) so that with CROPS_PER_IMAGE=8 the loader still reads 8
# *distinct* images per batch. The pre-multi-crop code read 32 images to build a
# 32-candidate pool, i.e. every candidate came from a different day; taking
# several crops per image trades some of that day-diversity for throughput, and
# keeping n_images=8 holds the trade to a sensible point (measured: 3.48
# batches/s = ~46x the training loop's demand, distribution parity 0.99x median).
CGAN_ZARR_OVERSAMPLE = int(os.environ.get("CGAN_ZARR_OVERSAMPLE", "8"))
CGAN_ZARR_CROP = int(os.environ.get("CGAN_ZARR_CROP", "128"))
# How many random crops to cut from each loaded image. A stored sample is a full
# 384x352 image but a crop uses only 12% of it, so 1 crop/image throws away 88%
# of every byte read (Phase 6: that is why the zarr loader measured ~80x slower
# than tfrecords). Raising this cuts bytes-read-per-crop ~linearly and enlarges
# the class-selection pool; the cost is that crops sharing an image are
# correlated, so keep enough distinct images per batch.
CGAN_ZARR_CROPS_PER_IMAGE = int(os.environ.get("CGAN_ZARR_CROPS_PER_IMAGE", "8"))


# Incredibly slim wrapper around tfrecords_generator.DataGenerator.  Can probably remove...
def setup_batch_gen(train_years,
                    batch_size=16,
                    autocoarsen=False,
                    weights=None):

    # print(f"autocoarsen flag is {autocoarsen}")
    batch_gen_train = DataGenerator(train_years,
                                    batch_size=batch_size,
                                    autocoarsen=autocoarsen,
                                    weights=weights)
    return batch_gen_train


def setup_batch_gen_zarr(train_years,
                         batch_size=16,
                         autocoarsen=False,
                         weights=None):
    """Zarr/insitubatch equivalent of `setup_batch_gen`, selected by
    `CGAN_DATA_BACKEND=zarr`. Reads the pre-converted Zarr store (`write_zarr.py`)
    instead of `.tfrecords`; `zarr_transforms.CropConstantsClassBalance` does the
    crop + constants-injection + rain-class-weighted resample that
    `write_data()`/`create_mixed_dataset()` used to do at write/read time.

    `autocoarsen` is not supported on this backend (untested and unused in this
    project on the tfrecords path either -- `data_generator.DataGenerator`
    asserts `autocoarsen is False`); raise loudly rather than silently ignore it.
    """
    if autocoarsen:
        raise NotImplementedError(
            "CGAN_DATA_BACKEND=zarr does not support autocoarsen; use the tfrecords backend")

    from insitubatch import obstore_store, open_geometries
    from insitubatch.source import InSituDataset

    from zarr_splits import build_year_split_manifest
    from zarr_transforms import CropConstantsClassBalance
    from zarr_tf_dataset import as_cgan_tf_dataset

    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)

    store = obstore_store(CGAN_ZARR_URL)
    geoms = open_geometries(store, variables=["fcst", "truth", "mask", "rain_class"])
    n_fcst_channels = geoms["fcst"].inner_shape[-1]

    manifest = build_year_split_manifest(CGAN_ZARR_URL, train_years, val_years=())
    cpi = CGAN_ZARR_CROPS_PER_IMAGE
    transform = CropConstantsClassBalance(crop=CGAN_ZARR_CROP,
                                          class_weights=tuple(weights),
                                          target_batch_size=batch_size,
                                          crops_per_image=cpi)
    # Candidate crops = images_read * cpi, and we want batch_size*OVERSAMPLE of
    # them -- so reading cpi times fewer images gives the SAME selection pool for
    # a fraction of the IO. Floor at 2 images so a batch is never drawn from a
    # single day.
    n_images = max(2, -(-batch_size * CGAN_ZARR_OVERSAMPLE // cpi))
    ds = InSituDataset(store, manifest, geometries=geoms,
                       batch_size=n_images,
                       shuffle=True, batch_transforms=[transform])
    # repeat=True is required, not cosmetic: gan.py pulls ~384 batches per
    # checkpoint from a single iterator, but one epoch of this store is only ~46.
    return as_cgan_tf_dataset(ds, crop=CGAN_ZARR_CROP,
                              n_fcst_channels=n_fcst_channels, split="train", repeat=True)


def setup_full_image_dataset(years,
                             batch_size=1,
                             autocoarsen=False):

    from data_generator import DataGenerator as DataGeneratorFull
    from data import get_dates

    from data import HOURS
    if isinstance(years, (list, tuple)):
        dates = []
        for y in years:
            dates += get_dates(y, start_hour=0, end_hour=HOURS)
    else:
        dates = get_dates(years, start_hour=0, end_hour=HOURS)
    data_full = DataGeneratorFull(dates=dates,
                                  fcst_fields=all_fcst_fields,
                                  start_hour=0,
                                  end_hour=HOURS,
                                  batch_size=batch_size,
                                  log_precip=True,
                                  shuffle=True,
                                  constants=True,
                                  fcst_norm=True,
                                  autocoarsen=autocoarsen)
    return data_full


def setup_data(train_years=None,
               val_years=None,
               autocoarsen=False,
               weights=None,
               batch_size=None):

    train_fn = setup_batch_gen_zarr if CGAN_DATA_BACKEND == "zarr" else setup_batch_gen
    batch_gen_train = None if train_years is None \
        else train_fn(train_years=train_years,
                      batch_size=batch_size,
                      autocoarsen=autocoarsen,
                      weights=weights)

    # Validation is backend-agnostic already -- setup_full_image_dataset reads
    # netCDF directly via data_generator.DataGenerator, never tfrecords, so it
    # needs no zarr equivalent.
    data_gen_valid = None if val_years is None \
        else setup_full_image_dataset(val_years,
                                      autocoarsen=autocoarsen)

    gc.collect()
    return batch_gen_train, data_gen_valid
