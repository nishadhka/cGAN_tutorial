"""Phase 4 of the insitubatch migration: parity check against the real,
already-written run11 tfrecords (see INSITUBATCH_MIGRATION_PLAN.md, Phase 4).

Reads a handful of real patches straight out of
``/home/ezra/rfe_tfrecords/run11_clim_meansd/*.tfrecords`` -- the production
dataset `write_data()` actually wrote and trained run11 on -- and checks that
each patch's tensors are byte-identical to a crop of the Zarr store at the
matching (date, window). A tfrecords patch carries no date/offset metadata
(`write_data()` picked random, unseeded crop locations), so the matching
window is *located* by content: a cheap point-value prefilter across every
day of the patch's year narrows candidates to (usually) one, then every
tensor is verified with a full ``np.array_equal``.

This is a stronger check than Phase 1's parity (which only compared full,
uncropped images against a fresh `data_generator.DataGenerator` call) -- it
validates the full historical write path (crop -> `tf.train.Example` ->
GZIP -> file -> read back -> parse) against the new Zarr path for data that
was *actually used to train run11*, not merely re-derived.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import zarr

from data import CLIM_CHANNELS, USE_CLIMATOLOGY, all_fcst_fields, load_hires_constants
from insitubatch import obstore_store

CROP = 128
FCST_CHANNELS = 2 * len(all_fcst_fields) + (CLIM_CHANNELS if USE_CLIMATOLOGY else 0)


def _feature_description():
    import tensorflow as tf
    return {
        "generator_input": tf.io.FixedLenFeature((CROP, CROP, FCST_CHANNELS), tf.float32),
        "constants": tf.io.FixedLenFeature((CROP, CROP, 2), tf.float32),
        "generator_output": tf.io.FixedLenFeature((CROP, CROP, 1), tf.float32),
    }


def read_tfrecord_patches(path: str, n: int):
    """Yield ``(fcst, constants, truth)`` numpy arrays for the first ``n`` patches
    in one ``.tfrecords`` file (GZIP-compressed, one `tf.train.Example` per patch)."""
    import tensorflow as tf
    fd = _feature_description()
    ds = tf.data.TFRecordDataset(path, compression_type="GZIP").take(n)
    for raw in ds:
        ex = tf.io.parse_single_example(raw, fd)
        yield (ex["generator_input"].numpy(), ex["constants"].numpy(), ex["generator_output"].numpy())


def locate_patch(truth_patch: np.ndarray, raw_truth: np.ndarray):
    """Find ``(day, h0, w0)`` in ``raw_truth`` (shape ``(n_days, H, W, 1)``) whose
    128x128 window equals ``truth_patch``, via a fast point-value prefilter
    (exact float32 equality at 3 corner pixels is an extremely strong filter for
    real, high-entropy rainfall data) before confirming with a full compare.
    """
    p00, p0e, pe0 = truth_patch[0, 0, 0], truth_patch[0, -1, 0], truth_patch[-1, 0, 0]
    n_days, H, W, _ = raw_truth.shape
    crop = truth_patch.shape[0]

    cand_mask = raw_truth[:, :, :, 0] == p00  # (n_days, H, W), cheap
    days, hs, ws = np.nonzero(cand_mask)
    for day, h0, w0 in zip(days.tolist(), hs.tolist(), ws.tolist()):
        if h0 + crop > H or w0 + crop > W:
            continue
        if raw_truth[day, h0, w0 + crop - 1, 0] != p0e:
            continue
        if raw_truth[day, h0 + crop - 1, w0, 0] != pe0:
            continue
        if np.array_equal(raw_truth[day, h0:h0 + crop, w0:w0 + crop, :], truth_patch):
            return day, h0, w0
    return None


def check_file(tfrecords_path: str, zarr_url: str, n_patches: int = 5):
    year = int(os.path.basename(tfrecords_path).split("_")[0])

    group = zarr.open_group(store=obstore_store(zarr_url), mode="r")
    dates = list(group.attrs["dates"])
    year_lo = dates.index(f"{year}0101") if f"{year}0101" in dates else None
    year_days = [i for i, d in enumerate(dates) if d.startswith(str(year))]
    assert year_days, f"no {year} dates found in zarr store attrs"
    lo, hi = year_days[0], year_days[-1] + 1

    raw_truth = group["truth"][lo:hi]   # (n_year_days, H, W, 1)
    raw_fcst = group["fcst"][lo:hi]     # (n_year_days, H, W, C)
    const_full = load_hires_constants(batch_size=1)[0]  # (H, W, 2)

    results = []
    for i, (fcst_patch, const_patch, truth_patch) in enumerate(
            read_tfrecord_patches(tfrecords_path, n_patches)):
        loc = locate_patch(truth_patch, raw_truth)
        if loc is None:
            results.append({"patch": i, "located": False})
            continue
        day_rel, h0, w0 = loc
        fcst_ok = np.array_equal(raw_fcst[day_rel, h0:h0 + CROP, w0:w0 + CROP, :], fcst_patch)
        const_ok = np.array_equal(const_full[h0:h0 + CROP, w0:w0 + CROP, :], const_patch)
        results.append({
            "patch": i, "located": True, "date": dates[lo + day_rel],
            "h0": h0, "w0": w0, "fcst_match": fcst_ok, "const_match": const_ok,
        })
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 4: parity check vs real run11 tfrecords")
    p.add_argument("--tfrecords-dir", default="/home/ezra/rfe_tfrecords/run11_clim_meansd/")
    p.add_argument("--zarr-url", default="file:///tank/projects/cGAN/zarr/run11_clim_meansd/")
    p.add_argument("--files", nargs="+", default=None,
                   help="specific .tfrecords basenames to check; default: one per class for 2018")
    p.add_argument("--n-patches", type=int, default=5)
    args = p.parse_args()

    files = args.files or ["2018_1.0.tfrecords", "2018_1.1.tfrecords",
                            "2018_1.2.tfrecords", "2018_1.3.tfrecords"]

    total, ok = 0, 0
    for fname in files:
        path = os.path.join(args.tfrecords_dir, fname)
        if not os.path.exists(path):
            print(f"{fname}: MISSING, skipped")
            continue
        results = check_file(path, args.zarr_url, args.n_patches)
        for r in results:
            total += 1
            good = r["located"] and r["fcst_match"] and r["const_match"]
            ok += int(good)
            print(f"{fname} patch {r['patch']}: {r}")
    print(f"\nTOTAL: {ok}/{total} patches fully verified (located + fcst + constants match)")
