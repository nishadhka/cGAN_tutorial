"""Phase 1 of the insitubatch migration: netCDF -> Zarr (see INSITUBATCH_MIGRATION_PLAN.md).

Writes one Zarr v3 store holding **full, uncropped** daily images -- the same
per-day tensors that ``tfrecords_generator.write_data()`` currently crops into
128x128 patches and splits into 4 rain-class GZIP tfrecords. Cropping becomes an
insitubatch ``batch_transform`` (Phase 3) instead of a write-time step, so it can
change without regenerating the store; rain-class-weighted sampling is expected
to use the ``rain_class`` array written here to build one ``SplitManifest`` per
class (Phase 2/3), replacing ``tf.data.Dataset.sample_from_datasets``.

Reuses ``data.py`` / ``data_generator.py`` unchanged -- only the sink changes,
from ``tf.io.TFRecordWriter`` to Zarr array assignment. Must run under an
environment with TensorFlow (``data_generator.DataGenerator`` imports it) *and*
``zarr``/``obstore``/``insitubatch`` -- i.e. ``cgan_env`` plus the core
insitubatch package (no ``[tf]`` extra needed here; TF is already present).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zarr

import data
from data import HOURS, LEAD_IDX, all_fcst_fields, denormalise, get_dates
from data_generator import DataGenerator as DataGeneratorFull
from insitubatch import ensure_local_dir, obstore_store

IMG_H, IMG_W = 384, 352
N_RAIN_CLASSES = 4


def _truth_mean(truth_log_image: np.ndarray) -> float:
    """Whole-image mean rainfall (mm/hr) for one day's truth, log-transform undone."""
    return float(denormalise(truth_log_image).mean())


def _quantile_classes(means: np.ndarray, n_classes: int = N_RAIN_CLASSES) -> tuple[np.ndarray, list]:
    """Digitize whole-image daily means into ``n_classes`` roughly-equal-count bins.

    ``write_data()``'s fixed bins ``[0.0059, 0.0362, 0.0761]`` were tuned on
    *per-patch* means (RFE2 2018 patch distribution) so a random 128x128 crop
    lands roughly evenly across 4 classes. Applied to a *whole-domain* daily
    mean instead -- there is no patch yet at write time, cropping is now a
    batch_transform (Phase 3) -- those same edges skew almost everything into
    the top classes: a full-domain mean is far less likely to sit near zero
    than a small patch is. Empirically (this store, 2018-2021): 0/125/612/724
    days in classes 0-3. Deriving the edges from this store's own quantiles
    instead gives a ~balanced split by construction, whatever the domain/years.
    """
    edges = np.quantile(means, np.linspace(0, 1, n_classes + 1)[1:-1]).tolist()
    classes = np.digitize(means, edges).astype("i1")
    return classes, edges


def write_zarr(years,
               url,
               fcst_fields=all_fcst_fields,
               log_precip=True,
               fcst_norm=True,
               day_chunk=32,
               compress=True,
               limit_days=None):
    """Write one Zarr store's ``fcst``/``truth``/``mask``/``rain_class`` arrays
    for every valid date in ``years``, in slabs of ``day_chunk`` whole chunks
    (bounds writer RAM, keeps every write chunk-aligned -- same technique as
    insitubatch's own ``examples/advection/data.py:make_advection_store``).

    ``limit_days`` truncates the date list (smoke-testing on a handful of days
    before committing disk/time to a full multi-year conversion).
    """
    all_dates: list[str] = []
    for year in years:
        all_dates.extend(get_dates(year, start_hour=HOURS, end_hour=HOURS))
    if limit_days is not None:
        all_dates = all_dates[:limit_days]
    n_days = len(all_dates)
    if n_days == 0:
        raise ValueError(f"no valid dates found for years={years}")
    print(f"write_zarr: {n_days} days across years {years} -> {url}", flush=True)

    dgc = DataGeneratorFull(all_dates,
                            fcst_fields=fcst_fields,
                            start_hour=HOURS,
                            end_hour=HOURS,
                            batch_size=1,
                            log_precip=log_precip,
                            shuffle=False,
                            constants=True,
                            fcst_norm=fcst_norm)

    first = dgc[0]
    n_fcst_channels = first[0]["lo_res_inputs"].shape[-1]
    print(f"  fcst channels: {n_fcst_channels} (field_set={data.FIELD_SET}, "
          f"fields={fcst_fields}, use_climatology={data.USE_CLIMATOLOGY})", flush=True)

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    compressors = "auto" if compress else None

    fcst_arr = group.create_array(
        "fcst", shape=(n_days, IMG_H, IMG_W, n_fcst_channels),
        chunks=(day_chunk, IMG_H, IMG_W, n_fcst_channels), dtype="f4",
        compressors=compressors, dimension_names=("day", "lat", "lon", "channel"))
    truth_arr = group.create_array(
        "truth", shape=(n_days, IMG_H, IMG_W, 1),
        chunks=(day_chunk, IMG_H, IMG_W, 1), dtype="f4",
        compressors=compressors, dimension_names=("day", "lat", "lon", "channel"))
    mask_arr = group.create_array(
        "mask", shape=(n_days, IMG_H, IMG_W),
        chunks=(day_chunk, IMG_H, IMG_W), dtype="bool",
        compressors=compressors, dimension_names=("day", "lat", "lon"))
    class_arr = group.create_array(
        "rain_class", shape=(n_days,), chunks=(day_chunk,), dtype="i1",
        compressors=compressors, dimension_names=("day",))

    # Provenance the code already warns is safety-critical (data.py: run06 and
    # run11 both resolve to 28 channels by coincidence -- record what actually
    # built this store so a mismatched read fails loudly, not silently).
    group.attrs["dates"] = list(all_dates)
    group.attrs["years"] = list(years)
    group.attrs["field_set"] = data.FIELD_SET
    group.attrs["fcst_fields"] = list(fcst_fields)
    group.attrs["use_climatology"] = bool(data.USE_CLIMATOLOGY)
    group.attrs["clim_channels"] = int(data.CLIM_CHANNELS)
    group.attrs["lead_idx"] = int(LEAD_IDX)
    group.attrs["log_precip"] = bool(log_precip)
    group.attrs["fcst_norm"] = bool(fcst_norm)

    fcst_buf = np.empty((day_chunk, IMG_H, IMG_W, n_fcst_channels), dtype="f4")
    truth_buf = np.empty((day_chunk, IMG_H, IMG_W, 1), dtype="f4")
    mask_buf = np.empty((day_chunk, IMG_H, IMG_W), dtype=bool)
    all_means = np.empty((n_days,), dtype="f8")

    t0 = time.perf_counter()
    for start in range(0, n_days, day_chunk):
        stop = min(start + day_chunk, n_days)
        n = stop - start
        for i in range(n):
            sample = dgc[start + i]
            fcst_buf[i] = sample[0]["lo_res_inputs"][0]
            truth_buf[i, ..., 0] = sample[1]["output"][0]
            mask_buf[i] = sample[1]["mask"][0]
            all_means[start + i] = _truth_mean(sample[1]["output"][0])
        fcst_arr[start:stop] = fcst_buf[:n]
        truth_arr[start:stop] = truth_buf[:n]
        mask_arr[start:stop] = mask_buf[:n]
        elapsed = time.perf_counter() - t0
        rate = stop / elapsed if elapsed else 0.0
        print(f"  wrote {stop}/{n_days} days ({stop / n_days:.0%})  "
              f"{elapsed:.1f}s  {rate:.1f} days/s", flush=True)

    # Deferred to a second pass over the (already in-memory) daily means: the
    # quantile edges need every day's mean before any day can be classified.
    classes, edges = _quantile_classes(all_means)
    class_arr[:] = classes
    group.attrs["rain_bins"] = edges
    print(f"  rain_class quantile edges (mm/hr): {edges}", flush=True)
    print(f"done: {url} in {time.perf_counter() - t0:.1f}s", flush=True)


def rebin_rain_class(url: str, n_classes: int = N_RAIN_CLASSES) -> list:
    """Recompute ``rain_class`` in an already-written store from its ``truth``
    array, without re-running the netCDF conversion. Fixes a store written
    before this quantile-based classification existed (see ``_quantile_classes``).
    """
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="r+")
    truth = group["truth"][:]  # (n_days, H, W, 1) -- small enough to hold in RAM
    means = denormalise(truth).mean(axis=(1, 2, 3))
    classes, edges = _quantile_classes(means, n_classes)
    group["rain_class"][:] = classes
    group.attrs["rain_bins"] = edges
    print(f"rebin_rain_class: {url} -> edges (mm/hr) {edges}", flush=True)
    return edges


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1: netCDF -> Zarr converter for insitubatch")
    p.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    p.add_argument("--url", default=None,
                   help="Zarr store URL (file:// or s3://); default is "
                        "/tank/projects/cGAN/zarr/run11_clim_meansd -- /tank/projects has "
                        "2.2T free vs. 45G on /home/ezra, so store datasets there, not in $HOME")
    p.add_argument("--day-chunk", type=int, default=32)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--limit-days", type=int, default=None,
                   help="truncate the date list -- smoke-test before a full run")
    p.add_argument("--rebin-only", action="store_true",
                   help="skip the netCDF conversion; just recompute rain_class "
                        "on an already-written store at --url")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    url = args.url or "file:///tank/projects/cGAN/zarr/run11_clim_meansd/"
    if args.rebin_only:
        rebin_rain_class(url)
    else:
        write_zarr(args.years, url, day_chunk=args.day_chunk,
                  compress=not args.no_compress, limit_days=args.limit_days)
