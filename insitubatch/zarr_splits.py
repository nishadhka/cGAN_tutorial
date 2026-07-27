"""Phase 2 of the insitubatch migration: chunk-aligned SplitManifest matching this
project's explicit ``train_years``/``val_years`` config (see
INSITUBATCH_MIGRATION_PLAN.md, Phase 2 + "Open risks"), since insitubatch's
``split_by_chunk`` only knows fractional (.8/.1/.1) splits, not "these named years
are validation".

A chunk is assigned to a split if *any* date inside it falls in that split's year
list -- the same overlap-inclusive convention insitubatch's own
``split_by_chunk(sample_range=...)`` already uses for boundary chunks (chunks here
are date-contiguous and only straddle a year boundary a handful of times at
``day_chunk=32`` for 2018-2021: ~3 of 46). Train and val are allowed to overlap by
design, exactly reproducing this project's current setup -- ``config.yaml``'s
``TRAIN.train_years=[2018..2021]`` already contains ``VAL.val_years=[2020]``: 2020
patches DO train the model; the same year's full images are ALSO used for
progress-plot validation. This is *not* a held-out test split -- it is parity with
what the tfrecords pipeline already does.
"""

from __future__ import annotations

import zarr

from insitubatch import SplitManifest, SplitName, obstore_store, open_geometries


def build_year_split_manifest(url, train_years, val_years, test_years=(), variable="fcst"):
    """A chunk-aligned :class:`SplitManifest` from named years instead of fractions.

    ``variable`` picks which array's geometry defines the chunk grid (all arrays in
    a `write_zarr.py` store share one day-chunking, so any of them works).
    """
    store = obstore_store(url)
    group = zarr.open_group(store=store, mode="r")
    dates = list(group.attrs["dates"])
    geom = open_geometries(store, variables=[variable])[variable]

    def chunk_years(c: int) -> set[int]:
        lo = c * geom.sample_chunk_size
        hi = min(lo + geom.sample_chunk_size, geom.n_samples)
        return {int(d[:4]) for d in dates[lo:hi]}

    def assign(years) -> list[int]:
        years = set(years)
        return sorted(c for c in range(geom.n_chunks) if chunk_years(c) & years)

    return SplitManifest(
        n_chunks=geom.n_chunks,
        sample_chunk_size=geom.sample_chunk_size,
        n_samples=geom.n_samples,
        chunks={
            SplitName.TRAIN.value: assign(train_years),
            SplitName.VAL.value: assign(val_years),
            SplitName.TEST.value: assign(test_years),
        },
        seed=0,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Phase 2: build + report a year-based SplitManifest")
    p.add_argument("--url", default="file:///tank/projects/cGAN/zarr/run11_clim_meansd/")
    p.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    p.add_argument("--val-years", type=int, nargs="+", default=[2020])
    args = p.parse_args()

    manifest = build_year_split_manifest(args.url, args.train_years, args.val_years)
    store = obstore_store(args.url)
    geom = open_geometries(store, variables=["fcst"])["fcst"]
    for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
        chunk_idxs = manifest.chunks[split.value]
        n_samples = len(manifest.sample_indices(split, geom))
        print(f"{split.value:>5}: {len(chunk_idxs):2d} chunks, {n_samples:5d} samples "
              f"-> chunk idxs {chunk_idxs}")
