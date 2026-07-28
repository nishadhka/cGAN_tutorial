"""Phase 6 (loader half): throughput of the Zarr/insitubatch backend vs the GZIP
tfrecords backend (see INSITUBATCH_MIGRATION_PLAN.md).

Measures the *data pipeline only* -- how fast each backend can hand assembled
batches to the training loop -- which is the question the migration turns on and
which needs no GPU. Runs with CUDA disabled so it cannot disturb a training job
on a shared card; the end-to-end wall-clock-per-checkpoint comparison still
needs an idle GPU and is deliberately left out of this script.

One backend per process: `CGAN_DATA_BACKEND` is read at import time by
`setupdata`, and TF/insitubatch state is process-global, so run e.g.

    CGAN_DATA_BACKEND=tfrecords python bench_loaders.py --n 300
    CGAN_DATA_BACKEND=zarr      python bench_loaders.py --n 300

Reports steady-state throughput after a warmup, plus time-to-first-batch
(cold-start latency, where insitubatch claims an advantage over worker pools).
"""

from __future__ import annotations

import argparse
import os
import time

# Keep this benchmark off the GPU entirely -- it measures IO/CPU, and the card
# may be running someone else's job. Must precede the tensorflow import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6: data-loader throughput benchmark")
    p.add_argument("--n", type=int, default=300, help="timed batches")
    p.add_argument("--warmup", type=int, default=20, help="untimed warmup batches")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    args = p.parse_args()

    import setupdata

    backend = setupdata.CGAN_DATA_BACKEND
    weights = [0.4, 0.3, 0.2, 0.1]

    t_build0 = time.perf_counter()
    batch_gen, _ = setupdata.setup_data(train_years=args.train_years, val_years=None,
                                        autocoarsen=False, weights=weights,
                                        batch_size=args.batch_size)
    build_s = time.perf_counter() - t_build0

    it = iter(batch_gen)

    t_first0 = time.perf_counter()
    inputs, outputs = it.get_next()
    first_batch_s = time.perf_counter() - t_first0

    for _ in range(args.warmup):
        it.get_next()

    rows = 0
    t0 = time.perf_counter()
    for _ in range(args.n):
        inputs, outputs = it.get_next()
        rows += int(inputs["lo_res_inputs"].shape[0])
    elapsed = time.perf_counter() - t0

    print(f"backend                : {backend}")
    print(f"batch_size             : {args.batch_size}")
    print(f"dataset build          : {build_s:.2f} s")
    print(f"time to first batch    : {first_batch_s:.3f} s")
    print(f"timed batches          : {args.n} ({rows} rows) after {args.warmup} warmup")
    print(f"elapsed                : {elapsed:.2f} s")
    print(f"throughput             : {args.n / elapsed:.2f} batches/s   {rows / elapsed:.1f} samples/s")
    print(f"per-batch latency      : {1000 * elapsed / args.n:.1f} ms")


if __name__ == "__main__":
    main()
