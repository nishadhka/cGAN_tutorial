"""End-to-end GPU training-step benchmark -- the half of Phase 6 that needed an
idle GPU (see INSITUBATCH_MIGRATION_PLAN.md).

Builds the *real* model from a config (same `setupmodel.setup_model` call
`main.py` makes) and times actual `model.train()` iterations, so it answers two
questions the loader-only benchmark could not:

  1. Does the data backend change end-to-end training wall-clock? (`--backend`
     is set by the CGAN_DATA_BACKEND env var, one backend per process.)
  2. What does one training iteration actually cost, and which knobs move it?

`--steps` counts *generator* iterations; each one also runs `training_ratio`
discriminator steps, so the loader is asked for ~`training_ratio + 1` batches
per iteration. Timing excludes a warmup (first steps pay graph tracing / cuDNN
autotune, which would otherwise dominate a short run).

Knob overrides (`--filters-gen`, `--ensemble-size`, `--batch-size`, ...) let one
process measure a single change against the config baseline without editing
YAML. `--mixed-precision` flips TF's global policy to float16, and
`--xla` enables JIT -- both change numerics, so treat any run with them as a
speed probe, not a training run.
"""

from __future__ import annotations

import argparse
import os

# Must precede any tensorflow import, exactly as main.py:5 does -- models.py is
# Keras-2 code and Keras 3 (bundled with TF 2.21) rejects it with "A KerasTensor
# cannot be used as input to a TensorFlow function".
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time  # noqa: E402

import yaml  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="time real cGAN training iterations")
    p.add_argument("--config", default="config_run11.yaml")
    p.add_argument("--steps", type=int, default=6, help="timed generator iterations")
    p.add_argument("--warmup", type=int, default=2, help="untimed warmup iterations")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--filters-gen", type=int, default=None)
    p.add_argument("--filters-disc", type=int, default=None)
    p.add_argument("--ensemble-size", type=int, default=None)
    p.add_argument("--training-ratio", type=int, default=2)
    p.add_argument("--mixed-precision", action="store_true", help="float16 policy (changes numerics)")
    p.add_argument("--xla", action="store_true", help="enable XLA JIT (changes numerics)")
    p.add_argument("--label", default="", help="tag for the printed result line")
    return p


def main() -> None:
    args = build_parser().parse_args()

    import tensorflow as tf
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if args.xla:
        tf.config.optimizer.set_jit(True)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    import data
    import noise
    import read_config
    import setupdata
    import setupmodel

    with open(args.config) as f:
        sp = yaml.safe_load(f)

    mode = sp["GENERAL"]["mode"]
    arch = sp["MODEL"]["architecture"]
    padding = sp["MODEL"]["padding"]
    filters_gen = args.filters_gen or sp["GENERATOR"]["filters_gen"]
    filters_disc = args.filters_disc or sp["DISCRIMINATOR"]["filters_disc"]
    noise_channels = sp["GENERATOR"]["noise_channels"]
    latent_variables = sp["GENERATOR"]["latent_variables"]
    batch_size = args.batch_size or sp["TRAIN"]["batch_size"]
    ensemble_size = args.ensemble_size or sp["TRAIN"]["ensemble_size"]
    train_years = sp["TRAIN"]["train_years"]
    training_weights = sp["TRAIN"]["training_weights"]

    input_channels = 2 * len(data.all_fcst_fields)
    if getattr(data, "USE_CLIMATOLOGY", False):
        input_channels += getattr(data, "CLIM_CHANNELS", 1)

    df_dict = read_config.read_downscaling_factor()

    model = setupmodel.setup_model(
        mode=mode, arch=arch, downscaling_steps=df_dict["steps"],
        input_channels=input_channels, constant_fields=2,
        latent_variables=latent_variables, filters_gen=filters_gen,
        filters_disc=filters_disc, noise_channels=noise_channels, padding=padding,
        lr_disc=float(sp["DISCRIMINATOR"]["learning_rate_disc"]),
        lr_gen=float(sp["GENERATOR"]["learning_rate_gen"]),
        kl_weight=float(sp["TRAIN"]["kl_weight"]), ensemble_size=ensemble_size,
        CLtype=sp["TRAIN"]["CL_type"],
        content_loss_weight=float(sp["TRAIN"]["content_loss_weight"]))

    batch_gen, _ = setupdata.setup_data(train_years=train_years, val_years=None,
                                        autocoarsen=False, weights=training_weights,
                                        batch_size=batch_size)

    noise_shape = (128, 128, noise_channels)
    noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)

    model.train(batch_gen, noise_gen, args.warmup, training_ratio=args.training_ratio)

    t0 = time.perf_counter()
    model.train(batch_gen, noise_gen, args.steps, training_ratio=args.training_ratio)
    elapsed = time.perf_counter() - t0

    per_iter = elapsed / args.steps
    print("RESULT "
          f"label={args.label or 'baseline'} "
          f"backend={setupdata.CGAN_DATA_BACKEND} "
          f"batch={batch_size} gen={filters_gen} disc={filters_disc} "
          f"ens={ensemble_size} ratio={args.training_ratio} "
          f"amp={int(args.mixed_precision)} xla={int(args.xla)} "
          f"| {per_iter:.2f} s/iter  {batch_size / per_iter:.2f} samples/s  "
          f"| {args.steps} iters in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
