#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "tensorflow==2.15",
#     "numpy<2.0",
#     "numba",
#     "matplotlib",
#     "seaborn",
#     "cartopy",
#     "jupyter",
#     "xarray",
#     "netcdf4",
#     "scikit-learn",
#     "cfgrib",
#     "dask",
#     "tqdm",
#     "properscoring",
#     "climlab",
#     "scitools-iris",
#     "ecmwf-api-client",
#     "xesmf",
#     "flake8",
#     "regionmask",
#     "schedule",
#     "pyyaml",
#     "cftime",
# ]
# ///
"""
cGAN GEFS Inference Script - Raw NetCDF Method
===============================================

This script runs cGAN inference using raw NetCDF files with full ensemble data.
It reads raw data, applies normalization, and computes ensemble mean/std internally.

Input format:
- {field}_{year}.nc with dims (time, member, step, latitude, longitude)
- Raw data WITHOUT pre-computed ensemble statistics

Usage:
------
    uv run run_gefs_inference_raw.py
"""

import sys
import os
import pathlib
import yaml
import pickle
import netCDF4 as nc
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate, Conv2D, Dense, GlobalAveragePooling2D,
    Input, LeakyReLU, UpSampling2D, Layer, Add, BatchNormalization, AveragePooling2D
)
from cftime import date2num
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURATION - Raw NetCDF for 2025-09-18
# ==============================================================================
# Use relative paths from script directory for portability
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))

CONFIG = {
    # Model files from cgan_compact_20260202.zip
    "model_folder": _os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/logfile_gefs_v3/"),
    "checkpoint": 345600,
    # Raw NetCDF from GIK pipeline (stream_gefs_for_cgan.py + zarr_to_raw_netcdf.py)
    "input_folder": _os.path.join(_SCRIPT_DIR, "gik_cgan_pipeline_output/netcdf/"),
    "constants_path": _os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/CONSTANTS/"),
    # Output folder
    "output_folder": _os.path.join(_SCRIPT_DIR, "cgan_output/"),
    "dates": ["2025-09-18"],
    "start_hour": 30,
    "end_hour": 54,
    "ensemble_members": 50,
    "normalization_mode": "gefs",
    "gefs_norm_file": _os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/CONSTANTS/FCSTNorm_GEFS_2018.pkl"),
}

# Field definitions
HOURS = 6
all_fcst_fields = ["cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl", "apcp"]
nonnegative_fields = ["cape", "msl", "pres", "pwat", "tmp"]

# ICPAC region coordinates
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)


# ==============================================================================
# CUSTOM LAYERS (using tensorflow.keras)
# ==============================================================================
class ReflectionPadding2D(Layer):
    """Reflection padding layer."""
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1] + 2 * self.padding[0],
            None if s[2] is None else s[2] + 2 * self.padding[1],
            s[3],
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], "REFLECT")


class SymmetricPadding2D(Layer):
    """Symmetric padding layer."""
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1] + 2 * self.padding[0],
            None if s[2] is None else s[2] + 2 * self.padding[1],
            s[3],
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], "SYMMETRIC")


class Conv2DPadding(Layer):
    """Conv2D with custom padding support."""
    def __init__(self, filters, kernel_size, stride, padding, dilations):
        super(Conv2DPadding, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilations
        if not isinstance(dilations, int):
            raise NotImplementedError("Only integer dilation is supported.")
        if padding is None:
            raise ValueError("padding should not be None")

    def build(self, x):
        if self.padding in ("reflect", "symmetric"):
            pad = tuple(
                (self.dilation * (s - 1)) // 2 for s in self.kernel_size
            )
            if self.padding == "reflect":
                self.padref = ReflectionPadding2D(padding=pad)
            elif self.padding == "symmetric":
                self.symref = SymmetricPadding2D(padding=pad)
            self.convval = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=(self.stride, self.stride),
                padding="valid",
                dilation_rate=self.dilation,
            )
        else:
            self.convsam = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=(self.stride, self.stride),
                padding="same",
                dilation_rate=self.dilation,
            )

    def call(self, x):
        if self.padding in ("reflect", "symmetric"):
            if self.padding == "reflect":
                x = self.padref(x)
            elif self.padding == "symmetric":
                x = self.symref(x)
            return self.convval(x)
        else:
            return self.convsam(x)


# ==============================================================================
# MODEL BUILDING BLOCKS
# ==============================================================================
def residual_block(x, filters, conv_size=(3, 3), stride=1, dilations=1,
                   relu_alpha=0.2, norm=None, padding=None, force_1d_conv=False):
    """Residual block with optional normalization."""
    in_channels = int(x.shape[-1])
    x_in = x

    if stride > 1:
        x_in = AveragePooling2D(pool_size=(stride, stride))(x_in)
    if force_1d_conv or (filters != in_channels):
        x_in = Conv2D(filters=filters, kernel_size=(1, 1))(x_in)

    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(
        filters=filters, kernel_size=conv_size, stride=stride,
        dilations=dilations, padding=padding
    )(x)
    if norm == "batch":
        x = BatchNormalization()(x)

    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(
        filters=filters, kernel_size=conv_size, stride=1,
        dilations=dilations, padding=padding
    )(x)
    if norm == "batch":
        x = BatchNormalization()(x)

    x = Add()([x, x_in])
    return x


def const_upscale_block(const_input, steps, filters):
    """Downscale constant fields to match forecast resolution."""
    const_output = const_input
    for step in steps:
        const_output = Conv2D(
            filters=filters,
            kernel_size=(step, step),
            strides=step,
            padding="valid",
            activation="relu",
        )(const_output)
    return const_output


# ==============================================================================
# GENERATOR MODEL
# ==============================================================================
def generator(mode, arch, downscaling_steps, input_channels, constant_fields,
              filters_gen, latent_variables=1, noise_channels=4,
              conv_size=(3, 3), padding=None, relu_alpha=0.2, norm=None):
    """Build generator model."""

    forceconv = True if arch in ("forceconv", "forceconv-long") else False

    generator_input = Input(shape=(None, None, input_channels), name="lo_res_inputs")
    const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")

    upscaled_const_input = const_upscale_block(const_input, steps=downscaling_steps, filters=filters_gen)

    if mode in ('det', 'VAEGAN'):
        generator_output = concatenate([generator_input, upscaled_const_input])
    elif mode == 'GAN':
        noise_input = Input(shape=(None, None, noise_channels), name="noise_input")
        generator_output = concatenate([generator_input, upscaled_const_input, noise_input])

    for ii in range(3):
        generator_output = residual_block(
            generator_output, filters=filters_gen, conv_size=conv_size, stride=1,
            relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv
        )

    if arch == "forceconv-long":
        for ii in range(3):
            generator_output = residual_block(
                generator_output, filters=filters_gen, conv_size=conv_size, stride=1,
                relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv
            )

    block_channels = [2 * filters_gen] * (len(downscaling_steps) - 1) + [filters_gen]
    for ii, step in enumerate(downscaling_steps):
        generator_output = UpSampling2D(size=(step, step), interpolation='bilinear')(generator_output)
        generator_output = residual_block(
            generator_output, filters=block_channels[ii], conv_size=conv_size, stride=1,
            relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv
        )

    generator_output = concatenate([generator_output, const_input])

    for ii in range(3):
        generator_output = residual_block(
            generator_output, filters=filters_gen, conv_size=conv_size, stride=1,
            relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv
        )

    generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='softplus', name="output")(generator_output)

    if mode == 'GAN':
        model = Model(inputs=[generator_input, const_input, noise_input], outputs=generator_output, name='gen')
        return model
    elif mode == 'det':
        model = Model(inputs=[generator_input, const_input], outputs=generator_output, name='gen')
        return model


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_fcst_norm(constants_path, mode="auto", gefs_norm_file=None):
    """Load normalization statistics for forecast fields."""
    if mode == "gefs" and gefs_norm_file:
        with open(gefs_norm_file, "rb") as f:
            fcst_norm = pickle.load(f)
        gefs_fields = {"cape", "pres", "pwat", "tmp", "ugrd", "vgrd", "msl"}
        if gefs_fields.issubset(set(fcst_norm.keys())):
            if "apcp" not in fcst_norm:
                fcst_norm["apcp"] = {"min": 0, "max": 200, "mean": 2, "std": 10}
            print(f"Loaded native GEFS normalization from: {gefs_norm_file}")
            return fcst_norm
        else:
            mode = "auto"

    norm_file = os.path.join(constants_path, "FCSTNorm2018.pkl")
    with open(norm_file, "rb") as f:
        loaded_norm = pickle.load(f)

    if mode == "auto":
        if "pres" in loaded_norm and "tmp" in loaded_norm:
            if "apcp" not in loaded_norm:
                loaded_norm["apcp"] = {"min": 0, "max": 200, "mean": 2, "std": 10}
            return loaded_norm
        elif "sp" in loaded_norm and "t2m" in loaded_norm:
            mode = "ifs_mapped"

    if mode == "ifs_mapped":
        fcst_norm = {
            "cape": loaded_norm.get("cape", {"min": 0, "max": 12000, "mean": 400, "std": 650}),
            "pres": loaded_norm.get("sp", {"min": 65000, "max": 104000, "mean": 95000, "std": 5500}),
            "pwat": loaded_norm.get("tcwv", {"min": 0, "max": 80, "mean": 31, "std": 13}),
            "tmp": loaded_norm.get("t2m", {"min": 270, "max": 322, "mean": 298, "std": 5.6}),
            "ugrd": loaded_norm.get("u700", {"min": -35, "max": 33, "mean": -1.3, "std": 5}),
            "vgrd": loaded_norm.get("v700", {"min": -37, "max": 33, "mean": -0.5, "std": 3.6}),
            "msl": {
                "min": loaded_norm.get("sp", {}).get("min", 65000) + 30000,
                "max": loaded_norm.get("sp", {}).get("max", 104000) + 1000,
                "mean": 101325,
                "std": loaded_norm.get("sp", {}).get("std", 5500)
            },
            "apcp": loaded_norm.get("tp", {"min": 0, "max": 25, "mean": 0.1, "std": 0.2}),
        }
        print("Loaded normalization with IFS->GEFS field mapping")
        return fcst_norm

    return loaded_norm


def load_hires_constants(constants_path, batch_size=1):
    """Load elevation and land-sea mask constants."""
    elev = xr.open_dataset(os.path.join(constants_path, "elev.nc"))
    lsm = xr.open_dataset(os.path.join(constants_path, "lsm.nc"))

    elev_var = list(elev.data_vars)[0]
    lsm_var = list(lsm.data_vars)[0]

    elev_data = elev[elev_var].values
    elev_data = elev_data / 10000.0  # Normalise elevation (metres) to O(1)
    lsm_data = lsm[lsm_var].values

    constants = np.stack([elev_data, lsm_data], axis=-1)
    constants = np.expand_dims(constants, axis=0)
    constants = np.tile(constants, (batch_size, 1, 1, 1))

    return constants.astype(np.float32)


def denormalise(data):
    """Convert from log-space to mm/h, capped at 100 mm/h."""
    return np.minimum(np.power(10.0, data) - 1.0, 100.0)


class NoiseGenerator:
    """Generator for noise input to the GAN."""
    def __init__(self, shape, batch_size=1, random_seed=None):
        self.shape = shape
        self.batch_size = batch_size
        if random_seed is not None:
            np.random.seed(random_seed)

    def __call__(self):
        return np.random.normal(size=(self.batch_size,) + self.shape).astype(np.float32)


# ==============================================================================
# OUTPUT FILE CREATION
# ==============================================================================
def create_output_file(nc_out_path, ensemble_members):
    """Create NetCDF output file with proper structure."""
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 6-hour rainfall ensemble members in the ICPAC region."

    rootgrp.createDimension("latitude", len(latitude))
    rootgrp.createDimension("longitude", len(longitude))
    rootgrp.createDimension("member", ensemble_members)
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)

    lat_var = rootgrp.createVariable("latitude", "f4", ("latitude",))
    lat_var.units = "degrees_north"
    lat_var[:] = latitude

    lon_var = rootgrp.createVariable("longitude", "f4", ("longitude",))
    lon_var.units = "degrees_east"
    lon_var[:] = longitude

    member_var = rootgrp.createVariable("member", "i4", ("member",))
    member_var.units = "ensemble member"
    member_var[:] = range(1, ensemble_members + 1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time", "f4", ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable(
        "fcst_valid_time", "f4", ("time", "valid_time")
    )
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["precipitation"] = rootgrp.createVariable(
        "precipitation", "f4",
        ("time", "member", "valid_time", "latitude", "longitude"),
        compression="zlib",
        chunksizes=(1, 1, 1, len(latitude), len(longitude))
    )
    netcdf_dict["precipitation"].units = "mm h**-1"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    return netcdf_dict


# ==============================================================================
# MAIN INFERENCE
# ==============================================================================
def run_inference():
    """Main inference function."""

    print("=" * 60)
    print("cGAN GEFS Inference (Raw NetCDF Method)")
    print("=" * 60)

    model_folder = CONFIG["model_folder"]
    checkpoint = CONFIG["checkpoint"]
    input_folder = CONFIG["input_folder"]
    constants_path = CONFIG["constants_path"]
    output_folder = CONFIG["output_folder"]
    dates = CONFIG["dates"]
    start_hour = CONFIG["start_hour"]
    end_hour = CONFIG["end_hour"]
    ensemble_members = CONFIG["ensemble_members"]

    print(f"Model: {model_folder}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Input: {input_folder}")
    print(f"Dates: {dates}")
    print(f"Hours: {start_hour} to {end_hour}")
    print(f"Ensemble members: {ensemble_members}")
    print("=" * 60)

    # Set GPU mode
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

    # Load normalization statistics
    normalization_mode = CONFIG.get("normalization_mode", "auto")
    gefs_norm_file = CONFIG.get("gefs_norm_file", None)
    fcst_norm = load_fcst_norm(constants_path, mode=normalization_mode, gefs_norm_file=gefs_norm_file)

    # Load constants
    network_const_input = load_hires_constants(constants_path, batch_size=1)

    # Read model setup params
    config_path = os.path.join(model_folder, "setup_params.yaml")
    with open(config_path, "r") as f:
        setup_params = yaml.safe_load(f)

    arch = setup_params["MODEL"]["architecture"]
    padding = setup_params["MODEL"]["padding"]
    filters_gen = setup_params["GENERATOR"]["filters_gen"]
    noise_channels = setup_params["GENERATOR"]["noise_channels"]
    latent_variables = setup_params["GENERATOR"]["latent_variables"]

    input_channels = 4 * len(all_fcst_fields)
    constant_fields = 2
    downscaling_steps = [1]

    # Build model
    print("\nBuilding model...")
    gen = generator(
        mode="GAN",
        arch=arch,
        downscaling_steps=downscaling_steps,
        input_channels=input_channels,
        constant_fields=constant_fields,
        filters_gen=filters_gen,
        noise_channels=noise_channels,
        latent_variables=latent_variables,
        padding=padding
    )

    # Load weights
    weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07d}.h5")
    print(f"Loading weights: {weights_fn}")
    gen.load_weights(weights_fn)
    print("Model loaded successfully!")

    # Process dates
    dates_array = np.asarray(dates, dtype='datetime64[ns]')
    valid_times = np.arange(start_hour, end_hour + 1, HOURS)

    for day in dates_array:
        d = datetime(
            day.astype('datetime64[D]').astype(object).year,
            day.astype('datetime64[D]').astype(object).month,
            day.astype('datetime64[D]').astype(object).day
        )

        print(f"\nProcessing: {d.year}-{d.month:02d}-{d.day:02d}")

        # Input/output paths
        input_folder_year = os.path.join(input_folder, str(d.year))
        output_folder_year = os.path.join(output_folder, str(d.year))
        pathlib.Path(output_folder_year).mkdir(parents=True, exist_ok=True)

        nc_out_path = os.path.join(
            output_folder_year, f"GAN_{d.year}{d.month:02d}{d.day:02d}.nc"
        )

        # Create output file
        netcdf_dict = create_output_file(nc_out_path, ensemble_members)
        netcdf_dict["time_data"][0] = date2num(
            d, units="hours since 1900-01-01 00:00:00.0"
        )

        # Loop over time chunks
        for out_time_idx, in_time_idx in enumerate(range(start_hour // HOURS, end_hour // HOURS)):
            print(f"\n  Valid time: +{valid_times[out_time_idx]}h")

            netcdf_dict["valid_time_data"][0, out_time_idx] = date2num(
                d + timedelta(hours=int(valid_times[out_time_idx])),
                units="hours since 1900-01-01 00:00:00.0"
            )

            field_arrays = []

            # Load and normalize each field
            for field in all_fcst_fields:
                input_file = f"{field}_{d.year}.nc"
                nc_in_path = os.path.join(input_folder_year, input_file)

                nc_file = xr.open_dataset(nc_in_path)
                nc_file = nc_file.sel({"time": day})

                # Get step indices (hours 1-80, need indices for hour pairs)
                # For forecast hour 30, need steps at hour 24 and 30
                # in_time_idx = 5 (30//6) -> need steps 4 and 5 (indices 23 and 29)
                # But our steps start at 1, so:
                # For hours 30-36: steps at indices 29 and 35 (hour 30 and 36)
                hour_idx_1 = valid_times[out_time_idx] - 1  # hour index (0-based from hour 1)
                hour_idx_2 = valid_times[out_time_idx] + HOURS - 1

                # Select two consecutive timesteps for this valid time
                nc_file = nc_file.isel({"step": [hour_idx_1, hour_idx_2]})

                short_name = [var for var in nc_file.data_vars][0]

                # Shape: (member, step, lat, lon)
                data = nc_file[short_name].values  # (member, step, lat, lon)
                n_members = data.shape[0]
                n_steps = data.shape[1]

                # Reshape for resizing: combine member and step dims -> (member*step, lat, lon)
                # Then add channel dim for tf.image.resize
                data_flat = data.reshape(n_members * n_steps, data.shape[2], data.shape[3])
                data_flat = np.expand_dims(data_flat, axis=-1)  # (member*step, lat, lon, 1)

                # Resize to model input size (384 x 352)
                data_resized = tf.image.resize(data_flat, [384, 352]).numpy()

                # Reshape back: (member*step, 384, 352, 1) -> (member, step, 384, 352)
                data_resized = data_resized[:, :, :, 0].reshape(n_members, n_steps, 384, 352)

                # Transpose to (384, 352, member, step) for processing
                data = np.transpose(data_resized, (2, 3, 0, 1))

                # Apply non-negativity constraint
                if field in nonnegative_fields:
                    data = np.maximum(data, 0.0)

                # Normalize based on field type and compute ensemble mean/std
                if field == "apcp":
                    data = np.log10(1 + data)
                    data_mean = np.nanmean(data, axis=-2)  # Mean over members: (384, 352, step)
                    data_std = np.nanstd(data, axis=-2)    # Std over members: (384, 352, step)
                    data = np.concatenate([
                        data_mean[..., [0]], data_std[..., [0]],
                        data_mean[..., [1]], data_std[..., [1]]
                    ], axis=-1)
                elif field in ["msl", "pres", "tmp"]:
                    data -= fcst_norm[field]["mean"]
                    data /= fcst_norm[field]["std"]
                    data_mean = np.nanmean(data, axis=-2)
                    data_std = np.nanstd(data, axis=-2)
                    data = np.concatenate([
                        data_mean[..., [0]], data_std[..., [0]],
                        data_mean[..., [1]], data_std[..., [1]]
                    ], axis=-1)
                elif field in nonnegative_fields:
                    data /= fcst_norm[field]["max"]
                    data_mean = np.nanmean(data, axis=-2)
                    data_std = np.nanstd(data, axis=-2)
                    data = np.concatenate([
                        data_mean[..., [0]], data_std[..., [0]],
                        data_mean[..., [1]], data_std[..., [1]]
                    ], axis=-1)
                elif field in ["ugrd", "vgrd"]:
                    data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
                    data_mean = np.nanmean(data, axis=-2)
                    data_std = np.nanstd(data, axis=-2)
                    data = np.concatenate([
                        data_mean[..., [0]], data_std[..., [0]],
                        data_mean[..., [1]], data_std[..., [1]]
                    ], axis=-1)

                field_arrays.append(data)

            # Prepare network inputs
            network_fcst_input = np.concatenate(field_arrays, axis=-1)
            network_fcst_input = np.expand_dims(network_fcst_input, axis=0)

            noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=1)

            # Generate ensemble members
            progbar = Progbar(ensemble_members)
            for ii in range(ensemble_members):
                gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
                gan_prediction = gen.predict(gan_inputs, verbose=False)
                netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(
                    gan_prediction[0, :, :, 0]
                )
                progbar.add(1)

        netcdf_dict["rootgrp"].close()
        print(f"\n  Output saved: {nc_out_path}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
