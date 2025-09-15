# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_fcst_fields

import os
import pathlib
import yaml

import netCDF4 as nc
import numpy as np
from tensorflow.keras.utils import Progbar

import sys 
sys.path.insert("../data/")
from data_old import HOURS, all_fcst_fields, accumulated_fields, nonnegative_fields, fcst_norm, logprec, denormalise, load_hires_constants

import sys
sys.path.insert(1,"../")
fron config import set_gpu_mode, get_data_paths, read_downscaling_factor
from model import NoiseGenerator
from setupmodel import setup_model

# Define the latitude and longitude arrays for later
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)

# Some setup
set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
data_paths = get_data_paths()  # need the constants directory
downscaling_steps = read_downscaling_factor()["steps"]
assert fcst_norm is not None

# Open and parse forecast.yaml
fcstyaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast.yaml")
with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

model_folder = fcst_params["MODEL"]["folder"]
checkpoint = fcst_params["MODEL"]["checkpoint"]
input_folder = fcst_params["INPUT"]["folder"]
input_file = fcst_params["INPUT"]["file"]
start_hour = fcst_params["INPUT"]["start_hour"]
end_hour = fcst_params["INPUT"]["end_hour"]
output_folder = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]

assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
assert end_hour % HOURS == 0, f"end_hour must be divisible by {HOURS}"

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]  # TODO: avoid setting up discriminator in forecast mode?
constant_fields = 2

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = 4*len(all_fcst_fields)

model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=constant_fields,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1)  # 1 x lats x lons x 2

# Open input netCDF file
nc_in_path = os.path.join(input_folder, input_file)
nc_in = nc.Dataset(nc_in_path, mode="r")

# Create output netCDF file
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
nc_out_path = os.path.join(output_folder, f"GAN_{input_file}")  # e.g., GAN_20200315.nc


def create_output_file(nc_out_path):
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 6-hour rainfall ensemble members in the ICPAC region."

    # Create output file dimensions
    rootgrp.createDimension("latitude", len(latitude))
    rootgrp.createDimension("longitude", len(longitude))
    rootgrp.createDimension("member", ensemble_members)
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)

    # Create variables
    latitude_data = rootgrp.createVariable("latitude",
                                           "f4",
                                           ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude     # Write the latitude data

    longitude_data = rootgrp.createVariable("longitude",
                                            "f4",
                                            ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude   # Write the longitude data

    ensemble_data = rootgrp.createVariable("member",
                                           "i4",
                                           ("member",))
    ensemble_data.units = "ensemble member"
    ensemble_data[:] = range(1, ensemble_members+1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time",
                                                      "f4",
                                                      ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable("fcst_valid_time",
                                                            "f4",
                                                            ("time", "valid_time"))
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["precipitation"] = rootgrp.createVariable("precipitation",
                                                          "f4",
                                                          ("time", "member", "valid_time",
                                                           "latitude", "longitude"),
                                                          compression="zlib",
                                                          chunksizes=(1, 1, 1, len(latitude), len(longitude)))
    netcdf_dict["precipitation"].units = "mm h**-1"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    return netcdf_dict


netcdf_dict = create_output_file(nc_out_path)
netcdf_dict["time_data"][0] = nc_in["time"][0]

# loop over time chunks. output forecasts may not start from hour 0, so
# generate output and input valid time indices using enumerate(...)
for out_time_idx, in_time_idx in enumerate(range(start_hour//HOURS, end_hour//HOURS)):
    # copy across valid_time from input file
    netcdf_dict["valid_time_data"][0, out_time_idx] = nc_in["valid_time"][out_time_idx]
    field_arrays = []

    # the contents of the next loop are v. similar to load_fcst from data.py,
    # but not quite the same, since that has different assumptions on how the
    # forecast data is stored.  TODO: unify the data normalisation between these?
    for field in all_fcst_fields:
        # nc_in[field] has shape 1 x 50 x 29 x 384 x 352
        # corresponding to n_forecasts x n_ensemble_members x n_valid_times x n_lats x n_lons

        # grab start and end of 6-hour block in one operation
        temp_data = nc_in[f"{field}_ensemble_mean"][0, :, in_time_idx:in_time_idx+2, :, :]  # ens x 2 x lat x lon
        temp_start = temp_data[:, 0, :, :]  # ens x lat x lon, start of timestep
        temp_end = temp_data[:, 1, :, :]  # ens x lat x lon, end of timestep

        # return 4 channels per field
        if field in accumulated_fields:
            temp_diff = temp_end - temp_start  # ens x lat x lon
            temp_mean = np.mean(temp_diff, axis=0)  # lat x lon
            temp_std = np.std(temp_diff, axis=0, ddof=1)
            zeros = np.zeros(temp_mean.shape)
            data = np.stack([temp_mean,
                             temp_std,
                             zeros,
                             zeros],
                            axis=-1)  # lat x lon x 4
        else:
            temp_start_mean = np.mean(temp_start, axis=0)  # lat x lon
            temp_start_std = np.std(temp_start, axis=0, ddof=1)
            temp_end_mean = np.mean(temp_end, axis=0)
            temp_end_std = np.std(temp_end, axis=0, ddof=1)
            data = np.stack([temp_start_mean,
                             temp_start_std,
                             temp_end_mean,
                             temp_end_std],
                            axis=-1)  # lat x lon x 4

        # perform normalisation on forecast data
        if field in nonnegative_fields:
            data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

        if field in ["tp", "cp"]:
            # precip is measured in metres, so multiply to get mm
            data *= 1000
            data /= HOURS  # convert to mm/hr
        elif field in accumulated_fields:
            # for all other accumulated fields [just ssr for us]
            data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

        if field in ["tp", "cp"]:
            data = logprec(data, True)
        else:
            # apply transformation to make fields O(1), based on historical
            # forecast data from one of the training years
            if field in ["mcc"]:
                # already 0-1
                pass
            elif field in ["sp", "t2m"]:
                # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
                data[:, :, 0] -= fcst_norm[field]["mean"]
                data[:, :, 2] -= fcst_norm[field]["mean"]
                data /= fcst_norm[field]["std"]
            elif field in nonnegative_fields:
                data /= fcst_norm[field]["max"]
            else:
                # winds
                data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])

        field_arrays.append(data)

    network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 4*len(all_fcst_fields)
    network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 4*len(...)

    noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
    noise_gen = NoiseGenerator(noise_shape, batch_size=1)
    progbar = Progbar(ensemble_members)
    for ii in range(ensemble_members):
        gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
        gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
        netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(gan_prediction[0, :, :, 0])
        progbar.add(1)

nc_in.close()
netcdf_dict["rootgrp"].close()
