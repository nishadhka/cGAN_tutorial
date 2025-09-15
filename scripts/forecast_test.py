import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
from noise import NoiseGenerator
from data import denormalise

import setupmodel
import importlib
importlib.reload(setupmodel)
from setupmodel import setup_model

import yaml
import read_config
from tensorflow.keras.utils import Progbar
from data import (
    HOURS,
    all_fcst_fields,
    accumulated_fields,
    nonnegative_fields,
    fcst_norm,
    logprec,
    denormalise,
    load_hires_constants,
)

import xarray as xr
import numpy as np

model_folder='logfile/'

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

downscaling_steps = read_config.read_downscaling_factor()["steps"]

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
weights_fn = os.path.join(model_folder, 'models', f"gen_weights-0115200.h5")
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
network_const_input = load_hires_constants(batch_size=1)
field_arrays = []

input_folder_year = "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/netcdf/2023/"
ensemble_members = 50
# the contents of the next loop are v. similar to load_fcst from data.py,
# but not quite the same, since that has different assumptions on how the
# forecast data is stored.  TODO: unify the data normalisation between these?
for field in all_fcst_fields.keys():
    # Original:
    # nc_in[field] has shape 1 x 50 x 29 x 384 x 352
    # corresponding to n_forecasts x n_ensemble_members x n_valid_times x n_lats x n_lons
    # Ensemble mean:
    # nc_in[field] has shape len(nc_in["time"]) x 29 x 384 x 352
    
    # Open input netCDF file
    input_file = f"{all_fcst_fields[field]}.zarr"
    #input_file = 'IFS_20180606_00Z.nc'
    #input_file = f'IFS_{d.year}{d.month:02}{d.day:02}_00Z.nc'
    nc_in_path = os.path.join(input_folder_year, input_file)
    nc_in = xr.open_zarr(nc_in_path)#.sel({'time':f"{d.year}-{d.month:02}-{d.day:02}"})#nc.Dataset(nc_in_path, mode="r")

    #print(nc_in.time.values)
    data = np.moveaxis(np.squeeze(nc_in.isel({"time":[35]}).to_dataarray().values),0,-1)
    
    field_arrays.append(data)

network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 4*len(all_fcst_fields)
network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 4*len(...)

noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
noise_gen = NoiseGenerator(noise_shape, batch_size=1)
progbar = Progbar(ensemble_members)
for ii in range(ensemble_members):
    gan_inputs = [np.asarray(network_fcst_input), np.asarray(network_const_input), np.asarray(noise_gen())]
    gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
    netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(gan_prediction[0, :, :, 0])
    progbar.add(1)
