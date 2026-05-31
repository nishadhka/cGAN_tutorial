import yaml
import numpy as np
import xarray as xr
import tensorflow as tf
from properscoring import crps_ensemble
from tqdm import tqdm
import time

import sys
import os
import importlib
sys.path.append("/home/n/nath/xarray_batcher/")

from xarray_batcher.get_fcst_and_truth import get_all

from datetime import datetime, timedelta

# %%
# Define the latitude and longitude arrays for later
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)

# %%
# Open and parse forecast.yaml
#fcstyaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast.yaml")
fcstyaml_path = "../config/forecast_gfs.yaml"
with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

def crps_xr(observation,forecast):

    ## perform conversion within crps_ensemble_calculation
    crps = crps_ensemble(observation,forecast)

    return crps

out_path = "/network/group/aopp/predict/AWH024_COOPERNATH_IFS/cGAN_gefs/predictions/"#fcst_params["OUTPUT"]["folder"]
if not os.path.exists(out_path):
    os.makedirs(out_path+'crps/')

for year in [2023,2024]:
    
    valid_times = np.arange(30,54,6)
    if year==2023:
        dates = np.arange("%i-06-01"%year,"%i-12-30"%(year),np.timedelta64(1,"D"),dtype="datetime64[D]")

    if year==2024:
        dates = np.arange("%i-01-01"%year,"%i-09-01"%year,np.timedelta64(1,"D"),dtype="datetime64[D]")

    ds_truth = get_all([year], model='truth', log_precip=False, batch_type='6-hourly')

    for d in tqdm(dates):

        print('Evaluating')
        file_name = f"{d.astype(object).year}/GAN_{d.astype(object).year}{d.astype(object).month:02}{d.astype(object).day:02}.nc"
        if not os.path.exists(out_path+'crps/'):
            os.makedirs(out_path+'crps/')
        
        start_time = time.time()
        preds = xr.open_dataset("/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/cGAN/predictions/gfs/"+file_name,
                                decode_times=False).drop_duplicates(dim="time")
        preds['latitude'] = ds_truth.lat.values
        preds['longitude'] = ds_truth.lon.values

        valid_dates = [d.astype('datetime64[ns]')+np.timedelta64(dt,'h') for dt in [30,36,42,48]]
        preds['time'] = [d.astype('datetime64[ns]')]
        preds['valid_time'] = valid_dates

        truth_vals = ds_truth.sel({'time':valid_dates}).precipitation.values[None,:,:,:]
        truth_vals = xr.DataArray(data=truth_vals, dims=['time','valid_time','latitude','longitude'],
                                  coords = {'time':[d.astype('datetime64[ns]')],
                                           'valid_time':valid_dates,
                                           'latitude':ds_truth.lat.values,
                                           'longitude':ds_truth.lon.values
                                           }).rename('precipitation')
        
        crps = xr.apply_ufunc(crps_xr,
                              truth_vals,
                              preds.precipitation,
                              input_core_dims = [[],['member']],
                              vectorize=True).rename('crps')
        
        file_name = f'crps/GFS_crps_2024.zarr'
        if os.path.exists(out_path+file_name):
            crps.to_zarr(out_path+file_name,mode='a-',append_dim='time')
        else:
            crps.to_zarr(out_path+file_name,mode='w')
        del crps

                