""" File for handling data loading and saving. """
import os
import glob
import time
import datetime
import pickle
import tensorflow as tf
import numpy as np

import netCDF4 as nc
import xarray as xr
import xesmf
import h5py

import sys
sys.path.insert(1,"../")
from config import get_data_paths


data_paths = get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = ['cape','pres','pwat','tmp','ugrd','vgrd','msl','apcp']#'hgt'
nonnegative_fields = ['cape','msl','pres','pwat','tmp']

HOURS = 6  # 6-hr data modified to 24 hour

lat_reg_b = np.arange(-14.0,25.25,0.25)[::-1]- 0.125

lat_reg = 0.5 * (lat_reg_b[1:] + lat_reg_b[:-1])

lon_reg_b = np.arange(19,55,0.25) - 0.125

lon_reg = 0.5 * (lon_reg_b[1:] + lon_reg_b[:-1])

data_path = glob.glob(TRUTH_PATH + "*.nc")

IMERG_data_dir = "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/"
IMERG_file_name = f"{IMERG_data_dir}/2018/Jan/3B-HHR.MS.MRG.3IMERG.20180101-S000000-E002959.0000.V07B.HDF5"
        
# HDF5 in the ICPAC region
h5_file = h5py.File(IMERG_file_name)
lat_reg_IMERG = h5_file['Grid']["lat"][763:1147]
lon_reg_IMERG = h5_file['Grid']["lon"][1991:2343]


lat_reg_IMERG_b = np.append((lat_reg_IMERG - 0.05), lat_reg_IMERG[-1] + 0.05)
# print(lat_reg_IMERG_b)
lon_reg_IMERG_b = np.append((lon_reg_IMERG - 0.05), lon_reg_IMERG[-1] + 0.05)
# print(lon_reg_IMERG_b)
# utility function; generator to iterate over a range of dates


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 100 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1.0, 100.0)


def logprec(y, log_precip=False):
    if log_precip:
        return np.log10(1.0 + y)
    else:
        return y


def get_dates(year, start_hour, end_hour):
    """
    Returns list of valid forecast start dates for which 'truth' data
    exists, given the other input parameters. If truth data is not available
    for certain days/hours, this will not be the full year. Dates are returned
    as a list of YYYYMMDD strings.

    Parameters:
        year (int): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
    """
    # sanity checks for our dataset
    assert year in (2017,2018, 2019, 2020, 2021, 2022, 2023)
    assert start_hour >= 0
    assert end_hour <= 168
    assert start_hour % HOURS == 0
    assert end_hour % HOURS == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist
    valid_dates = []
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(
        year + 1, 1, end_hour // 24 + 2
    )  # go a bit into following year
    #print(start_date, end_date)
    for curdate in daterange(start_date, end_date):
        datestr = curdate.strftime("%Y%m%d")

        datestr_true = (curdate + datetime.timedelta(days=1)).strftime("%Y%m%d")
        for hr in [6]:
            fname = f"{datestr_true}_{hr:02}"
            # print(os.path.join(FCST_PATH, test_file))
            if os.path.exists(
                os.path.join(TRUTH_PATH, str(year), f"{fname}.nc")):
                valid_dates.append(datestr)
    return valid_dates


def load_truth_and_mask(date, time_idx, log_precip=False, consolidated=False):
    """
    Returns a single (truth, mask) item of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index
        log_precip: whether to apply log10(1+x) transformation
    """
    # convert date and time_idx to get the correct truth file
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
    valid_dt = fcst_date + datetime.timedelta(hours=int(time_idx))  # needs to change for 12Z forecasts
    
    if consolidated:

        d = valid_dt
        d_end = d + datetime.timedelta(hours=6)
        
        # A single IMERG data file to get latitude and longitude
        IMERG_data_dir = '/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07'
        IMERG_file_name = f"{IMERG_data_dir}/2024/Apr/3B-HHR.MS.MRG.3IMERG.20230429-S063000-E065959.0390.V07B.HDF5"
        
        # HDF5 in the ICPAC region
        h5_file = h5py.File(IMERG_file_name)
        latitude = h5_file['Grid']["lat"][763:1147]
        longitude = h5_file['Grid']["lon"][1991:2343]
        h5_file.close()
    
        # The 6h average rainfall
        rain_IMERG_6h = np.zeros((len(longitude), len(latitude)))
        
        start_time = time.time()

        while (d<d_end):
            
            # Load an IMERG file with the current date
            d2 = d + datetime.timedelta(seconds=30*60-1)
            
            # Number of minutes since 00:00
            count = int((d - datetime.datetime(d.year, d.month, d.day)).seconds / 60)
            IMERG_file_name = glob.glob(f"{IMERG_data_dir}/{d.year}/{d.strftime('%b')}/3B-HHR-L.MS.MRG.3IMERG.{d.year}{d.month:02d}{d.day:02d}-S{d.hour:02d}{d.minute:02d}00-E{d2.hour:02d}{d2.minute:02d}{d2.second:02d}.{count:04d}.V06*.HDF5")[0]
            
            h5_file = h5py.File(IMERG_file_name)
            times = h5_file['Grid']["time"][:]
            rain_IMERG = h5_file['Grid']["precipitationCal"][0,1991:2343,763:1147]
            h5_file.close()
    
            # Check the time is correct
            if (d != datetime.datetime(1970,1,1) + datetime.timedelta(seconds=int(times[0]))):
                print(f"Incorrect time for {d}")
    
            # Accumulate the rainfall
            rain_IMERG_6h += rain_IMERG
    
            # Move to the next date
            d += datetime.timedelta(minutes=30)
            
        # Normalise to mm/h
        rain_IMERG_6h /= (2*6)
        rain_IMERG_6h = np.moveaxis(rain_IMERG_6h, [0, 1], [1,0])
        #print(rain_IMERG_6h.shape)
        mask = np.full(rain_IMERG_6h.shape, False, dtype=bool)
        if log_precip:
            return np.log10(1 + rain_IMERG_6h), mask
        else:
            return rain_IMERG_6h, mask

    else:
        fname = valid_dt.strftime("%Y%m%d_%H")
        year_folder = valid_dt.strftime("%Y")
        data_path = glob.glob(TRUTH_PATH + f"{year_folder}/{fname}.nc")
        ds = xr.open_dataset(data_path[0])
    
        da = ds["precipitation"]
        y = da.values
        ds.close()
    
        # mask: False for valid truth data, True for invalid truth data
        # (compatible with the NumPy masked array functionality)
        # if all data is valid:
        mask = np.full(y.shape, False, dtype=bool)
    
        if log_precip:
            return np.log10(1 + y), mask
        else:
            return y, mask


def load_hires_constants(batch_size=1):
    oro_path = os.path.join(CONSTANTS_PATH, "elev.nc")
    df = xr.load_dataset(oro_path)
    # Orography in m.  Divide by 10,000 to give O(1) normalisation
    z = df["elevation"].values
    z /= 10000.0
    df.close()

    lsm_path = os.path.join(CONSTANTS_PATH, "lsm.nc")
    df = xr.load_dataset(lsm_path)
    # LSM is already 0:1
    lsm = df["lsm"].values
    df.close()

    temp = np.stack([z, lsm], axis=-1)  # shape H x W x 2
    return np.repeat(
        temp[np.newaxis, ...], batch_size, axis=0
    )  # shape batch_size x H x W x 2


def load_fcst_truth_batch(
    dates_batch,
    time_idx_batch,
    fcst_fields=all_fcst_fields,
    log_precip=False,
    norm=False,
    consolidated=False
):
    """
    Returns a batch of (forecast, truth, mask) data, although usually the batch size is 1
    Parameters:
        dates_batch (iterable of strings): Dates of forecasts
        time_idx_batch (iterable of ints): Corresponding 'valid_time' array indices
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related forecast fields, and truth
        norm (bool): Whether to apply normalisation to forecast fields to make O(1)
    """
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    for time_idx, date in zip(time_idx_batch, dates_batch):
        batch_x_temp = load_fcst_stack(
            fcst_fields, date, time_idx, log_precip=log_precip, norm=norm, consolidated=consolidated
        )
        batch_x_temp[np.isnan(batch_x_temp)] = 0
        batch_x_temp[np.isinf(batch_x_temp)] = 0
        batch_x.append(batch_x_temp)
        truth, mask = load_truth_and_mask(date, time_idx, log_precip=log_precip, consolidated=consolidated)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)

def load_fcst(field, date, time_idx, log_precip=False, norm=False, consolidated=None):
    yearstr = date[:4]
    year = int(yearstr)
    ds_path = os.path.join(FCST_PATH, yearstr, f"{field}_{yearstr}.zarr")
    
    nc_file = xr.open_dataset(ds_path, engine="zarr", consolidated=False)
    nc_file = nc_file.sel(
            {"time":date, "step":[np.timedelta64(time_idx,'h'),np.timedelta64(time_idx+HOURS,'h')]}
        )
    short_name = [var for var in nc_file.data_vars][0]
    data = np.moveaxis(np.squeeze(nc_file[short_name].values),0,-1)
    data=tf.constant(data)
    data = tf.image.resize(data,[384,352]).numpy()
                          
    if field in nonnegative_fields:
        data = np.maximum(data, 0.0)

    if norm:
        if field in ["apcp"]:
            data = np.log10(1+data)
            data_mean = np.moveaxis(np.nanmean(data,axis=-1),0,-1)
            data_std = np.moveaxis(np.nanstd(data,axis=-1),0,-1)
            data = np.concatenate([data_mean[...,[0]],data_std[...,[0]],
                                  data_mean[...,[1]],data_std[...,[1]]],axis=-1)
        if field in ["msl", "pres","tmp"]:
            # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
            data -= fcst_norm[field]["mean"]
            data /= fcst_norm[field]["std"]
            data_mean = np.moveaxis(np.nanmean(data,axis=-1),0,-1)
            data_std = np.moveaxis(np.nanstd(data,axis=-1),0,-1)
            data = np.concatenate([data_mean[...,[0]],data_std[...,[0]],
                                  data_mean[...,[1]],data_std[...,[1]]],axis=-1)
        elif field in nonnegative_fields:
            data /= fcst_norm[field]["max"]
            data_mean = np.moveaxis(np.nanmean(data,axis=-1),0,-1)
            data_std = np.moveaxis(np.nanstd(data,axis=-1),0,-1)
            data = np.concatenate([data_mean[...,[0]],data_std[...,[0]],
                                  data_mean[...,[1]],data_std[...,[1]]],axis=-1)
        elif field in ["ugrd","vgrd"]:
            data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
            data_mean = np.moveaxis(np.nanmean(data,axis=-1),0,-1)
            data_std = np.moveaxis(np.nanstd(data,axis=-1),0,-1)
            data = np.concatenate([data_mean[...,[0]],data_std[...,[0]],
                                  data_mean[...,[1]],data_std[...,[1]]],axis=-1)
    
    return data
    


def load_fcst_stack(fields, date, time_idx, log_precip=False, norm=False, consolidated=False):
    """
    Returns forecast fields, for the given date and time interval.
    Each field returned by load_fcst has two channels (see load_fcst for details),
    then these are concatentated to form an array of H x W x 4*len(fields)
    """
    field_arrays = []
    for f in fields:
        #print(f)
        field_arrays.append(
            load_fcst(f, date, time_idx, log_precip=log_precip, norm=norm, consolidated=consolidated)
        )
    return np.concatenate(field_arrays, axis=-1)


def get_fcst_stats_slow(field, year=2018):
    """
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done via the data loading routines, which is
    slightly inefficient.
    """
    dates = get_dates(year, start_hour=0, end_hour=168)

    mi = 0.0
    mx = 0.0
    dsum = 0.0
    dsqrsum = 0.0
    nsamples = 0
    for datestr in dates:
        for time_idx in range(28):
            data = load_fcst(field, datestr, time_idx)[:, :, 0]
            mi = min(mi, data.min())
            mx = max(mx, data.max())
            dsum += np.mean(data)
            dsqrsum += np.mean(np.square(data))
            nsamples += 1
    mn = dsum / nsamples
    sd = (dsqrsum / nsamples - mn**2) ** 0.5
    return mi, mx, mn, sd


def get_fcst_stats_fast(field, year=2018, model="gefs"):
    """
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done directly from the forecast netcdf file,
    which is somewhat faster, as long as it fits into memory.
    """

    if model == "gfs":
        ds_path = glob.glob(
            FCST_PATH
            + f"gfs{str(year)}*_t00z_f030_f054_{field.replace(' ','-')}_{all_fcst_levels[field]}.zarr"
        )
    
        z2z = [ZarrToZarr(ds).translate() for ds in ds_path]
    
        mode_length = np.array([len(z.keys()) for z in z2z]).flatten()
        modals, counts = np.unique(mode_length, return_counts=True)
        index = np.argmax(counts)
    
        z2zs = [z for z in z2z if len(z.keys()) == modals[index]]
    
        mzz = MultiZarrToZarr(
            z2zs,
            concat_dims=["time"],
            identical_dims=["step", "latitude", "longitude", all_fcst_levels[field]],
        )
    
        ref = mzz.translate()
    
        backend_kwargs = {
            "consolidated": False,
            "storage_options": {
                "fo": ref,
            },
        }
        nc_file = xr.open_dataset(
            "reference://", engine="zarr", backend_kwargs=backend_kwargs
        ).sel({"latitude": lat_reg, "longitude": lon_reg})
        short_name = [var for var in nc_file.data_vars][0]

    else:
        ds_path = os.path.join(FCST_PATH, str(year), f"{field}_{year}.zarr")
        nc_file = xr.open_zarr(ds_path)
    
    short_name = [var for var in nc_file.data_vars][0]
    if field in ["Convective precipitation (water)", "Total precipitation"]:
        # precip is measured in metres, so multiply to get mm
        data = nc_file[short_name].sum("step").values
        
        #data *= 1000 only for IFS
        data /= HOURS  # convert to mm/hr
        data = np.maximum(data, 0.0)  # shouldn't be necessary, but just in case

    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data = nc_file[short_name].sum("step").values
        data /= HOURS * 3600  # convert from a 6-hr difference to a per-second rate

    else:
        ## Format of time, member, step, lat, lon
        data = nc_file[short_name].values

    mi = data.min()
    mx = data.max()
    mn = np.mean(data, dtype=np.float64)
    sd = np.std(data, dtype=np.float64)
    return mi, mx, mn, sd


def gen_fcst_norm(year=2018):
    """
    One-off function, used to generate normalisation constants, which
    are used to normalise the various input fields for training/inference.
    """

    stats_dic = {}
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")

    # make sure we can actually write there, before doing computation!!!
    with open(fcstnorm_path, "wb") as f:
        pickle.dump(stats_dic, f)

    start_time = time.time()
    for field in all_fcst_fields:
        print(field)
        mi, mx, mn, sd = get_fcst_stats_fast(field, year)
        stats_dic[field] = {}
        stats_dic[field]["min"] = mi
        stats_dic[field]["max"] = mx
        stats_dic[field]["mean"] = mn
        stats_dic[field]["std"] = sd
        print(
            "Got normalisation constants for",
            field,
            " in ----",
            time.time() - start_time,
            "s----",
        )

    with open(fcstnorm_path, "wb") as f:
        pickle.dump(stats_dic, f)


def load_fcst_norm(year=2018):
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, "rb") as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2018)
    # print(fcst_norm.keys())
except:  # noqa
    fcst_norm = None
