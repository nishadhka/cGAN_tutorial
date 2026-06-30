#!/usr/bin/env python3
"""
make_local_tfrecords.py — build a small LOCAL test set of .tfrecords with the
real tfrecords_generator.write_data (14-field config), using synthetic data so
it needs no network and no 388 GB download.

Run:  micromamba run -n tf215gpu python make_local_tfrecords.py
Out:  ./local_tfrecords_test/tfrecords/<YEAR>_1.{0,1,2,3}.tfrecords
"""
import sys
import glob
import datetime
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
OUT = HERE / "local_tfrecords_test"
DATA = OUT / "data"
REPO = Path("/scratch/notebook/SEWAA-forecasts-RFE2/SEWAA-forecasts/"
            "24h_accumulations/cGAN/dsrnngan")
sys.path.insert(0, str(REPO))

YEAR = 2021
N_DAYS, N_VT, H, W = 3, 7, 384, 352
FIELDS = ["cape", "cp", "mcc", "sp", "ssr", "t2m", "tciw", "tclw",
          "tcrw", "tcw", "tcwv", "tp", "u700", "v700"]
# rough realistic magnitudes per field so normalisation + bins behave
SCALE = {"cape": 500, "cp": 0.002, "mcc": 1, "sp": 101000, "ssr": 1e6,
         "t2m": 300, "tciw": 0.5, "tclw": 0.5, "tcrw": 1.0, "tcw": 40,
         "tcwv": 30, "tp": 0.003, "u700": 8, "v700": 8}

IFS = DATA / "IFS_training" / str(YEAR)
RFE = DATA / "RFE" / str(YEAR)
CON = DATA / "cGAN_data"
TFR = DATA / "tfrecords"
for d in (IFS, RFE, CON, TFR):
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
lat = np.linspace(24.65, -13.65, H).astype("float32")
lon = np.linspace(19.15, 54.25, W).astype("float32")


def synth_field(name):
    s = SCALE[name]
    mean = np.abs(rng.normal(0, 1, (N_DAYS, N_VT, H, W))).astype("float32") * s
    if name in ("u700", "v700"):  # winds can be negative
        mean = rng.normal(0, s, (N_DAYS, N_VT, H, W)).astype("float32")
    if name in ("cp", "ssr", "tp"):  # accumulated -> monotone in valid_time
        mean = np.cumsum(np.abs(mean), axis=1).astype("float32")
    sd = (np.abs(rng.normal(0, 0.1, mean.shape)) * s).astype("float32")
    xr.Dataset(
        {f"{name}_mean": (("time", "valid_time", "latitude", "longitude"), mean),
         f"{name}_sd":   (("time", "valid_time", "latitude", "longitude"), sd)},
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(IFS / f"{name}.nc")
    return mean


print(f"[synth] {len(FIELDS)} fields x {N_DAYS}d x {N_VT}vt @ {H}x{W}")
tp = None
for f in FIELDS:
    m = synth_field(f)
    if f == "tp":
        tp = m

# truth (mm/day) for D+1 of each forecast date. A smooth W->E gradient from
# ~0 to ~4 mm/day (= ~0 to ~0.17 mm/hr) so patches land across ALL 4 rain bins
# (thresholds 0.0059/0.0362/0.0761 mm/hr), plus light noise for texture.
for di in range(N_DAYS):
    td = datetime.date(YEAR, 1, 1) + datetime.timedelta(days=di + 1)
    grad = np.linspace(-3.0, 4.0, W)[None, :] * np.ones((H, 1))  # mm/day; <0 clipped to 0
    grad = np.clip(grad, 0.0, None)                              # wide dry band -> bin 0
    grad = np.roll(grad, di * (W // N_DAYS), axis=1)             # shift per day
    truth = np.maximum(grad + rng.normal(0, 0.15, (H, W)), 0).astype("float32")
    xr.Dataset({"precipitation": (("latitude", "longitude"), truth)},
               coords={"latitude": lat, "longitude": lon}).to_netcdf(RFE / f"{td:%Y%m%d}.nc")

# constants
yy, xx = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
xr.Dataset({"elevation": (("latitude", "longitude"), (1500 * np.sin(3 * xx) * np.cos(3 * yy)).astype("float32"))},
           coords={"latitude": lat, "longitude": lon}).to_netcdf(CON / "elev.nc")
xr.Dataset({"lsm": (("latitude", "longitude"), (xx > 0.3).astype("float32"))},
           coords={"latitude": lat, "longitude": lon}).to_netcdf(CON / "lsm.nc")
print("[synth] truth + constants written")

# ── patch config, then run the REAL write_data ──
import read_config  # noqa: E402
read_config.get_data_paths = lambda: {
    "GENERAL": {"TRUTH_PATH": str(RFE.parent), "FORECAST_PATH": str(IFS.parent),
                "CONSTANTS_PATH": str(CON), "NORMALISATION_PATH": str(DATA), "LEAD_IDX": 1},
    "TFRecords": {"tfrecords_path": str(TFR)}}
read_config.read_downscaling_factor = lambda: {"downscaling_factor": 1, "steps": [1]}

import data  # noqa: E402
import tfrecords_generator as tg  # noqa: E402

print("\n[norm] gen_fcst_norm ...")
data.gen_fcst_norm(YEAR)
data.fcst_norm = data.load_fcst_norm(YEAR)
assert tg.DEFAULT_FCST_SHAPE == (128, 128, 28), tg.DEFAULT_FCST_SHAPE

print("[write] write_data(%d) ..." % YEAR)
tg.write_data(YEAR)

# ── report ──
import tensorflow as tf  # noqa: E402
files = sorted(glob.glob(str(TFR / "*.tfrecords")))
print("\nEmitted .tfrecords:")
total = 0
for f in files:
    n = sum(1 for _ in tf.data.TFRecordDataset(f, compression_type="GZIP"))
    total += n
    print(f"  {Path(f).name:24s} {Path(f).stat().st_size:>10,d} B   {n} records")
print(f"  TOTAL records: {total}")
print(f"\nLocal tfrecords ready at: {TFR}")
