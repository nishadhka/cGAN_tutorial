# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorflow==2.15.0",
#     "h5py",
#     "pyyaml",
#     "numpy",
# ]
# ///
"""Quick test: compare cGAN output with and without latitude flip.

Uses h5py directly to read NetCDF/HDF5 files (avoids netCDF4 library issues).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import h5py
import tensorflow as tf
import pickle
import yaml
import sys

tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from inference script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "inference",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_gefs_inference_raw.py")
)
inference_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_mod)

NoiseGenerator = inference_mod.NoiseGenerator
generator_fn = inference_mod.generator
denormalise = inference_mod.denormalise
all_fcst_fields = inference_mod.all_fcst_fields
nonnegative_fields = inference_mod.nonnegative_fields
HOURS = inference_mod.HOURS

constants_path = "cgan_compact_20260202/CONSTANTS"
model_folder = "cgan_compact_20260202/logfile_gefs_v3"

# Load normalization stats
with open(os.path.join(constants_path, "FCSTNorm_GEFS_2018.pkl"), "rb") as f:
    fcst_norm = pickle.load(f)

# Load constants via h5py
with h5py.File(os.path.join(constants_path, "elev.nc"), 'r') as f:
    z = f['elevation'][:] / 10000.0
    elev_lat = f['lat'][:]
with h5py.File(os.path.join(constants_path, "lsm.nc"), 'r') as f:
    lsm = f['lsm'][:]
network_const_input = np.stack([z, lsm], axis=-1)[np.newaxis, ...]
print(f"Constants: shape={network_const_input.shape}")
print(f"  Elev lat: {elev_lat[0]:.2f} to {elev_lat[-1]:.2f} ({'ascending S->N' if elev_lat[0] < elev_lat[-1] else 'descending N->S'})")

# Build model
with open(os.path.join(model_folder, "setup_params.yaml"), "r") as f:
    setup_params = yaml.safe_load(f)

arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]

gen = generator_fn(
    mode="GAN", arch=arch, downscaling_steps=[1],
    input_channels=4*len(all_fcst_fields), constant_fields=2,
    filters_gen=filters_gen, noise_channels=noise_channels,
    latent_variables=latent_variables, padding=padding
)
gen.load_weights(os.path.join(model_folder, "models", "gen_weights-0345600.h5"))
print("Model loaded\n")

input_folder = "gik_cgan_output/netcdf/20240520_00z"
valid_time_hour = 30


def load_field_h5(field, flip_lat):
    """Load one field from NetCDF via h5py, select time=0 and two step slices."""
    nc_path = os.path.join(input_folder, f"{field}_20240520_00z.nc")
    with h5py.File(nc_path, 'r') as f:
        # Find the data variable (not a coordinate)
        coords = {'time', 'member', 'step', 'latitude', 'longitude'}
        short_name = [k for k in f.keys() if k not in coords][0]
        # Full shape: (time, member, step, lat, lon)
        # Select time=0
        all_data = f[short_name][0]  # (member, step, lat, lon)

        # Get step values and find indices for our hours
        step_vals = f['step'][:]
        idx1 = np.where(step_vals == valid_time_hour)[0]
        idx2 = np.where(step_vals == valid_time_hour + HOURS)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            # Try timedelta interpretation (ns)
            h1_ns = valid_time_hour * 3600 * 1e9
            h2_ns = (valid_time_hour + HOURS) * 3600 * 1e9
            idx1 = np.where(np.abs(step_vals - h1_ns) < 1e6)[0]
            idx2 = np.where(np.abs(step_vals - h2_ns) < 1e6)[0]
        data = all_data[:, [idx1[0], idx2[0]], :, :]  # (member, 2, lat, lon)

        lat_vals = f['latitude'][:]

    if flip_lat and lat_vals[0] > lat_vals[-1]:
        data = data[:, :, ::-1, :]

    return data


def load_fields(flip_lat):
    field_arrays = []
    for field in all_fcst_fields:
        data = load_field_h5(field, flip_lat)  # (member, 2, lat, lon)

        n_members, n_steps = data.shape[0], data.shape[1]
        data_flat = data.reshape(n_members * n_steps, data.shape[2], data.shape[3])
        data_flat = np.expand_dims(data_flat, axis=-1)
        data_resized = tf.image.resize(data_flat, [384, 352]).numpy()
        data_resized = data_resized[:, :, :, 0].reshape(n_members, n_steps, 384, 352)
        data = np.transpose(data_resized, (2, 3, 0, 1))  # (384, 352, member, step)

        if field in nonnegative_fields:
            data = np.maximum(data, 0.0)

        if field == "apcp":
            data = np.log10(1 + data)
        elif field in ["msl", "pres", "tmp"]:
            data -= fcst_norm[field]["mean"]
            data /= fcst_norm[field]["std"]
        elif field in nonnegative_fields:
            data /= fcst_norm[field]["max"]
        elif field in ["ugrd", "vgrd"]:
            data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])

        data_mean = np.nanmean(data, axis=-2)
        data_std = np.nanstd(data, axis=-2)
        data = np.concatenate([data_mean[..., [0]], data_std[..., [0]],
                               data_mean[..., [1]], data_std[..., [1]]], axis=-1)
        field_arrays.append(data)
    return np.concatenate(field_arrays, axis=-1)[np.newaxis, ...]


noise_shape = (384, 352, noise_channels)
N_MEMBERS = 5

# Test 1: WITH flip (current code)
print("=== Test 1: WITH flip (ascending, match constants) - 30 members ===")
fcst = load_fields(flip_lat=True)
print(f"  Input range: [{fcst.min():.3f}, {fcst.max():.3f}]")
results = []
for i in range(N_MEMBERS):
    np.random.seed(42 + i)
    ng = NoiseGenerator(noise_shape, batch_size=1)
    pred = gen.predict([fcst, network_const_input, ng()], verbose=0)
    r = denormalise(pred[0, :, :, 0])
    results.append(r)
em = np.array(results).mean(axis=0)
print(f"  5-member ens mean: max={em.max():.2f}, mean={em.mean():.4f}")

# Test 2: WITHOUT flip
print("\n=== Test 2: WITHOUT flip (descending, original GEFS) - 30 members ===")
fcst = load_fields(flip_lat=False)
results = []
for i in range(N_MEMBERS):
    np.random.seed(42 + i)
    ng = NoiseGenerator(noise_shape, batch_size=1)
    pred = gen.predict([fcst, network_const_input, ng()], verbose=0)
    r = denormalise(pred[0, :, :, 0])
    results.append(r)
em = np.array(results).mean(axis=0)
print(f"  5-member ens mean: max={em.max():.2f}, mean={em.mean():.4f}")

# Test 3: ZEROS input (baseline check)
print("\n=== Test 3: ZEROS input (baseline - model should produce near-zero) ===")
fcst_zeros = np.zeros_like(fcst)
results = []
for i in range(N_MEMBERS):
    np.random.seed(42 + i)
    ng = NoiseGenerator(noise_shape, batch_size=1)
    pred = gen.predict([fcst_zeros, network_const_input, ng()], verbose=0)
    r = denormalise(pred[0, :, :, 0])
    results.append(r)
em = np.array(results).mean(axis=0)
print(f"  5-member ens mean: max={em.max():.2f}, mean={em.mean():.4f}")

# Test 4: Use only 10 members for ensemble stats (training used ensemble_size=10)
print("\n=== Test 4: WITH flip, 10 members for ens stats (matching training) ===")


def load_fields_nmembers(flip_lat, n_members_for_stats):
    """Load fields using only first N members for ensemble statistics."""
    field_arrays = []
    for field in all_fcst_fields:
        nc_path = os.path.join(input_folder, f"{field}_20240520_00z.nc")
        with h5py.File(nc_path, 'r') as f:
            coords = {'time', 'member', 'step', 'latitude', 'longitude'}
            short_name = [k for k in f.keys() if k not in coords][0]
            all_data = f[short_name][0]  # (member, step, lat, lon)
            step_vals = f['step'][:]
            idx1 = np.where(step_vals == valid_time_hour)[0]
            idx2 = np.where(step_vals == valid_time_hour + HOURS)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                h1_ns = valid_time_hour * 3600 * 1e9
                h2_ns = (valid_time_hour + HOURS) * 3600 * 1e9
                idx1 = np.where(np.abs(step_vals - h1_ns) < 1e6)[0]
                idx2 = np.where(np.abs(step_vals - h2_ns) < 1e6)[0]
            data = all_data[:n_members_for_stats, [idx1[0], idx2[0]], :, :]
            lat_vals = f['latitude'][:]

        if flip_lat and lat_vals[0] > lat_vals[-1]:
            data = data[:, :, ::-1, :]

        n_m, n_s = data.shape[0], data.shape[1]
        data_flat = data.reshape(n_m * n_s, data.shape[2], data.shape[3])
        data_flat = np.expand_dims(data_flat, axis=-1)
        data_resized = tf.image.resize(data_flat, [384, 352]).numpy()
        data_resized = data_resized[:, :, :, 0].reshape(n_m, n_s, 384, 352)
        data = np.transpose(data_resized, (2, 3, 0, 1))

        if field in nonnegative_fields:
            data = np.maximum(data, 0.0)

        if field == "apcp":
            data = np.log10(1 + data)
        elif field in ["msl", "pres", "tmp"]:
            data -= fcst_norm[field]["mean"]
            data /= fcst_norm[field]["std"]
        elif field in nonnegative_fields:
            data /= fcst_norm[field]["max"]
        elif field in ["ugrd", "vgrd"]:
            data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])

        data_mean = np.nanmean(data, axis=-2)
        data_std = np.nanstd(data, axis=-2)
        data = np.concatenate([data_mean[..., [0]], data_std[..., [0]],
                               data_mean[..., [1]], data_std[..., [1]]], axis=-1)
        field_arrays.append(data)
    return np.concatenate(field_arrays, axis=-1)[np.newaxis, ...]


fcst_10 = load_fields_nmembers(flip_lat=True, n_members_for_stats=10)
print(f"  Input range: [{fcst_10.min():.3f}, {fcst_10.max():.3f}]")
results = []
for i in range(N_MEMBERS):
    np.random.seed(42 + i)
    ng = NoiseGenerator(noise_shape, batch_size=1)
    pred = gen.predict([fcst_10, network_const_input, ng()], verbose=0)
    r = denormalise(pred[0, :, :, 0])
    results.append(r)
em = np.array(results).mean(axis=0)
print(f"  5-member ens mean: max={em.max():.2f}, mean={em.mean():.4f}")

# Test 5: Check raw model output (before denormalization) to understand log-space values
print("\n=== Test 5: Raw model output (before denormalization) ===")
fcst_with = load_fields(flip_lat=True)
np.random.seed(42)
ng = NoiseGenerator(noise_shape, batch_size=1)
pred = gen.predict([fcst_with, network_const_input, ng()], verbose=0)
raw = pred[0, :, :, 0]
print(f"  Raw output: min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")
print(f"  After denorm: min={denormalise(raw).min():.4f}, max={denormalise(raw).max():.4f}")
print(f"  For 60mm output, raw would need to be: {np.log10(61):.4f}")
