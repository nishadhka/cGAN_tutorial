# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "tensorflow==2.15.0",
#     "h5py",
#     "pyyaml",
#     "numpy",
#     "xarray",
#     "netcdf4",
# ]
# ///
"""Quick test: compare cGAN output with bucket-incremental vs cumulative APCP.

Reads from the pipeline output NetCDF files and runs a few ensemble members.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import xarray as xr
import h5py
import tensorflow as tf
import pickle
import yaml
import gc

tf.get_logger().setLevel('ERROR')

# Import model components from inference script
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
with h5py.File(os.path.join(constants_path, "lsm.nc"), 'r') as f:
    lsm = f['lsm'][:]
network_const_input = np.stack([z, lsm], axis=-1)[np.newaxis, ...]

# Build model
with open(os.path.join(model_folder, "setup_params.yaml"), "r") as f:
    setup_params = yaml.safe_load(f)

gen = generator_fn(
    mode="GAN", arch=setup_params["MODEL"]["architecture"],
    downscaling_steps=[1],
    input_channels=4*len(all_fcst_fields), constant_fields=2,
    filters_gen=setup_params["GENERATOR"]["filters_gen"],
    noise_channels=setup_params["GENERATOR"]["noise_channels"],
    latent_variables=setup_params["GENERATOR"]["latent_variables"],
    padding=setup_params["MODEL"]["padding"]
)
gen.load_weights(os.path.join(model_folder, "models", "gen_weights-0345600.h5"))
print("Model loaded\n")

noise_channels = setup_params["GENERATOR"]["noise_channels"]
noise_shape = (384, 352, noise_channels)
N_MEMBERS = 5
valid_time_hour = 30

# --- Helper to load fields from NetCDF and prepare model input ---
def load_fields_from_netcdf(input_folder, flip_lat=True):
    """Load all fields from pipeline NetCDF output, normalize, return model input."""
    field_arrays = []
    for field in all_fcst_fields:
        nc_path = os.path.join(input_folder, f"{field}_20240520_00z.nc")
        with h5py.File(nc_path, 'r') as f:
            coords = {'time', 'member', 'step', 'latitude', 'longitude'}
            short_name = [k for k in f.keys() if k not in coords][0]
            all_data = f[short_name][0]  # (member, step, lat, lon)
            step_vals = f['step'][:]
            # Find step indices for our hours
            idx1 = np.where(step_vals == valid_time_hour)[0]
            idx2 = np.where(step_vals == valid_time_hour + HOURS)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                h1_ns = valid_time_hour * 3600 * 1e9
                h2_ns = (valid_time_hour + HOURS) * 3600 * 1e9
                idx1 = np.where(np.abs(step_vals - h1_ns) < 1e6)[0]
                idx2 = np.where(np.abs(step_vals - h2_ns) < 1e6)[0]
            data = all_data[:, [idx1[0], idx2[0]], :, :]
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
            # Print raw apcp stats before normalization
            step1_mean = np.nanmean(data[:, :, :, 0])
            step2_mean = np.nanmean(data[:, :, :, 1])
            step1_max = np.nanmax(data[:, :, :, 0])
            step2_max = np.nanmax(data[:, :, :, 1])
            print(f"    apcp raw: step1 mean={step1_mean:.2f} max={step1_max:.2f}, "
                  f"step2 mean={step2_mean:.2f} max={step2_max:.2f}")
            data = np.log10(1 + data)
            log_mean = np.nanmean(data[:, :, :, 0])
            log_max = np.nanmax(data[:, :, :, 0])
            print(f"    apcp log10: step1 mean={log_mean:.4f} max={log_max:.4f}")
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


def run_model(fcst, label):
    """Run model on input and report results."""
    results = []
    for i in range(N_MEMBERS):
        np.random.seed(42 + i)
        ng = NoiseGenerator(noise_shape, batch_size=1)
        pred = gen.predict([fcst, network_const_input, ng()], verbose=0)
        raw = pred[0, :, :, 0]
        r = denormalise(raw)
        results.append(r)
        if i == 0:
            print(f"    Raw model output (member 1): min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}")
    em = np.array(results).mean(axis=0)
    max_loc = np.unravel_index(em.argmax(), em.shape)
    print(f"    {N_MEMBERS}-member ens mean: max={em.max():.2f}mm mean={em.mean():.4f}mm")
    print(f"    Max location: row={max_loc[0]} col={max_loc[1]} (lat≈{-13.65 + max_loc[0]*0.1:.2f}°)")
    gc.collect()
    return em


# Test 1: Current cumulative APCP data (from pipeline with --cumulative_apcp)
print("=" * 70)
print("TEST 1: CUMULATIVE APCP (total accumulated from hour 0)")
print("=" * 70)
cumulative_folder = "gik_cgan_output/netcdf/20240520_00z"
print(f"  Loading from: {cumulative_folder}")
fcst_cumulative = load_fields_from_netcdf(cumulative_folder)
print(f"  Full input range: [{fcst_cumulative.min():.3f}, {fcst_cumulative.max():.3f}]")
em_cumulative = run_model(fcst_cumulative, "cumulative")

# Test 2: Previous bucket-incremental data
# Check if old data exists (from previous pipeline run before cumulative)
old_folder = "gik_cgan_output/netcdf/20240520_00z"
print(f"\n{'=' * 70}")
print("TEST 2: Original GEFS data for comparison (same folder, different reference)")
print("=" * 70)
print(f"  (Using same NetCDF files — now contain cumulative APCP)")
print(f"  For bucket-incremental comparison, see test_lat_flip.py results:")
print(f"  Previous bucket result: max=4.25mm (raw max=1.0134 in log-space)")
print(f"  Reference image target: max≈60mm (raw needs 1.7853 in log-space)")
print(f"\n  CUMULATIVE result:     max={em_cumulative.max():.2f}mm")
improvement = em_cumulative.max() / 4.25
print(f"  Improvement factor:    {improvement:.1f}x vs bucket-incremental")
if em_cumulative.max() > 20:
    print("  ✓ Significant improvement — cumulative hypothesis likely CORRECT")
elif em_cumulative.max() > 8:
    print("  ~ Moderate improvement — cumulative helps but may not be the full story")
else:
    print("  ✗ Minimal improvement — cumulative alone doesn't explain the gap")
