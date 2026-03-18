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
#     "xarray",
#     "netcdf4",
#     "scikit-learn",
#     "cfgrib",
#     "dask",
#     "tqdm",
#     "properscoring",
#     "gribberish",
#     "kerchunk",
#     "zarr",
#     "pandas",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "google-cloud-storage",
#     "google-auth",
#     "distributed",
#     "requests",
#     "pyyaml",
#     "cftime",
#     "pyarrow",
# ]
# ///
"""
GEFS GIK-cGAN Unified Pipeline
==============================

This script provides an end-to-end workflow from GEFS data streaming via
Grib-Index-Kerchunk (GIK) to cGAN precipitation downscaling inference.

Pipeline Stages:
    1. Download GIK templates from Hugging Face (if not present)
    2. Create parquet references for target date
    3. Stream all required variables from AWS S3
    4. Convert to NetCDF format for cGAN
    5. Run cGAN inference to produce high-resolution precipitation

Usage:
    uv run run_gefs_gik_cgan_pipeline.py --date 20250918 --output_dir gik_cgan_output

Author: ICPAC GIK-cGAN Integration Team
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# GIK template source
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz"
DEFAULT_TEMPLATE_FILE = "gik-fmrc-gefs-20241112.tar.gz"

# cGAN model paths - use relative paths from script directory
# Model files should be extracted from cgan_compact_20260202.zip
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
DEFAULT_MODEL_FOLDER = os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/logfile_gefs_v3/")
DEFAULT_CONSTANTS_PATH = os.path.join(_SCRIPT_DIR, "cgan_compact_20260202/CONSTANTS/")
DEFAULT_CHECKPOINT = 345600

# Ensemble configuration
DEFAULT_ENSEMBLE_MEMBERS = 30
FORECAST_HOURS = (30, 54)  # Start and end hours for cGAN inference
CGAN_HOURS = 6  # cGAN time aggregation window


def compute_cgan_step_indices(start_hour=30, end_hour=54, hours=6, step_interval=3):
    """Compute GEFS step positions and forecast hours for cGAN inference.

    The cGAN training code (data/data_gefs.py) selects two timesteps per
    valid time window:
        step_1 = valid_time          (e.g. 30h)
        step_2 = valid_time + hours  (e.g. 36h)

    GEFS data has 81 3-hourly steps (hours 0, 3, 6, ..., 240).  The
    positional index in the step dimension is hour // step_interval.

    Returns
    -------
    step_positions : list[int]
        Positional indices for streaming (e.g. [10, 12, 14, 16, 18]).
    step_hours : list[int]
        Actual forecast hours for the NetCDF step coordinate
        (e.g. [30, 36, 42, 48, 54]).
    """
    n_valid = end_hour // hours - start_hour // hours
    valid_times = np.arange(start_hour, end_hour + 1, hours)
    needed_hours = set()
    for i in range(n_valid):
        vt = valid_times[i]
        needed_hours.add(vt)
        needed_hours.add(vt + hours)
    step_hours = sorted(needed_hours)
    step_positions = [h // step_interval for h in step_hours]
    return step_positions, step_hours


def compute_apcp_cumulative_steps(end_hour=54, step_interval=3):
    """Compute ALL GEFS step positions from hour 0 to end_hour for cumulative APCP.

    GEFS APCP uses 6-hour accumulation buckets (resets at hours 0, 6, 12, ...).
    To compute total accumulated precipitation from hour 0 to any target hour,
    we need all intermediate steps so we can sum bucket-boundary values.

    Returns
    -------
    step_positions : list[int]
        All positional indices from 0 to end_hour // step_interval.
    step_hours : list[int]
        All forecast hours from 0 to end_hour.
    """
    step_hours = list(range(0, end_hour + step_interval, step_interval))
    step_positions = [h // step_interval for h in step_hours]
    return step_positions, step_hours


def bucket_to_total_accumulated(data, step_hours, bucket_period=6):
    """Convert GEFS bucket-incremental APCP to total accumulated from hour 0.

    GEFS APCP accumulates within 6-hour buckets that reset at hours 0, 6, 12, ...
    At each step, the value represents precipitation accumulated since the
    last bucket boundary.  This function sums bucket contributions to produce
    the total accumulated precipitation from hour 0.

    Args:
        data: array of shape (member, n_steps, lat, lon)
        step_hours: list of forecast hours, one per step in the data
        bucket_period: bucket reset interval in hours (default 6)

    Returns:
        cumulative: array of same shape, values = total accumulated from hour 0
    """
    cumulative = np.zeros_like(data)

    # Build a mapping from hour to step index
    hour_to_idx = {h: i for i, h in enumerate(step_hours)}

    for i, hour in enumerate(step_hours):
        if hour == 0:
            cumulative[:, i, :, :] = 0
            continue

        # Determine the start of the current bucket
        current_bucket_start = (hour // bucket_period) * bucket_period

        total = np.zeros_like(data[:, 0, :, :])

        # Sum all bucket-boundary (end-of-bucket) values for complete buckets
        # Bucket boundaries: bucket_period, 2*bucket_period, ..., current_bucket_start
        for boundary in range(bucket_period, current_bucket_start + 1, bucket_period):
            if boundary in hour_to_idx:
                total = total + data[:, hour_to_idx[boundary], :, :]

        # If this hour is NOT a bucket boundary, add the partial (within-bucket)
        # accumulation.  If it IS a boundary, its value is already included above.
        if hour % bucket_period != 0:
            total = total + data[:, i, :, :]

        cumulative[:, i, :, :] = total

    return cumulative

# Region (ICPAC East Africa)
REGION = {
    'lat_min': -13.65, 'lat_max': 24.65,
    'lon_min': 19.15, 'lon_max': 54.25
}


# ==============================================================================
# STAGE 1: GIK Template Management
# ==============================================================================

def download_template(url: str, local_path: str) -> bool:
    """Download GIK template tar.gz from Hugging Face."""
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Template already exists: {local_path} ({size_mb:.1f} MB)")
        return True

    print(f"  Downloading from: {url}")
    print(f"  This may take several minutes...")

    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

        print()
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Downloaded: {local_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  Error downloading template: {e}")
        return False


# ==============================================================================
# STAGE 2: GIK Reference Creation
# ==============================================================================

def create_gik_references(
    template_path: str,
    target_date: str,
    target_run: str,
    output_dir: Path,
    ensemble_members: List[str]
) -> bool:
    """Create GIK parquet references for the target date."""
    print(f"  Target date: {target_date}")
    print(f"  Model run: {target_run}Z")
    print(f"  Members: {len(ensemble_members)}")

    # Import from local gefs_util.py (copied from grib-index-kerchunk)
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from gefs_util import (
            generate_axes,
            filter_build_grib_tree,
            build_gefs_deflated_store_from_template,
            calculate_time_dimensions,
            cs_create_mapped_index_local,
            prepare_zarr_store,
            process_unique_groups,
            LocalTarGzMappingManager
        )
    except ImportError as e:
        print(f"  Error importing GIK utilities: {e}")
        print(f"  Make sure gefs_util.py is in the same directory as this script")
        return False

    # Extended variables for cGAN
    forecast_dict = {
        "Surface pressure": "PRES:surface",
        "2 metre temperature": "TMP:2 m above ground",
        "10m U wind": "UGRD:10 m above ground",
        "10m V wind": "VGRD:10 m above ground",
        "Precipitable water": "PWAT:entire atmosphere (considered as a single layer)",
        "CAPE": "CAPE:surface",
        "Mean sea level pressure": "MSLET:mean sea level",
        "Total Precipitation": "APCP:surface",
    }

    # Initialize mapping manager
    print(f"  Loading template: {template_path}")
    mapping_manager = LocalTarGzMappingManager(template_path)
    available_members = mapping_manager.list_ensemble_members()
    print(f"  Available members in template: {len(available_members)}")

    # Load deflated store: try template-based method first (fast), fall back to scan_grib
    # The deflated-store template parquet may be:
    #   1. A standalone .parquet file alongside the tar.gz
    #   2. Embedded inside the tar.gz
    #   3. Not available → fall back to filter_build_grib_tree
    deflated_template = template_path.replace('.tar.gz', '').rstrip('.') + '/'
    # Look for standalone deflated-store parquet next to the tar.gz
    import glob as _glob
    candidates = (
        _glob.glob(os.path.join(os.path.dirname(template_path) or '.', 'gefs-deflated-store-template*.parquet'))
        + _glob.glob(os.path.join(os.path.dirname(os.path.abspath(template_path)), 'gefs-deflated-store-template*.parquet'))
    )
    if candidates:
        print(f"  Using standalone deflated-store template: {candidates[0]}")
        deflated_store = build_gefs_deflated_store_from_template(
            candidates[0], filter_vars=forecast_dict
        )
    else:
        try:
            deflated_store = build_gefs_deflated_store_from_template(
                template_path, filter_vars=forecast_dict
            )
        except FileNotFoundError:
            print("  Deflated-store template not found, falling back to scan_grib...")
            gefs_files = []
            sample_member = ensemble_members[0]
            for hour in [0, 3]:
                url = (f"s3://noaa-gefs-pds/gefs.{target_date}/{target_run}/atmos/pgrb2sp25/"
                       f"{sample_member}.t{target_run}z.pgrb2s.0p25.f{hour:03d}")
                gefs_files.append(url)
            _, deflated_store = filter_build_grib_tree(gefs_files, forecast_dict)

    # Generate time axes
    axes = generate_axes(target_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    successful = 0

    for member in ensemble_members:
        if member not in available_members:
            print(f"    {member}: skipped (not in template)")
            continue

        try:
            print(f"    Processing {member}...", end=' ', flush=True)

            # Create mapped index
            gefs_kind = cs_create_mapped_index_local(
                axes, target_date, member,
                tar_gz_path=template_path,
                mapping_manager=mapping_manager
            )

            # Prepare zarr store
            time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
            zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
            updated_zstore = process_unique_groups(
                zstore, chunk_index, time_dims, time_coords,
                times, valid_times, steps
            )

            # Save parquet file
            import json as json_module
            output_path = output_dir / f"{member}_{target_date}_{target_run}z.parquet"

            data = []
            for key, value in updated_zstore.items():
                if isinstance(value, str):
                    encoded = value.encode('utf-8')
                elif isinstance(value, (list, dict)):
                    encoded = json_module.dumps(value).encode('utf-8')
                else:
                    encoded = str(value).encode('utf-8')
                data.append((key, encoded))

            df = pd.DataFrame(data, columns=['key', 'value'])
            df.to_parquet(output_path)

            successful += 1
            print("OK")

        except Exception as e:
            print(f"FAILED ({str(e)[:40]})")

    mapping_manager.cleanup()

    print(f"  Created {successful}/{len(ensemble_members)} parquet files")
    return successful > 0


# ==============================================================================
# STAGE 3: Multi-Variable Data Streaming
# ==============================================================================

def stream_gefs_data(
    parquet_dir: Path,
    output_dir: Path,
    target_date: str,
    target_run: str,
    max_members: Optional[int] = None,
    step_filter: Optional[set] = None,
    step_hours: Optional[list] = None,
    cumulative_apcp: bool = False,
    apcp_step_filter: Optional[set] = None,
    apcp_step_hours: Optional[list] = None,
) -> bool:
    """Stream GEFS data using the parquet references.

    Args:
        step_filter: Optional set of step positions to stream (None = all).
        step_hours: Optional list of actual forecast hours matching step_filter.
        cumulative_apcp: If True, download extra APCP steps for cumulative calc.
        apcp_step_filter: Step positions for APCP (all steps 0 to end_hour).
        apcp_step_hours: Forecast hours for APCP steps.
    """
    # Import the streaming module
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    try:
        from stream_gefs_for_cgan import (
            create_cgan_zarr_store,
            stream_all_variables_for_member,
            read_parquet_refs,
            discover_variable_chunks,
            CGAN_VARIABLES
        )
    except ImportError:
        print("  Error: stream_gefs_for_cgan.py not found")
        print("  Make sure it's in the same directory as this script")
        return False

    # Find parquet files
    parquet_files = sorted(parquet_dir.glob(f"gep*_{target_date}_{target_run}z.parquet"))

    if not parquet_files:
        print(f"  No parquet files found in {parquet_dir}")
        return False

    if max_members:
        parquet_files = parquet_files[:max_members]

    n_members = len(parquet_files)
    print(f"  Found {n_members} parquet files")

    # Determine timesteps
    zstore = read_parquet_refs(str(parquet_files[0]))
    n_timesteps = 81  # Default

    for var_name, var_config in CGAN_VARIABLES.items():
        chunks = discover_variable_chunks(zstore, var_config['gefs_prefix'])
        if chunks:
            n_timesteps = len(chunks)
            print(f"  Total available timesteps: {n_timesteps} (from {var_name})")
            break

    # When step_filter is active, zarr only needs the filtered count
    if step_filter:
        step_indices = sorted(step_filter)
        n_zarr_steps = len(step_indices)
        print(f"  Filtering to {n_zarr_steps} steps: {step_indices}")
    else:
        n_zarr_steps = n_timesteps

    # Per-variable step count overrides for cumulative apcp
    n_timesteps_overrides = None
    step_filter_overrides = None
    if cumulative_apcp and apcp_step_filter:
        apcp_n_steps = len(sorted(apcp_step_filter))
        n_timesteps_overrides = {'apcp': apcp_n_steps}
        step_filter_overrides = {'apcp': apcp_step_filter}
        print(f"  Cumulative APCP: downloading {apcp_n_steps} steps (hours 0-{max(apcp_step_hours or [0])})")

    # Create zarr store
    zarr_output = output_dir / f"zarr_{target_date}_{target_run}z"
    zarr_store, _ = create_cgan_zarr_store(
        zarr_output, n_members, n_zarr_steps,
        n_timesteps_overrides=n_timesteps_overrides,
    )

    # Store step metadata for Stage 4 (plain ints for JSON serialization)
    if step_filter:
        zarr_store.attrs['step_indices'] = [int(s) for s in step_indices]
        # Store actual forecast hours for the NetCDF step coordinate
        if step_hours is not None:
            zarr_store.attrs['step_hours'] = [int(h) for h in step_hours]
        else:
            # Compute hours from positions (3-hourly GEFS data)
            zarr_store.attrs['step_hours'] = [int(s * 3) for s in step_indices]

    # Store apcp-specific metadata for cumulative calculation in Stage 4
    if cumulative_apcp and apcp_step_hours:
        zarr_store.attrs['cumulative_apcp'] = True
        zarr_store.attrs['apcp_step_hours'] = [int(h) for h in apcp_step_hours]

    # Stream all members
    for member_idx, pf in enumerate(parquet_files):
        stream_all_variables_for_member(
            str(pf), zarr_store, member_idx,
            step_filter=step_filter,
            step_filter_overrides=step_filter_overrides,
        )

    print(f"  Output: {zarr_output}")
    return True


# ==============================================================================
# STAGE 4: Zarr to NetCDF Conversion
# ==============================================================================

def convert_zarr_to_netcdf(
    zarr_dir: Path,
    output_dir: Path,
    target_date: str,
    target_run: str = "00",
) -> bool:
    """Convert zarr store to NetCDF format for cGAN.

    When the zarr store has 'cumulative_apcp' metadata, the APCP variable
    is converted from GEFS bucket-incremental to total-accumulated values
    before writing to NetCDF.  Only the cGAN-needed step hours are written.
    """
    import zarr

    print(f"  Reading zarr from: {zarr_dir}")

    store = zarr.open_group(str(zarr_dir), mode='r')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Variable mapping: zarr name -> cGAN NetCDF name
    var_mapping = {
        'cape': 'cape',
        'pres': 'pres',
        'pwat': 'pwat',
        'tmp': 'tmp',
        'ugrd': 'ugrd',
        'vgrd': 'vgrd',
        'msl': 'msl',
        'apcp': 'apcp',
    }

    # Also handle mslet -> msl mapping
    if 'mslet' in store.keys():
        var_mapping['mslet'] = 'msl'

    lats = store['latitude'][:]
    lons = store['longitude'][:]
    members = store['member'][:]
    n_timesteps = store.attrs.get('n_timesteps', 81)

    # Step coordinate: prefer actual forecast hours from metadata, fall back
    # to position indices, then sequential integers.
    step_hours_meta = store.attrs.get('step_hours', None)
    step_indices_meta = store.attrs.get('step_indices', None)
    if step_hours_meta is not None:
        step_coord = np.array(step_hours_meta, dtype=np.int64)
        print(f"  Using forecast hours as step coordinate: {list(step_coord)}")
    elif step_indices_meta is not None:
        # Legacy: convert position indices to hours (3-hourly GEFS data)
        step_coord = np.array(step_indices_meta, dtype=np.int64) * 3
        print(f"  Converting position indices to hours: {list(step_coord)}")
    else:
        # Full dataset: position indices × 3 = forecast hours
        step_coord = np.arange(n_timesteps, dtype=np.int64) * 3
        print(f"  Using full 3-hourly step hours: 0 to {(n_timesteps-1)*3}")

    # Check for cumulative APCP mode
    do_cumulative_apcp = store.attrs.get('cumulative_apcp', False)
    apcp_all_hours = store.attrs.get('apcp_step_hours', None)
    if do_cumulative_apcp:
        print(f"  Cumulative APCP mode: converting bucket-incremental → total accumulated")
        if apcp_all_hours:
            print(f"    APCP step hours: {list(apcp_all_hours[:5])} ... {list(apcp_all_hours[-3:])}")

    # Parse init time
    init_time = datetime.strptime(target_date, '%Y%m%d')

    saved_files = []

    for zarr_var, cgan_name in var_mapping.items():
        if zarr_var not in store.keys():
            print(f"    {zarr_var} -> {cgan_name}: not found, skipping")
            continue

        print(f"    {zarr_var} -> {cgan_name}...", end=' ', flush=True)

        data = store[zarr_var][:]  # (member, step, lat, lon)

        # Handle cumulative APCP conversion
        if zarr_var == 'apcp' and do_cumulative_apcp and apcp_all_hours:
            apcp_all_hours_list = [int(h) for h in apcp_all_hours]

            # Convert bucket-incremental to total accumulated
            data = bucket_to_total_accumulated(data, apcp_all_hours_list)
            print("(cumsum) ", end='', flush=True)

            # Extract only the cGAN-needed step hours
            needed_hours = set(int(h) for h in step_coord)
            keep_indices = [i for i, h in enumerate(apcp_all_hours_list) if h in needed_hours]
            data = data[:, keep_indices, :, :]

            # Verify shape matches step_coord
            if data.shape[1] != len(step_coord):
                print(f"WARNING: APCP shape mismatch after extraction "
                      f"({data.shape[1]} vs {len(step_coord)} expected)")

        data_with_time = np.expand_dims(data, axis=0)  # (1, member, step, lat, lon)

        ds_out = xr.Dataset(
            {
                cgan_name: xr.DataArray(
                    data=data_with_time.astype(np.float32),
                    dims=['time', 'member', 'step', 'latitude', 'longitude'],
                    attrs={'long_name': cgan_name}
                )
            },
            coords={
                'time': [np.datetime64(init_time)],
                'member': members,
                'step': step_coord,
                'latitude': lats,
                'longitude': lons,
            }
        )

        output_file = output_dir / f"{cgan_name}_{target_date}_{target_run}z.nc"
        ds_out.to_netcdf(output_file)
        saved_files.append(output_file)
        print("OK")

    print(f"  Created {len(saved_files)} NetCDF files in {output_dir}")
    return len(saved_files) > 0


# ==============================================================================
# STAGE 5: cGAN Inference
# ==============================================================================

def run_cgan_inference(
    netcdf_dir: Path,
    output_dir: Path,
    target_date: str,
    model_folder: str,
    constants_path: str,
    checkpoint: int,
    start_hour: int = 30,
    end_hour: int = 54,
    ensemble_members: int = 50,
    target_run: str = "00",
) -> bool:
    """Run cGAN inference on the prepared NetCDF data."""
    # Check for TensorFlow
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  GPUs available: {len(gpus)}")
    except ImportError:
        print("  Error: TensorFlow not available")
        print("  Install with: pip install tensorflow")
        return False

    # Import the inference script components
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    # Create a temporary config for the inference script
    year = target_date[:4]
    date_formatted = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"

    config = {
        "model_folder": model_folder,
        "checkpoint": checkpoint,
        "input_folder": str(netcdf_dir.parent),  # Parent because inference looks for {input_folder}/{date}_{run}z/
        "constants_path": constants_path,
        "output_folder": str(output_dir),
        "dates": [date_formatted],
        "start_hour": start_hour,
        "end_hour": end_hour,
        "ensemble_members": ensemble_members,
        "normalization_mode": "gefs",
        "gefs_norm_file": os.path.join(constants_path, "FCSTNorm_GEFS_2018.pkl"),
        "run": target_run,
    }

    # Check if model files exist
    weights_file = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07d}.h5")
    if not os.path.exists(weights_file):
        print(f"  Warning: Model weights not found at {weights_file}")
        print(f"  Please ensure the cGAN model is available")
        return False

    # Run inference
    print(f"  Model: {model_folder}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Date: {date_formatted}")
    print(f"  Hours: {start_hour} to {end_hour}")

    try:
        # Import inference function from the existing script
        from run_gefs_inference_raw import run_inference, CONFIG as inference_config

        # Update config
        for key, value in config.items():
            inference_config[key] = value

        run_inference()
        return True

    except ImportError:
        print("  Could not import run_gefs_inference_raw.py")
        print("  Running as subprocess instead...")

        # Alternative: run as subprocess
        cmd = [
            sys.executable,
            str(script_dir / "run_gefs_inference_raw.py")
        ]

        # Note: This would require modifying run_gefs_inference_raw.py to accept CLI args
        # For now, return False and inform user
        print("  Please run inference manually with updated CONFIG in run_gefs_inference_raw.py")
        return False


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GEFS GIK to cGAN Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for a specific date
    python run_gefs_gik_cgan_pipeline.py --date 20250918

    # Only create GIK references
    python run_gefs_gik_cgan_pipeline.py --date 20250918 --stages 1,2

    # Only run inference (if data already prepared)
    python run_gefs_gik_cgan_pipeline.py --date 20250918 --stages 5 --netcdf_dir ./cgan_input/2025
        """
    )

    parser.add_argument('--date', type=str, required=True,
                       help='Target forecast date YYYYMMDD')
    parser.add_argument('--run', type=str, default='00',
                       help='Model run hour (default: 00)')
    parser.add_argument('--output_dir', type=str, default='gik_cgan_pipeline_output',
                       help='Base output directory')
    parser.add_argument('--stages', type=str, default='1,2,3,4,5',
                       help='Pipeline stages to run (comma-separated: 1,2,3,4,5)')

    # GIK options
    parser.add_argument('--template', type=str, default=DEFAULT_TEMPLATE_FILE,
                       help='GIK template tar.gz file')
    parser.add_argument('--max_members', type=int, default=DEFAULT_ENSEMBLE_MEMBERS,
                       help=f'Max ensemble members (default: {DEFAULT_ENSEMBLE_MEMBERS})')

    # cGAN options
    parser.add_argument('--model_folder', type=str, default=DEFAULT_MODEL_FOLDER,
                       help='cGAN model folder')
    parser.add_argument('--constants_path', type=str, default=DEFAULT_CONSTANTS_PATH,
                       help='cGAN constants folder')
    parser.add_argument('--checkpoint', type=int, default=DEFAULT_CHECKPOINT,
                       help=f'cGAN checkpoint number (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--netcdf_dir', type=str, default=None,
                       help='Pre-existing NetCDF directory (for stage 5 only)')
    parser.add_argument('--cgan_steps_only', action='store_true',
                       help='Only stream the timesteps needed for cGAN inference (5 of 81, ~93%% faster)')
    parser.add_argument('--cumulative_apcp', action='store_true',
                       help='Convert bucket-incremental APCP to total accumulated from hour 0. '
                            'Downloads all APCP steps (0 to end_hour) and computes cumulative sum '
                            'during zarr-to-NetCDF conversion. Implies --cgan_steps_only for non-APCP variables.')

    args = parser.parse_args()

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(',')]

    target_date = args.date
    target_run = args.run
    output_base = Path(args.output_dir)

    # Ensemble members
    ensemble_members = [f'gep{i:02d}' for i in range(1, args.max_members + 1)]

    # Compute step filter if requested
    step_filter = None
    step_hours_meta = None
    cumulative_apcp = args.cumulative_apcp
    apcp_step_filter = None
    apcp_step_hours_meta = None

    if args.cgan_steps_only or cumulative_apcp:
        step_positions, step_hours_meta = compute_cgan_step_indices(
            FORECAST_HOURS[0], FORECAST_HOURS[1], CGAN_HOURS
        )
        step_filter = set(step_positions)

    if cumulative_apcp:
        # For cumulative APCP, download ALL steps from hour 0 to end_hour
        max_cgan_hour = max(step_hours_meta) if step_hours_meta else FORECAST_HOURS[1]
        apcp_positions, apcp_step_hours_meta = compute_apcp_cumulative_steps(
            end_hour=max_cgan_hour
        )
        apcp_step_filter = set(apcp_positions)

    print("="*70)
    print("GEFS GIK → cGAN Unified Pipeline")
    print("="*70)
    print(f"Target Date: {target_date}")
    print(f"Model Run: {target_run}Z")
    print(f"Output Base: {output_base}")
    print(f"Stages: {stages}")
    print(f"Ensemble Members: {args.max_members}")
    if step_filter:
        print(f"Step Filter: positions {sorted(step_filter)} → hours {step_hours_meta} ({len(step_filter)} of 81 steps)")
    if cumulative_apcp:
        print(f"Cumulative APCP: downloading {len(apcp_step_filter)} steps (hours 0-{max_cgan_hour})")
    print("="*70)

    pipeline_start = time.time()
    output_base.mkdir(parents=True, exist_ok=True)

    # Directory structure
    parquet_dir = output_base / "parquet_refs"
    zarr_dir = output_base / f"zarr_{target_date}_{target_run}z"
    netcdf_dir = Path(args.netcdf_dir) if args.netcdf_dir else output_base / "netcdf" / f"{target_date}_{target_run}z"
    cgan_output = output_base / "cgan_output"

    # STAGE 1: Download template
    if 1 in stages:
        print("\n" + "="*70)
        print("STAGE 1: Download GIK Template")
        print("="*70)
        if not download_template(TEMPLATE_URL, args.template):
            print("Stage 1 failed. Cannot continue.")
            return 1

    # STAGE 2: Create GIK references
    if 2 in stages:
        print("\n" + "="*70)
        print("STAGE 2: Create GIK Parquet References")
        print("="*70)
        if not create_gik_references(
            args.template, target_date, target_run,
            parquet_dir, ensemble_members
        ):
            print("Stage 2 failed.")
            if 3 in stages or 4 in stages:
                print("Cannot continue to streaming stages.")
                return 1

    # STAGE 3: Stream GEFS data
    if 3 in stages:
        print("\n" + "="*70)
        print("STAGE 3: Stream GEFS Multi-Variable Data")
        print("="*70)
        if not stream_gefs_data(
            parquet_dir, output_base, target_date, target_run, args.max_members,
            step_filter=step_filter, step_hours=step_hours_meta,
            cumulative_apcp=cumulative_apcp,
            apcp_step_filter=apcp_step_filter,
            apcp_step_hours=apcp_step_hours_meta,
        ):
            print("Stage 3 failed.")
            if 4 in stages:
                print("Cannot continue to NetCDF conversion.")
                return 1

    # STAGE 4: Convert to NetCDF
    if 4 in stages:
        print("\n" + "="*70)
        print("STAGE 4: Convert Zarr to NetCDF")
        print("="*70)
        zarr_path = output_base / f"zarr_{target_date}_{target_run}z"
        if zarr_path.exists():
            if not convert_zarr_to_netcdf(zarr_path, netcdf_dir, target_date, target_run):
                print("Stage 4 failed.")
                if 5 in stages:
                    print("Cannot continue to cGAN inference.")
                    return 1
        else:
            print(f"  Zarr directory not found: {zarr_path}")
            print("  Run stage 3 first or provide existing zarr data.")
            if 5 in stages:
                return 1

    # STAGE 5: cGAN Inference
    if 5 in stages:
        print("\n" + "="*70)
        print("STAGE 5: cGAN Inference")
        print("="*70)
        if not run_cgan_inference(
            netcdf_dir, cgan_output, target_date,
            args.model_folder, args.constants_path, args.checkpoint,
            target_run=target_run,
        ):
            print("Stage 5 failed or requires manual execution.")
            print(f"\nTo run inference manually:")
            print(f"  1. Update CONFIG in run_gefs_inference_raw.py:")
            print(f"     'input_folder': '{netcdf_dir.parent}'")
            print(f"     'dates': ['{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}']")
            print(f"  2. Run: python run_gefs_inference_raw.py")

    # Summary
    total_time = time.time() - pipeline_start

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nOutput Locations:")
    print(f"  Parquet References: {parquet_dir}")
    print(f"  Zarr Store: {zarr_dir}")
    print(f"  NetCDF Files: {netcdf_dir}")
    print(f"  cGAN Output: {cgan_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
