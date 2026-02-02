#!/usr/bin/env python3
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
    python run_gefs_gik_cgan_pipeline.py --date 20250918 --output_dir gik_cgan_output

Prerequisites:
    # GIK dependencies
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs gribberish requests

    # cGAN dependencies
    pip install tensorflow numpy<2.0 netCDF4 pyyaml

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

            # Build GRIB tree structure (only need first 2 files)
            gefs_files = []
            for hour in [0, 3]:
                url = (f"s3://noaa-gefs-pds/gefs.{target_date}/{target_run}/atmos/pgrb2sp25/"
                       f"{member}.t{target_run}z.pgrb2s.0p25.f{hour:03d}")
                gefs_files.append(url)

            _, deflated_store = filter_build_grib_tree(gefs_files, forecast_dict)

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
    max_members: Optional[int] = None
) -> bool:
    """Stream GEFS data using the parquet references."""
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
            print(f"  Timesteps: {n_timesteps} (from {var_name})")
            break

    # Create zarr store
    zarr_output = output_dir / f"zarr_{target_date}_{target_run}z"
    zarr_store, _ = create_cgan_zarr_store(zarr_output, n_members, n_timesteps)

    # Stream all members
    for member_idx, pf in enumerate(parquet_files):
        stream_all_variables_for_member(str(pf), zarr_store, member_idx)

    print(f"  Output: {zarr_output}")
    return True


# ==============================================================================
# STAGE 4: Zarr to NetCDF Conversion
# ==============================================================================

def convert_zarr_to_netcdf(
    zarr_dir: Path,
    output_dir: Path,
    target_date: str
) -> bool:
    """Convert zarr store to NetCDF format for cGAN."""
    import zarr

    print(f"  Reading zarr from: {zarr_dir}")

    store = zarr.open_group(str(zarr_dir), mode='r')

    year = target_date[:4]
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

    # Create step coordinate (3-hour intervals)
    step_hours = np.arange(0, n_timesteps * 3, 3)
    step_ns = step_hours * 3.6e12  # Convert to nanoseconds

    # Parse init time
    init_time = datetime.strptime(target_date, '%Y%m%d')

    saved_files = []

    for zarr_var, cgan_name in var_mapping.items():
        if zarr_var not in store.keys():
            print(f"    {zarr_var} -> {cgan_name}: not found, skipping")
            continue

        print(f"    {zarr_var} -> {cgan_name}...", end=' ', flush=True)

        data = store[zarr_var][:]  # (member, step, lat, lon)
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
                'step': step_ns,
                'latitude': lats,
                'longitude': lons,
            }
        )

        output_file = output_dir / f"{cgan_name}_{year}.nc"
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
    ensemble_members: int = 50
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
        "input_folder": str(netcdf_dir.parent),  # Parent because inference looks for {input_folder}/{year}/
        "constants_path": constants_path,
        "output_folder": str(output_dir),
        "dates": [date_formatted],
        "start_hour": start_hour,
        "end_hour": end_hour,
        "ensemble_members": ensemble_members,
        "normalization_mode": "gefs",
        "gefs_norm_file": os.path.join(constants_path, "FCSTNorm_GEFS_2018.pkl"),
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

    args = parser.parse_args()

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(',')]

    target_date = args.date
    target_run = args.run
    output_base = Path(args.output_dir)

    # Ensemble members
    ensemble_members = [f'gep{i:02d}' for i in range(1, args.max_members + 1)]

    print("="*70)
    print("GEFS GIK → cGAN Unified Pipeline")
    print("="*70)
    print(f"Target Date: {target_date}")
    print(f"Model Run: {target_run}Z")
    print(f"Output Base: {output_base}")
    print(f"Stages: {stages}")
    print(f"Ensemble Members: {args.max_members}")
    print("="*70)

    pipeline_start = time.time()
    output_base.mkdir(parents=True, exist_ok=True)

    # Directory structure
    parquet_dir = output_base / "parquet_refs"
    zarr_dir = output_base / f"zarr_{target_date}_{target_run}z"
    netcdf_dir = Path(args.netcdf_dir) if args.netcdf_dir else output_base / "netcdf" / target_date[:4]
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
            parquet_dir, output_base, target_date, target_run, args.max_members
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
            if not convert_zarr_to_netcdf(zarr_path, netcdf_dir, target_date):
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
            args.model_folder, args.constants_path, args.checkpoint
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
