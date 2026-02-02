#!/usr/bin/env python3
"""
GEFS Multi-Variable Data Streaming for cGAN Inference
======================================================

This script streams multiple GEFS variables from AWS S3 using GIK (Grib-Index-Kerchunk)
parquet reference files and prepares the data for cGAN precipitation downscaling inference.

It extends the single-variable streaming from run_gefs_data_streaming.py to handle
all 8 variables required by the cGAN model:
    - cape: Convective Available Potential Energy
    - pres: Surface Pressure
    - pwat: Precipitable Water
    - tmp: 2-meter Temperature
    - ugrd: 10-meter U Wind
    - vgrd: 10-meter V Wind
    - msl: Mean Sea Level Pressure
    - apcp: Total Precipitation

Usage:
    python stream_gefs_for_cgan.py --parquet_dir output_parquet --output_dir cgan_zarr --date 20250918

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs gribberish

Author: ICPAC GIK-cGAN Integration Team
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import sys
import warnings
import time
import shutil
import gc
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import fsspec
import zarr
from zarr.codecs import BloscCodec

# Try to import gribberish for fast decoding
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib only")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# cGAN required variables mapping: cgan_name -> GEFS path prefix in zarr
# Note: GEFS uses different variable names than cGAN expects:
#   sp (surface pressure) -> pres
#   t2m (2m temperature) -> tmp
#   u10 (10m u-wind) -> ugrd
#   v10 (10m v-wind) -> vgrd
CGAN_VARIABLES = {
    'cape': {'gefs_prefix': 'cape/instant/surface/cape', 'grib_filter': 'CAPE:surface'},
    'pres': {'gefs_prefix': 'sp/instant/surface/sp', 'grib_filter': 'PRES:surface'},
    'pwat': {'gefs_prefix': 'pwat/instant/atmosphereSingleLayer/pwat', 'grib_filter': 'PWAT:entire atmosphere'},
    'tmp': {'gefs_prefix': 't2m/instant/heightAboveGround/t2m', 'grib_filter': 'TMP:2 m above ground'},
    'ugrd': {'gefs_prefix': 'u10/instant/heightAboveGround/u10', 'grib_filter': 'UGRD:10 m above ground'},
    'vgrd': {'gefs_prefix': 'v10/instant/heightAboveGround/v10', 'grib_filter': 'VGRD:10 m above ground'},
    'msl': {'gefs_prefix': 'mslet/instant/meanSea/mslet', 'grib_filter': 'MSLET:mean sea level'},
    'apcp': {'gefs_prefix': 'tp/accum/surface/tp', 'grib_filter': 'APCP:surface'},
}

# GEFS grid specification (0.25 degree global)
GEFS_GRID_SHAPE = (721, 1440)  # lat x lon
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# ICPAC/cGAN region coordinates (matching run_gefs_inference_raw.py)
CGAN_LAT_MIN, CGAN_LAT_MAX = -13.65, 24.65
CGAN_LON_MIN, CGAN_LON_MAX = 19.15, 54.25

# Calculate region indices for GEFS grid
# Note: GEFS lats go from 90 to -90 (north to south)
lat_mask = (GEFS_LATS >= CGAN_LAT_MIN) & (GEFS_LATS <= CGAN_LAT_MAX)
lon_mask = (GEFS_LONS >= CGAN_LON_MIN) & (GEFS_LONS <= CGAN_LON_MAX)
LAT_INDICES = np.where(lat_mask)[0]
LON_INDICES = np.where(lon_mask)[0]
REGION_LATS = GEFS_LATS[LAT_INDICES[0]:LAT_INDICES[-1]+1]
REGION_LONS = GEFS_LONS[LON_INDICES[0]:LON_INDICES[-1]+1]


# ==============================================================================
# PARQUET READING UTILITIES
# ==============================================================================

def read_parquet_refs(parquet_path: str) -> Dict:
    """Read parquet file and extract zstore references."""
    df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                value = json.loads(value)
            except:
                pass

        zstore[key] = value

    return zstore


def discover_variable_chunks(zstore: Dict, path_prefix: str) -> List[Tuple[int, str]]:
    """Discover chunks for a specific variable path in the zstore."""
    chunks = []
    chunk_pattern = re.compile(rf'^{re.escape(path_prefix)}/(\d+)\.0\.0$')

    for key in zstore.keys():
        match = chunk_pattern.match(key)
        if match:
            step_idx = int(match.group(1))
            chunks.append((step_idx, key))

    chunks.sort(key=lambda x: x[0])
    return chunks


def list_available_variables(zstore: Dict) -> List[str]:
    """List all variable path prefixes available in the zstore."""
    variables = set()
    for key in zstore.keys():
        # Extract variable path (first 4 components)
        parts = key.split('/')
        if len(parts) >= 4:
            var_prefix = '/'.join(parts[:4])
            if not any(ext in parts[-1] for ext in ['.zarray', '.zattrs', '.zgroup']):
                if '.' in parts[-1]:  # Likely a chunk reference
                    variables.add(var_prefix)

    return sorted(list(variables))


# ==============================================================================
# GRIBBERISH DATA STREAMING
# ==============================================================================

def fetch_grib_bytes(zstore: Dict, chunk_key: str, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    ref = zstore[chunk_key]

    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]
    else:
        raise ValueError(f"Invalid reference format for {chunk_key}: {ref}")

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish(grib_bytes: bytes, grid_shape=GEFS_GRID_SHAPE) -> np.ndarray:
    """Decode GRIB bytes using gribberish (fast path)."""
    if not GRIBBERISH_AVAILABLE:
        raise RuntimeError("gribberish not available")

    flat_array = gribberish.parse_grib_array(grib_bytes, 0)
    array_2d = flat_array.reshape(grid_shape)
    return array_2d


def decode_with_cfgrib(grib_bytes: bytes) -> np.ndarray:
    """Decode GRIB bytes using cfgrib (fallback path)."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        array_2d = ds[var_name].values.copy()
        ds.close()
    finally:
        os.unlink(tmp_path)

    return array_2d


def decode_grib_hybrid(grib_bytes: bytes, grid_shape=GEFS_GRID_SHAPE) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib on failure."""
    if GRIBBERISH_AVAILABLE:
        try:
            array_2d = decode_with_gribberish(grib_bytes, grid_shape)
            return array_2d, 'gribberish'
        except Exception:
            pass

    # Fallback to cfgrib
    array_2d = decode_with_cfgrib(grib_bytes)
    return array_2d, 'cfgrib'


# ==============================================================================
# MULTI-VARIABLE STREAMING
# ==============================================================================

def stream_variable_for_member(
    parquet_path: str,
    var_name: str,
    var_config: Dict,
    zarr_store: zarr.Group,
    member_idx: int,
    fs
) -> bool:
    """
    Stream a single variable for a single ensemble member.

    Args:
        parquet_path: Path to parquet reference file
        var_name: cGAN variable name (e.g., 'cape', 'pres')
        var_config: Configuration dict with gefs_prefix
        zarr_store: Zarr group to write data to
        member_idx: Index of this member in the zarr array
        fs: S3 filesystem instance

    Returns:
        True if successful, False otherwise
    """
    try:
        zstore = read_parquet_refs(parquet_path)
        path_prefix = var_config['gefs_prefix']
        chunks = discover_variable_chunks(zstore, path_prefix)

        if not chunks:
            # Try alternative path patterns
            alt_prefixes = [
                path_prefix.replace('atmosphereSingleLayer', 'entire atmosphere (considered as a single layer)'),
                path_prefix.replace('instant', 'accum'),
            ]
            for alt in alt_prefixes:
                chunks = discover_variable_chunks(zstore, alt)
                if chunks:
                    break

        if not chunks:
            print(f"      No chunks found for {var_name} (prefix: {path_prefix})")
            return False

        n_timesteps = len(chunks)

        for i, (step_idx, chunk_key) in enumerate(chunks):
            try:
                grib_bytes, _ = fetch_grib_bytes(zstore, chunk_key, fs)
                array_2d, decoder = decode_grib_hybrid(grib_bytes)

                # Subset to ICPAC region
                data_subset = array_2d[LAT_INDICES[0]:LAT_INDICES[-1]+1,
                                       LON_INDICES[0]:LON_INDICES[-1]+1]

                zarr_store[var_name][member_idx, i, :, :] = data_subset.astype(np.float32)

            except Exception as e:
                print(f"      Step {step_idx}: FAILED - {str(e)[:40]}")
                zarr_store[var_name][member_idx, i, :, :] = np.nan

        return True

    except Exception as e:
        print(f"    Error streaming {var_name}: {e}")
        return False


def stream_all_variables_for_member(
    parquet_path: str,
    zarr_store: zarr.Group,
    member_idx: int
) -> Dict[str, bool]:
    """
    Stream all cGAN variables for a single ensemble member.

    Returns dict of variable -> success status
    """
    member_name = Path(parquet_path).stem.split('_')[0]
    print(f"\n  Streaming {member_name} (member {member_idx + 1})...")

    fs = fsspec.filesystem('s3', anon=True)
    results = {}

    start_time = time.time()

    for var_name, var_config in CGAN_VARIABLES.items():
        print(f"    {var_name}...", end=' ', flush=True)
        success = stream_variable_for_member(
            parquet_path, var_name, var_config, zarr_store, member_idx, fs
        )
        results[var_name] = success
        status = "OK" if success else "FAILED"
        print(status)

    elapsed = time.time() - start_time
    successful = sum(1 for v in results.values() if v)
    print(f"    Completed: {successful}/{len(CGAN_VARIABLES)} variables in {elapsed:.1f}s")

    gc.collect()
    return results


# ==============================================================================
# ZARR STORAGE MANAGEMENT
# ==============================================================================

def create_cgan_zarr_store(
    output_dir: Path,
    n_members: int,
    n_timesteps: int
) -> Tuple[zarr.Group, Path]:
    """
    Create zarr store for cGAN input data with all required variables.
    """
    n_lats = len(REGION_LATS)
    n_lons = len(REGION_LONS)

    print(f"\n  Creating zarr store: {output_dir}")
    print(f"  Shape: ({n_members} members, {n_timesteps} steps, {n_lats} lat, {n_lons} lon)")

    output_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(output_dir), mode='w')

    # Create array for each cGAN variable
    for var_name in CGAN_VARIABLES.keys():
        store.create_dataset(
            var_name,
            shape=(n_members, n_timesteps, n_lats, n_lons),
            chunks=(1, n_timesteps, n_lats, n_lons),
            dtype=np.float32,
            fill_value=np.nan,
            compressors=[BloscCodec(cname='lz4', clevel=3)]
        )

    # Store coordinates (zarr v3 requires explicit shape)
    lat_arr = store.create_dataset('latitude', shape=(n_lats,), dtype=np.float64)
    lat_arr[:] = REGION_LATS
    lon_arr = store.create_dataset('longitude', shape=(n_lons,), dtype=np.float64)
    lon_arr[:] = REGION_LONS
    mem_arr = store.create_dataset('member', shape=(n_members,), dtype=np.int32)
    mem_arr[:] = np.arange(1, n_members + 1)

    # Store metadata
    store.attrs['n_members'] = n_members
    store.attrs['n_timesteps'] = n_timesteps
    store.attrs['variables'] = list(CGAN_VARIABLES.keys())
    store.attrs['region'] = {
        'lat_min': CGAN_LAT_MIN, 'lat_max': CGAN_LAT_MAX,
        'lon_min': CGAN_LON_MIN, 'lon_max': CGAN_LON_MAX
    }

    return store, output_dir


# ==============================================================================
# MAIN ROUTINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stream GEFS multi-variable data for cGAN inference'
    )
    parser.add_argument('--parquet_dir', type=str, default='output_parquet',
                       help='Directory containing parquet reference files')
    parser.add_argument('--output_dir', type=str, default='cgan_zarr_output',
                       help='Output directory for zarr store')
    parser.add_argument('--date', type=str, required=True,
                       help='Target date YYYYMMDD')
    parser.add_argument('--run', type=str, default='00',
                       help='Model run hour (default: 00)')
    parser.add_argument('--max_members', type=int, default=None,
                       help='Limit number of ensemble members (for testing)')

    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_base = Path(args.output_dir)
    target_date = args.date
    target_run = args.run

    print("="*70)
    print("GEFS Multi-Variable Data Streaming for cGAN")
    print("="*70)
    print(f"Target Date: {target_date}")
    print(f"Model Run: {target_run}Z")
    print(f"Parquet Dir: {parquet_dir}")
    print(f"Output Dir: {output_base}")
    print(f"Variables: {list(CGAN_VARIABLES.keys())}")
    print(f"Gribberish Available: {GRIBBERISH_AVAILABLE}")
    print(f"Region: {CGAN_LAT_MIN}N-{CGAN_LAT_MAX}N, {CGAN_LON_MIN}E-{CGAN_LON_MAX}E")
    print("="*70)

    start_time = time.time()

    # Find parquet files
    parquet_files = sorted(parquet_dir.glob(f"gep*_{target_date}_{target_run}z.parquet"))

    if not parquet_files:
        print(f"\nError: No parquet files found in {parquet_dir}")
        print(f"Expected pattern: gep*_{target_date}_{target_run}z.parquet")
        print("\nPlease run run_gefs_tutorial.py first to create parquet references.")
        return False

    if args.max_members:
        parquet_files = parquet_files[:args.max_members]

    n_members = len(parquet_files)
    print(f"\nFound {n_members} ensemble member files")

    # Determine number of timesteps from first file
    print("\n[Step 1] Discovering data structure...")
    zstore = read_parquet_refs(str(parquet_files[0]))

    # Find timesteps from first available variable
    n_timesteps = 0
    for var_name, var_config in CGAN_VARIABLES.items():
        chunks = discover_variable_chunks(zstore, var_config['gefs_prefix'])
        if chunks:
            n_timesteps = len(chunks)
            print(f"  Found {n_timesteps} timesteps from {var_name}")
            break

    if n_timesteps == 0:
        print("  Could not determine timesteps. Listing available paths:")
        available = list_available_variables(zstore)
        for v in available[:10]:
            print(f"    - {v}")
        print("\n  Using default 81 timesteps (10 days at 3-hour intervals)")
        n_timesteps = 81

    # Create zarr store
    print("\n[Step 2] Creating zarr storage...")
    output_path = output_base / f"{target_date}_{target_run}z"
    zarr_store, _ = create_cgan_zarr_store(output_path, n_members, n_timesteps)

    # Store additional metadata
    zarr_store.attrs['target_date'] = target_date
    zarr_store.attrs['target_run'] = target_run
    zarr_store.attrs['created'] = datetime.now().isoformat()

    # Stream data for all members
    print("\n[Step 3] Streaming GEFS Data")
    print("="*70)

    all_results = {}
    for member_idx, pf in enumerate(parquet_files):
        results = stream_all_variables_for_member(str(pf), zarr_store, member_idx)
        member_name = pf.stem.split('_')[0]
        all_results[member_name] = results

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("STREAMING COMPLETE!")
    print("="*70)

    # Count successes per variable
    print("\nVariable Success Rates:")
    for var_name in CGAN_VARIABLES.keys():
        successes = sum(1 for m, r in all_results.items() if r.get(var_name, False))
        print(f"  {var_name}: {successes}/{n_members} members")

    print(f"\nOutput: {output_path}")
    print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print("\n[Next Steps]")
    print(f"  1. Convert to NetCDF for cGAN:")
    print(f"     python zarr_to_raw_netcdf.py --input_dir {output_path} --output_dir cgan_raw_netcdf/{target_date[:4]} --date {target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}")
    print(f"  2. Or run the unified pipeline:")
    print(f"     python run_gefs_gik_cgan_pipeline.py --date {target_date}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
