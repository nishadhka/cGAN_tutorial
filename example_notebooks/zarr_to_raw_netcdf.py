#!/usr/bin/env python3
"""
Convert GEFS zarr files to raw NetCDF format for cGAN inference.

This script converts zarr files (with member subdirectories) to the raw NetCDF format
expected by run_gefs_inference.py:
- Output: {field}_{year}.nc
- Dimensions: (time, member, step, latitude, longitude)
- Raw data WITHOUT ensemble statistics computed

The run_gefs_inference.py script will then:
1. Load raw data
2. Apply normalization
3. Compute ensemble mean/std internally

Usage:
    micromamba run -n zarrv3 python zarr_to_raw_netcdf.py \
        --input_dir /path/to/zarr_stores/20250918_00 \
        --output_dir /path/to/netcdf/2025 \
        --date 2025-09-18
"""

import os
import sys
import re
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

# Variable mapping: zarr_name -> cgan_name
VARIABLE_MAPPING = {
    'cape': 'cape',
    'sp': 'pres',
    'mslet': 'msl',
    'pwat': 'pwat',
    't2m': 'tmp',
    'u10': 'ugrd',
    'v10': 'vgrd',
    'tp': 'apcp',
}


def find_member_zarrs(input_dir: Path) -> dict:
    """Find all member zarr files in the input directory."""
    members = {}
    pattern = re.compile(r'gep(\d+).*\.zarr$')

    for item in input_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            match = pattern.match(item.name)
            if match:
                member_num = int(match.group(1))
                members[member_num] = item

    return dict(sorted(members.items()))


def load_and_combine_members(member_paths: dict, variables: list = None) -> xr.Dataset:
    """Load all member zarr files and combine along 'member' dimension."""
    datasets = []
    member_ids = []

    for member_num, zarr_path in member_paths.items():
        print(f"  Loading member {member_num:02d}: {zarr_path.name}")
        ds = xr.open_zarr(zarr_path)

        if variables:
            available_vars = [v for v in variables if v in ds.data_vars]
            if not available_vars:
                continue
            ds = ds[available_vars]

        datasets.append(ds)
        member_ids.append(member_num)

    if not datasets:
        raise ValueError("No valid datasets found")

    print(f"  Combining {len(datasets)} members...")
    combined = xr.concat(datasets, dim='member')
    combined['member'] = member_ids

    return combined


def convert_step_to_hours(ds: xr.Dataset) -> np.ndarray:
    """Convert step coordinate from timedelta64 to hours."""
    step_values = ds.step.values
    if np.issubdtype(step_values.dtype, np.timedelta64):
        hours = step_values.astype('timedelta64[h]').astype(float)
    else:
        hours = step_values / 3.6e12
    return hours


def save_raw_netcdf(combined_ds: xr.Dataset, output_dir: Path,
                    init_time: datetime, year: int):
    """
    Save raw NetCDF files per variable in format expected by run_gefs_inference.py.

    Output dimensions: (time, member, step, latitude, longitude)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Get coordinates
    lats = combined_ds.latitude.values
    lons = combined_ds.longitude.values
    members = combined_ds.member.values
    step_hours = convert_step_to_hours(combined_ds)

    # Create timedelta step coordinate (in nanoseconds for compatibility)
    step_ns = step_hours * 3.6e12  # Convert hours to nanoseconds

    for zarr_var in combined_ds.data_vars:
        cgan_name = VARIABLE_MAPPING.get(zarr_var, zarr_var)
        print(f"\n  Processing {zarr_var} -> {cgan_name}")

        # Get data: current shape is (member, step, latitude, longitude)
        data = combined_ds[zarr_var].values

        # Add time dimension: (1, member, step, latitude, longitude)
        data_with_time = np.expand_dims(data, axis=0)

        print(f"    Shape: {data_with_time.shape}")
        print(f"    Range: [{np.nanmin(data_with_time):.4f}, {np.nanmax(data_with_time):.4f}]")

        # Create dataset
        ds_out = xr.Dataset(
            {
                cgan_name: xr.DataArray(
                    data=data_with_time.astype(np.float32),
                    dims=['time', 'member', 'step', 'latitude', 'longitude'],
                    attrs={
                        'long_name': cgan_name,
                        'original_variable': zarr_var,
                    }
                )
            },
            coords={
                'time': [np.datetime64(init_time)],
                'member': members,
                'step': step_ns,  # In nanoseconds for compatibility
                'latitude': lats,
                'longitude': lons,
            }
        )

        ds_out.attrs.update({
            'title': f'GEFS {cgan_name} raw ensemble data',
            'source': 'GEFS (Global Ensemble Forecast System)',
            'created_by': 'zarr_to_raw_netcdf.py',
            'init_time': str(init_time),
        })

        output_file = output_dir / f"{cgan_name}_{year}.nc"
        print(f"    Saving: {output_file}")
        ds_out.to_netcdf(output_file)
        saved_files.append(output_file)

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert GEFS zarr to raw NetCDF format for run_gefs_inference.py'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing member zarr files (e.g., zarr_stores/20250918_00)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for NetCDF files (e.g., netcdf/2025)')
    parser.add_argument('--date', type=str, default=None,
                       help='Initialization date YYYYMMDD or YYYY-MM-DD (default: inferred from input_dir)')
    parser.add_argument('--run', type=str, default='00',
                       help='Model run hour (default: 00)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Infer date from directory name if not provided
    if args.date:
        date_str = args.date.replace('-', '')
    else:
        match = re.match(r'(\d{8})_(\d{2})', input_dir.name)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            print(f"Warning: Could not infer date, using {date_str}")

    year = int(date_str[:4])
    init_time = datetime.strptime(f"{date_str}{args.run}", '%Y%m%d%H')

    print("=" * 70)
    print("GEFS Zarr to Raw NetCDF Converter")
    print("=" * 70)
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Init time:        {init_time}")
    print(f"  Year:             {year}")
    print()

    # Step 1: Find member zarr files
    print("1. Finding member zarr files...")
    member_paths = find_member_zarrs(input_dir)
    print(f"   Found {len(member_paths)} members")

    if not member_paths:
        print("Error: No member zarr files found!")
        sys.exit(1)

    # Step 2: Load and combine members
    print("\n2. Loading and combining members...")
    combined_ds = load_and_combine_members(member_paths)
    print(f"   Combined shape: {dict(combined_ds.sizes)}")
    print(f"   Variables: {list(combined_ds.data_vars)}")

    # Check step range
    step_hours = convert_step_to_hours(combined_ds)
    print(f"   Step range: {step_hours.min():.0f} to {step_hours.max():.0f} hours")

    # Step 3: Save raw NetCDF files
    print("\n3. Saving raw NetCDF files...")
    saved_files = save_raw_netcdf(combined_ds, output_dir, init_time, year)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Output files saved to: {output_dir}")
    for f in saved_files:
        print(f"    - {f.name}")

    print("\n4. To run inference, update run_gefs_inference.py CONFIG:")
    print(f'   "input_folder": "{output_dir.parent}",')
    print(f'   "dates": ["{init_time.strftime("%Y-%m-%d")}"],')

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
