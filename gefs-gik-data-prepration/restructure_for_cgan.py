#!/usr/bin/env python3
"""
Restructure GEFS ensemble NetCDF files for cGAN inference.

This script converts ensemble NetCDF files from the format:
    (member, valid_times, latitude, longitude)
to the CGAN-expected format:
    (time, member, valid_time, latitude, longitude)

It also filters forecast hours to 30-54 (every 6 hours) as required by the cGAN model.
"""

import xarray as xr
import numpy as np
import os
from pathlib import Path
import argparse


# Variable name mapping: current_name -> cgan_name
VARIABLE_MAPPING = {
    'cape': 'cape',     # Convective Available Potential Energy
    'sp': 'pres',       # Surface Pressure -> Pressure
    'mslet': 'msl',     # Mean Sea Level Pressure
    'pwat': 'pwat',     # Precipitable Water
    't2m': 'tmp',       # 2m Temperature -> Temperature
    'u10': 'ugrd',      # 10m U-wind component
    'v10': 'vgrd',      # 10m V-wind component
    'tp': 'apcp',       # Total Precipitation -> Accumulated Precipitation
}


def restructure_netcdf_for_cgan(
    input_file: str,
    output_file: str,
    start_hour: int = 30,
    end_hour: int = 54,
    hour_interval: int = 6,
    year: int = 2025
):
    """
    Restructure a NetCDF file from ensemble format to CGAN format.

    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file with dimensions (member, valid_times, lat, lon)
    output_file : str
        Path to output NetCDF file
    start_hour : int
        Starting forecast hour (default: 30)
    end_hour : int
        Ending forecast hour (default: 54)
    hour_interval : int
        Interval between forecast hours (default: 6)
    year : int
        Year for the output file naming
    """

    print(f"\nProcessing: {input_file}")

    # Open dataset without decoding times to avoid conflicts
    ds = xr.open_dataset(input_file, decode_times=False)

    # Get the variable name (should be only one data variable)
    var_names = list(ds.data_vars)
    if len(var_names) != 1:
        raise ValueError(f"Expected 1 data variable, found {len(var_names)}: {var_names}")

    original_var_name = var_names[0]
    print(f"  Variable: {original_var_name}")

    # Map to CGAN variable name
    if original_var_name in VARIABLE_MAPPING:
        cgan_var_name = VARIABLE_MAPPING[original_var_name]
    else:
        print(f"  Warning: No mapping found for {original_var_name}, using original name")
        cgan_var_name = original_var_name

    # Convert step from nanoseconds to hours
    step_hours = ds.step.values / 3.6e12
    print(f"  Available forecast hours: {step_hours.min():.0f} to {step_hours.max():.0f}")

    # Filter to desired forecast hours
    target_hours = np.arange(start_hour, end_hour + 1, hour_interval)
    print(f"  Target forecast hours: {target_hours}")

    # Find indices of target hours
    indices = []
    for hour in target_hours:
        # Find the closest step to target hour
        idx = np.argmin(np.abs(step_hours - hour))
        if np.abs(step_hours[idx] - hour) < 0.5:  # Within 30 minutes
            indices.append(idx)
        else:
            print(f"  Warning: No step found near hour {hour}")

    if len(indices) == 0:
        raise ValueError(f"No matching forecast hours found between {start_hour} and {end_hour}")

    print(f"  Found {len(indices)} matching forecast hours: {step_hours[indices]}")

    # Select the data
    data_subset = ds[original_var_name].isel(valid_times=indices)

    # Get the time coordinate (initialization time) - should be constant
    if 'time' in ds.coords:
        init_time = ds.time.isel(valid_times=0).values
    else:
        init_time = 0  # Fallback if no time coordinate

    # Create new dimensions
    # Current: (member, valid_times, latitude, longitude)
    # Target:  (time, member, valid_time, latitude, longitude)

    # Drop coordinates that would conflict with new dimensions
    data_subset = data_subset.reset_coords(drop=True)

    # Add a singleton time dimension at the beginning
    data_reshaped = data_subset.expand_dims(dim={'init_time': 1}, axis=0)

    # Rename dimensions to match CGAN expectations
    data_reshaped = data_reshaped.rename({
        'valid_times': 'valid_time',
        'init_time': 'time'
    })

    # Create output dataset
    ds_out = xr.Dataset(
        {
            cgan_var_name: data_reshaped
        },
        coords={
            'time': [init_time],
            'member': ds.member.values,
            'valid_time': np.arange(len(indices)),  # Simple index for valid_time
            'latitude': ds.latitude.values,
            'longitude': ds.longitude.values,
        }
    )

    # Add attributes
    ds_out[cgan_var_name].attrs.update({
        'long_name': ds[original_var_name].attrs.get('long_name', cgan_var_name),
        'units': ds[original_var_name].attrs.get('units', ''),
        'original_variable': original_var_name,
    })

    # Add global attributes
    ds_out.attrs.update({
        'title': f'GEFS {cgan_var_name} for cGAN inference',
        'source': 'GEFS (Global Ensemble Forecast System)',
        'institution': 'NOAA/NCEP',
        'created_by': 'restructure_for_cgan.py',
        'description': f'Restructured ensemble forecast for cGAN processing',
        'forecast_hours': f'{start_hour}-{end_hour} (every {hour_interval}h)',
        'ensemble_size': len(ds.member),
        'original_file': os.path.basename(input_file),
    })

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to NetCDF
    print(f"  Saving to: {output_file}")
    print(f"  Output dimensions: {dict(ds_out.dims)}")

    ds_out.to_netcdf(output_file)

    # Close datasets
    ds.close()
    ds_out.close()

    print(f"  ✓ Complete")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Restructure GEFS ensemble NetCDF files for cGAN inference'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing ensemble_*.nc files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for restructured files'
    )
    parser.add_argument(
        '--start_hour',
        type=int,
        default=30,
        help='Starting forecast hour (default: 30)'
    )
    parser.add_argument(
        '--end_hour',
        type=int,
        default=54,
        help='Ending forecast hour (default: 54)'
    )
    parser.add_argument(
        '--hour_interval',
        type=int,
        default=6,
        help='Interval between forecast hours (default: 6)'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='Year for output file naming (default: 2025)'
    )
    parser.add_argument(
        '--variables',
        nargs='+',
        default=None,
        help='Specific variables to process (default: all)'
    )

    args = parser.parse_args()

    # Find all ensemble files (excluding mean and std)
    input_dir = Path(args.input_dir)
    ensemble_files = sorted(input_dir.glob('ensemble_*.nc'))
    ensemble_files = [
        f for f in ensemble_files
        if 'mean' not in f.name and 'std' not in f.name
    ]

    if not ensemble_files:
        print(f"No ensemble files found in {input_dir}")
        return

    print(f"Found {len(ensemble_files)} ensemble files to process")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for input_file in ensemble_files:
        # Extract variable name from filename (ensemble_VARNAME.nc)
        var_name = input_file.stem.replace('ensemble_', '')

        # Skip if specific variables requested and this isn't one
        if args.variables and var_name not in args.variables:
            print(f"Skipping {var_name} (not in requested variables)")
            continue

        # Get the CGAN variable name
        cgan_var = VARIABLE_MAPPING.get(var_name, var_name)

        # Create output filename: {cgan_var}_{year}.nc
        output_file = output_dir / f"{cgan_var}_{args.year}.nc"

        try:
            restructure_netcdf_for_cgan(
                str(input_file),
                str(output_file),
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                hour_interval=args.hour_interval,
                year=args.year
            )
        except Exception as e:
            print(f"  ✗ Error processing {input_file}: {e}")
            continue

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output files saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
