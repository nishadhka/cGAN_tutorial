# Converting Zarr to NetCDF for Forecast Input

## Overview

The `forecast_gfs.py` script reads zarr files from the INPUT folder specified in `config/forecast_gfs.yaml`. This document explains how to convert those zarr files to NetCDF format.

## Current Configuration Flow

In `config/forecast_gfs.yaml`:
```yaml
INPUT:
    folder: "/home/nshruti_icpac_net/GEFS"
    dates: ["2024-04-20"]
    start_hour: 30
    end_hour: 54
```

The `forecast_gfs.py` script (lines 209-212):
1. Constructs zarr file path: `{input_folder}/{year}/{field}_{year}.zarr`
2. Opens with `xr.open_zarr()`
3. Selects the specified date and time steps
4. Processes the data

## Solution: Convert Zarr to NetCDF

### Step 1: Run the Conversion Script

```bash
cd /home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cGAN_tutorial/scripts

# Using config file
python convert_zarr_to_netcdf.py --config ../config/forecast_gfs.yaml

# Or with custom parameters
python convert_zarr_to_netcdf.py \
    --input_folder /home/nshruti_icpac_net/GEFS \
    --output_folder /home/nshruti_icpac_net/GEFS_netcdf \
    --dates 2024-04-20 \
    --year 2024
```

### Step 2: Update Configuration

Option A: Modify existing `config/forecast_gfs.yaml`:
```yaml
INPUT:
    folder: "/home/nshruti_icpac_net/GEFS_netcdf"  # Changed from GEFS to GEFS_netcdf
    dates: ["2024-04-20"]
    start_hour: 30
    end_hour: 54
```

Option B: Use the new config file `config/forecast_gfs_netcdf.yaml`

### Step 3: Modify forecast_gfs.py to Support NetCDF

In `forecast_gfs.py` line 210-212, change:
```python
# OLD (zarr):
input_file = f"{field}_{d.year}.zarr"
nc_in_path = os.path.join(input_folder_year, input_file)
nc_file = xr.open_zarr(nc_in_path)

# NEW (netcdf):
input_file = f"{field}_{d.year}.nc"
nc_in_path = os.path.join(input_folder_year, input_file)
nc_file = xr.open_dataset(nc_in_path)
```

## Files Converted

The script converts all fields in `all_fcst_fields`:
- `cape`
- `pres`
- `pwat`
- `tmp`
- `ugrd`
- `vgrd`
- `msl`
- `apcp`

Each field is converted from:
- **Input**: `/home/nshruti_icpac_net/GEFS/{year}/{field}_{year}.zarr`
- **Output**: `/home/nshruti_icpac_net/GEFS_netcdf/{year}/{field}_{year}.nc`

## Benefits of Using NetCDF

1. **Simpler file format** - More widely supported
2. **Easier debugging** - Can inspect with `ncdump`
3. **Better compatibility** - Works with more tools
4. **Same data structure** - No changes to data processing logic needed

## Troubleshooting

**Issue**: Missing zarr files
- Check that zarr files exist at: `/home/nshruti_icpac_net/GEFS/{year}/{field}_{year}.zarr`
- Verify the year in the path matches your dates

**Issue**: Permission errors
- Ensure you have write permissions to the output folder
- Create the output folder manually if needed

**Issue**: Memory errors
- Process one field at a time using `--fields` flag:
  ```bash
  python convert_zarr_to_netcdf.py --config ../config/forecast_gfs.yaml --fields cape
  ```

## Alternative: Keep Using Zarr

If you prefer to keep using zarr format, no conversion is needed. The current setup already works with zarr files using `xr.open_zarr()`.
