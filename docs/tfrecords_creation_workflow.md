# TFRecords Creation Workflow for cGAN Training

## Overview
This document explains how to create TFRecords from forecast and truth data for cGAN model training using a cost-optimized cloud infrastructure. The workflow is designed to work efficiently with both GPU and light non-GPU machines using the Terraform setup in `tf_gpu/`.

---

## Architecture

### Dual-SSD Setup on Google Cloud Platform

The infrastructure uses a **cost-optimized dual-SSD strategy**:

1. **50GB Persistent SSD** (`/dev/nvme0n2` → `/mnt/python-env`)
   - Stores Python environment with all libraries
   - Survives VM destruction/recreation
   - Prevents reinstallation overhead
   - ~$8.50/month when VM is down

2. **500GB Temporary SSD** (`/dev/nvme0n3` → `/mnt/training-data`)
   - Stores downloaded forecast/truth data during processing
   - Deleted after TFRecords creation to save costs
   - Fresh formatting each session
   - ~85% storage cost savings

3. **GCS Bucket**
   - Source repository for NetCDF files
   - Long-term storage for processed TFRecords
   - Multi-region for data redundancy

### Cost Breakdown
- **Per 8-hour session**: ~$22-27 (VM + temporary disk)
- **Storage only** (no VM running): ~$8.50/month (persistent disk only)
- **Key savings**: No need to keep 500GB disk when not processing

---

## Prerequisites

### 1. Google Cloud Platform Setup
- GCP project with billing enabled
- Service account with permissions:
  - Compute Engine Admin
  - Storage Admin
  - Service Account User
- APIs enabled:
  - Compute Engine API
  - Cloud Storage API

### 2. Local Requirements
- Terraform installed
- gcloud CLI installed and authenticated
- Service account JSON key file

### 3. Data Requirements
- Forecast data in GCS bucket (NetCDF format)
- Truth data in GCS bucket (IMERG precipitation data)
- Constants (elevation, land-sea mask)

---

## Step-by-Step Workflow

## Phase 1: Infrastructure Deployment

### 1.1 Configure Terraform Variables

Navigate to the `tf_gpu/` directory and set up your configuration:

```bash
cd tf_gpu/

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

**Required variables in `terraform.tfvars`:**
```hcl
project_id       = "your-gcp-project-id"
region          = "europe-west2"
zone            = "europe-west2-b"
vm_name         = "cgan-tfrecords-vm"
machine_type    = "n1-standard-8"  # or g2-standard-8 for GPU
service_account = "your-service-account@project.iam.gserviceaccount.com"
```

### 1.2 Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview what will be created
terraform plan

# Deploy the infrastructure
terraform apply
```

**What gets created:**
- VM instance (CPU or GPU based on machine_type)
- 50GB persistent disk (with lifecycle protection)
- 500GB temporary disk (for data processing)
- Firewall rules and networking

---

## Phase 2: VM Setup and Disk Configuration

### 2.1 SSH into the VM

```bash
gcloud compute ssh cgan-tfrecords-vm --zone=europe-west2-b
```

### 2.2 First-Time Disk Setup

**IMPORTANT**: Only run the format commands (`mkfs.ext4`) on first setup. On subsequent sessions, skip formatting the persistent disk!

```bash
# Create mount points
sudo mkdir -p /mnt/python-env /mnt/training-data

# Check attached disks
lsblk
# Expected output:
# nvme0n1 (boot disk - already mounted)
# nvme0n2 (50GB persistent - needs mounting)
# nvme0n3 (500GB temporary - needs formatting and mounting)

# ONE-TIME ONLY: Format persistent Python environment disk
# WARNING: This erases all data! Only do this the FIRST time!
sudo mkfs.ext4 /dev/nvme0n2

# Mount persistent disk
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env

# Format temporary training data disk (do this EVERY session)
sudo mkfs.ext4 /dev/nvme0n3
sudo mount /dev/nvme0n3 /mnt/training-data
sudo chown -R $USER:$USER /mnt/training-data

# Create symbolic links for convenience
ln -sf /mnt/python-env ~/python-env
ln -sf /mnt/training-data ~/training-data

# Verify mounts
df -h /mnt/python-env /mnt/training-data
```

### 2.3 Subsequent Sessions (Disk Already Set Up)

For all future sessions after the first one:

```bash
# Create mount points
sudo mkdir -p /mnt/python-env /mnt/training-data

# Mount persistent disk (already formatted, don't format again!)
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env

# Format and mount temporary disk (fresh each time)
sudo mkfs.ext4 /dev/nvme0n3
sudo mount /dev/nvme0n3 /mnt/training-data
sudo chown -R $USER:$USER /mnt/training-data

# Recreate symbolic links
ln -sf /mnt/python-env ~/python-env
ln -sf /mnt/training-data ~/training-data
```

---

## Phase 3: Python Environment Setup

### 3.1 Install Micromamba (First Time Only)

Micromamba provides fast, lightweight package management and will be installed on the persistent disk.

```bash
# Set environment variables for installation
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
export MAMBA_EXE="$HOME/python-env/.local/bin/micromamba"

# Download and install Micromamba to persistent disk
"${SHELL}" <(curl -L micro.mamba.pm/install.sh) -b -p "$HOME/python-env/.local"

# Add to PATH permanently
echo 'export PATH="$HOME/python-env/.local/bin:$PATH"' >> ~/.bashrc
echo 'export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
micromamba --version
```

### 3.2 Create TensorFlow Environment (First Time Only)

```bash
# Create Python 3.11 environment
micromamba create -n tf215gpu python=3.11 -y

# Activate environment
micromamba activate tf215gpu

# Install TensorFlow 2.15
pip install tensorflow==2.15

# Install core scientific libraries
pip install numba matplotlib seaborn numpy pandas scipy

# Install geospatial and climate data libraries
pip install cartopy xarray netcdf4 cfgrib iris regionmask xesmf

# Install machine learning and utility libraries
pip install scikit-learn dask jupyter tqdm properscoring climlab \
    ecmwf-api-client flake8 schedule joblib

# Install Google Cloud Storage client
pip install google-cloud-storage

# Verify TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### 3.3 Create Activation Script

```bash
cat > ~/python-env/activate-tf.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/python-env/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
micromamba activate tf215gpu
echo "====================================="
echo "TensorFlow environment activated!"
echo "Python: $(which python)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "====================================="
EOF

chmod +x ~/python-env/activate-tf.sh
```

### 3.4 Subsequent Sessions (Environment Already Created)

For future sessions, simply activate the existing environment:

```bash
source ~/python-env/activate-tf.sh
```

---

## Phase 4: Data Download from GCS

### 4.1 Prepare Download Script

The `tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py` script enables parallel downloads from GCS.

**Key features:**
- Parallel downloads (16 workers by default)
- Skip existing files (resume capability)
- Retry logic for transient errors
- Progress reporting

### 4.2 Download Forecast Data

```bash
# Activate environment
source ~/python-env/activate-tf.sh

# Download forecast data
python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://your-bucket-name/forecast-data/2018/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/FORECAST/ \
  --skip-existing \
  --workers 16
```

**Expected structure:**
```
/mnt/training-data/FORECAST/
├── 20180101/
│   ├── cape_mean_std.nc
│   ├── cp_mean_std.nc
│   ├── t2m_mean_std.nc
│   └── ...
├── 20180102/
└── ...
```

### 4.3 Download Truth Data

```bash
python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://your-bucket-name/truth-data/IMERG/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/TRUTH/ \
  --skip-existing \
  --workers 16
```

**Expected structure:**
```
/mnt/training-data/TRUTH/
├── 20180101_6hr_IMERG.nc
├── 20180102_6hr_IMERG.nc
└── ...
```

### 4.4 Download Constants

```bash
python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://your-bucket-name/constants/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/CONSTANTS/ \
  --skip-existing \
  --workers 16
```

**Expected files:**
```
/mnt/training-data/CONSTANTS/
├── elevation.nc
├── lsm.nc
└── FCSTNorm2018.pkl (if pre-generated)
```

### 4.5 Verify Downloads

```bash
# Check data structure
tree -L 2 /mnt/training-data/

# Check disk usage
df -h /mnt/training-data

# Count forecast files
find /mnt/training-data/FORECAST -name "*.nc" | wc -l

# Count truth files
find /mnt/training-data/TRUTH -name "*.nc" | wc -l
```

---

## Phase 5: Configuration Setup

### 5.1 Clone Repository (if not already done)

```bash
cd ~
git clone https://github.com/snath-xoc/cGAN_tutorial.git
cd cGAN_tutorial
```

### 5.2 Update Data Paths Configuration

Edit `config/data_paths.yaml` to add a new environment for the VM:

```bash
nano config/data_paths.yaml
```

Add the following configuration:

```yaml
VM_SESSION:
  GENERAL:
    TRUTH_PATH: '/mnt/training-data/TRUTH/'
    FORECAST_PATH: '/mnt/training-data/FORECAST/'
    CONSTANTS_PATH: '/mnt/training-data/CONSTANTS/'

  TFRecords:
    tfrecords_path: '/mnt/training-data/tfrecords/'
```

### 5.3 Update Local Configuration

Edit `config/local_config.yaml`:

```bash
nano config/local_config.yaml
```

Set the machine to use:

```yaml
MACHINE: 'VM_SESSION'
```

---

## Phase 6: TFRecords Creation

### 6.1 Understanding the Process

The TFRecords creation involves:

1. **Generate forecast normalization constants** (`FCSTNorm{year}.pkl`)
   - Calculates mean, std, min, max for each forecast variable
   - Used to normalize inputs during training
   - Stored in CONSTANTS_PATH

2. **Process data into TFRecords**
   - Loads forecast and truth data for each date
   - Patches domain into 128x128 pixel blocks
   - Bins data by precipitation class
   - Writes compressed TFRecord files

**TFRecord naming convention:**
```
{year}_{leadtime}.{class}.tfrecords
```
- `year`: Training year (e.g., 2018)
- `leadtime`: Forecast lead time in hours (30, 36, 42, 48)
- `class`: Precipitation bin (0-8, representing different rainfall amounts)

### 6.2 Create TFRecords via Python Script

```bash
# Activate environment
source ~/python-env/activate-tf.sh

# Navigate to repository
cd ~/cGAN_tutorial

# Run TFRecords creation script
python << 'EOFPYTHON'
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(1, "./")
from data import write_data, gen_fcst_norm
from config import get_data_paths
import joblib

# Get configured data paths
data_paths = get_data_paths()
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]
tfrecords_path = data_paths["TFRecords"]["tfrecords_path"]

print(f"Forecast path: {data_paths['GENERAL']['FORECAST_PATH']}")
print(f"Truth path: {data_paths['GENERAL']['TRUTH_PATH']}")
print(f"Constants path: {CONSTANTS_PATH}")
print(f"TFRecords will be written to: {tfrecords_path}")

# Generate normalization constants for reference year
FCSTNorm_year = "2018"
norm_file = f"{CONSTANTS_PATH}/FCSTNorm{FCSTNorm_year}.pkl"

if not os.path.exists(norm_file):
    print(f"\nGenerating forecast normalization constants for {FCSTNorm_year}...")
    gen_fcst_norm(year=FCSTNorm_year)
    fcstNorm = joblib.load(norm_file)
    print("Normalization constants generated:")
    for var, stats in fcstNorm.items():
        print(f"  {var}: min={stats['min']:.2f}, max={stats['max']:.2f}")
else:
    print(f"\nLoading existing normalization constants from {norm_file}")
    fcstNorm = joblib.load(norm_file)

# Create TFRecords for specified years
years = [2018, 2019, 2020]  # Modify as needed

for year in years:
    print(f"\n{'='*60}")
    print(f"Processing year {year}...")
    print(f"{'='*60}")

    try:
        write_data(year)
        print(f"\n✓ Completed year {year}")
    except Exception as e:
        print(f"\n✗ Error processing year {year}: {str(e)}")
        continue

print(f"\n{'='*60}")
print("TFRecords creation completed!")
print(f"{'='*60}")
print(f"Output directory: {tfrecords_path}")

# List created files
import glob
tfrecord_files = sorted(glob.glob(f"{tfrecords_path}/*.tfrecords"))
print(f"\nTotal TFRecord files created: {len(tfrecord_files)}")
if tfrecord_files:
    print("\nSample files:")
    for f in tfrecord_files[:5]:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size_mb:.2f} MB)")
EOFPYTHON
```

### 6.3 Alternative: Run via Jupyter Notebook

```bash
# Start Jupyter notebook server
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# Note the token from the output, then create SSH tunnel from local machine:
# ssh -L 8888:localhost:8888 user@vm-ip-address

# Open browser to: http://localhost:8888
# Navigate to example_notebooks/create_tfrecords.ipynb
```

### 6.4 Monitor Progress

The script will output progress for each time step:

```
Doing time index 30
100%|████████████████████| 8/8 [00:25<00:00,  3.20s/it]

Doing time index 36
100%|████████████████████| 8/8 [00:25<00:00,  3.19s/it]
...
```

**Expected processing time:**
- ~2-4 minutes per year per lead time
- ~15-30 minutes total per year (4 lead times)
- ~1-2 hours for 3 years

---

## Phase 7: Validation

### 7.1 Verify TFRecords Creation

```bash
source ~/python-env/activate-tf.sh
cd ~/cGAN_tutorial

python << 'EOFPYTHON'
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from config import get_data_paths
from data import _parse_batch
import glob

# Get TFRecords path
data_paths = get_data_paths()
tfrecords_path = data_paths["TFRecords"]["tfrecords_path"]

# List all TFRecord files
tfrecord_files = sorted(glob.glob(f"{tfrecords_path}/*.tfrecords"))
print(f"Total TFRecord files: {len(tfrecord_files)}\n")

# Group by year and lead time
from collections import defaultdict
by_year_leadtime = defaultdict(list)
for f in tfrecord_files:
    basename = os.path.basename(f)
    parts = basename.split('.')
    year_leadtime = parts[0]  # e.g., "2018_36"
    by_year_leadtime[year_leadtime].append(f)

print("Files by year and lead time:")
for key, files in sorted(by_year_leadtime.items()):
    total_size = sum(os.path.getsize(f) for f in files) / (1024**3)
    print(f"  {key}: {len(files)} files, {total_size:.2f} GB")

# Test reading a sample file
if tfrecord_files:
    sample_file = tfrecord_files[0]
    print(f"\nTesting sample file: {os.path.basename(sample_file)}")

    dataset = tf.data.TFRecordDataset(sample_file, compression_type='GZIP')
    dataset = dataset.map(lambda x: _parse_batch(x,
                                                   insize=(128,128,56),
                                                   consize=(128,128,2),
                                                   outsize=(128,128,1)))

    # Read one batch
    for inputs, outputs in dataset.take(1):
        print(f"  lo_res_inputs shape: {inputs['lo_res_inputs'].shape}")
        print(f"  hi_res_inputs shape: {inputs['hi_res_inputs'].shape}")
        print(f"  output shape: {outputs['output'].shape}")
        print(f"  mask shape: {outputs['mask'].shape}")
        print("\n  ✓ TFRecord file is valid and readable!")

EOFPYTHON
```

### 7.2 Check Disk Usage

```bash
# Check space used
du -sh /mnt/training-data/tfrecords/

# Check remaining space
df -h /mnt/training-data
```

---

## Phase 8: Upload Results to GCS

### 8.1 Upload TFRecords

```bash
# Upload all TFRecords to GCS
gsutil -m cp -r /mnt/training-data/tfrecords/* gs://your-bucket-name/tfrecords/

# Upload normalization constants
gsutil -m cp /mnt/training-data/CONSTANTS/FCSTNorm*.pkl gs://your-bucket-name/constants/
```

### 8.2 Verify Upload

```bash
# List uploaded files
gsutil ls -lh gs://your-bucket-name/tfrecords/

# Check total size
gsutil du -sh gs://your-bucket-name/tfrecords/
```

---

## Phase 9: Cleanup and Cost Optimization

### 9.1 Safe Cleanup (Recommended)

This preserves your Python environment while removing expensive compute resources:

```bash
# Exit the VM
exit

# From your local machine, in the tf_gpu/ directory
cd tf_gpu/

# Destroy only VM and temporary disk (keeps 50GB persistent disk)
terraform destroy -target=google_compute_instance.cgan_training_vm
terraform destroy -target=google_compute_disk.temp_training_data
```

**What this preserves:**
- 50GB persistent disk with Python environment (~$8.50/month)
- All service accounts and IAM settings
- Network configurations

**What this deletes:**
- VM instance (no more compute charges)
- 500GB temporary disk (saves ~$85/month)

### 9.2 Restart for Next Session

When you need to process more data:

```bash
# The persistent disk will automatically reattach
terraform apply

# SSH and remount disks (see Phase 2.3)
gcloud compute ssh cgan-tfrecords-vm --zone=europe-west2-b
```

### 9.3 Complete Cleanup (When Totally Done)

**WARNING**: This deletes EVERYTHING including your Python environment!

```bash
# Only do this when completely finished with the project
terraform destroy

# Confirm: yes
```

---

## Troubleshooting

### Issue: Disk Not Mounting

```bash
# Check if disks are attached
lsblk

# Manually mount if needed
sudo mount /dev/nvme0n2 /mnt/python-env
sudo mount /dev/nvme0n3 /mnt/training-data

# Check mount status
mount | grep nvme
```

### Issue: Micromamba Not Found

```bash
# Add to PATH
export PATH="$HOME/python-env/.local/bin:$PATH"
source ~/.bashrc

# Verify
which micromamba
```

### Issue: Out of Space on Temporary Disk

```bash
# Check usage
df -h /mnt/training-data

# Remove old downloaded data if needed
rm -rf /mnt/training-data/FORECAST/2017*
rm -rf /mnt/training-data/TRUTH/2017*

# Or increase temporary disk size in terraform.tfvars:
# temp_disk_size_gb = 750
```

### Issue: GCS Download Fails

```bash
# Test authentication
gcloud auth list

# Test GCS access
gsutil ls gs://your-bucket-name/

# Re-authenticate if needed
gcloud auth login

# Verify service account key
ls -lh ~/service-account-key.json
```

### Issue: Python Package Import Errors

```bash
# Reactivate environment
source ~/python-env/activate-tf.sh

# Reinstall problematic package
pip install --force-reinstall package-name

# Check installed packages
pip list | grep tensorflow
```

### Issue: TFRecords Creation Fails

```bash
# Check data paths configuration
python -c "from config import get_data_paths; import pprint; pprint.pprint(get_data_paths())"

# Verify data files exist
ls -lh /mnt/training-data/FORECAST/20180101/
ls -lh /mnt/training-data/TRUTH/

# Check available memory
free -h

# Check for Python errors
python -c "from data import write_data; write_data(2018)" 2>&1 | tee error.log
```

---

## Summary

### What This Workflow Accomplishes

1. ✅ Deploys cost-optimized cloud infrastructure
2. ✅ Sets up persistent Python environment (survives VM destruction)
3. ✅ Downloads training data from GCS efficiently
4. ✅ Generates forecast normalization constants
5. ✅ Creates TFRecords for cGAN training
6. ✅ Uploads results back to GCS
7. ✅ Minimizes costs by deleting temporary resources

### Key Benefits

- **Cost Savings**: 85% reduction in storage costs by using temporary disk
- **Time Savings**: No Python reinstallation on subsequent sessions
- **Scalability**: Process multiple years of data efficiently
- **Reproducibility**: Consistent environment across sessions
- **Flexibility**: Easy to restart processing anytime

### Next Steps

After creating TFRecords, you can:

1. **Train cGAN model** using the TFRecords
2. **Validate model performance** on test data
3. **Run inference** for operational forecasting
4. **Iterate** on model architecture and hyperparameters

See `example_notebooks/` for training and inference examples.

---

## Quick Reference Commands

```bash
# Deploy infrastructure
cd tf_gpu && terraform apply

# SSH to VM
gcloud compute ssh cgan-tfrecords-vm --zone=europe-west2-b

# Mount disks (subsequent sessions)
sudo mkdir -p /mnt/python-env /mnt/training-data
sudo mount /dev/nvme0n2 /mnt/python-env
sudo mkfs.ext4 /dev/nvme0n3 && sudo mount /dev/nvme0n3 /mnt/training-data

# Activate environment
source ~/python-env/activate-tf.sh

# Download data
python gcs_folder_download.py gs://bucket/path --creds key.json --dest /mnt/training-data/

# Create TFRecords
cd ~/cGAN_tutorial
python -c "from data import write_data; write_data(2018)"

# Upload results
gsutil -m cp -r /mnt/training-data/tfrecords/* gs://bucket/tfrecords/

# Cleanup (safe)
exit
cd tf_gpu && terraform destroy -target=google_compute_instance.cgan_training_vm

# Restart next time
terraform apply
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Repository**: https://github.com/snath-xoc/cGAN_tutorial
