# TFRecords Creation Infrastructure (CPU VM)

This Terraform configuration creates a cost-optimized Google Cloud Platform infrastructure specifically for creating TFRecords from forecast data. Unlike the GPU setup in `tf_gpu/`, this uses a CPU-only VM which is sufficient for data processing tasks.

## Architecture Overview

- **CPU VM**: n2-standard-8 (8 vCPUs, 32 GB RAM) - sufficient for TFRecords creation
- **Persistent SSD**: 50GB for Python environment (preserved across sessions)
- **Temporary SSD**: 500GB for data processing (deleted after TFRecords creation)
- **GCS Integration**: Bucket for downloading source data and uploading TFRecords

## Cost Optimization Strategy

### Dual SSD Approach
1. **Persistent Disk (50GB)**: Stores Python environment, libraries, and configurations
   - Preserved between processing sessions
   - Prevents reinstallation overhead
   - Lifecycle protection enabled
   - ~$8.50/month when VM is down

2. **Temporary Disk (500GB)**: Stores downloaded data during processing
   - Created only when needed
   - Deleted immediately after TFRecords creation
   - Reduces storage costs by ~85%
   - Fresh format each session

### Economic Benefits
- **Storage Cost Savings**: ~$85/month saved by deleting 500GB temp disk after processing
- **Setup Time Reduction**: Persistent Python environment eliminates reinstallation
- **Lower Compute Costs**: CPU VM is ~80% cheaper than GPU VM
- **Preemptible Option**: Further 60-80% savings if VM termination is acceptable

### Cost Comparison

**CPU VM (this setup) - Per 8-hour session:**
- CPU VM (n2-standard-8): ~$0.40/hour = $3.20
- Temporary 500GB SSD: ~$2.83/day
- Persistent 50GB SSD: ~$0.28/day
- **Total**: ~$6-7 per session

**GPU VM (tf_gpu/) - Per 8-hour session:**
- GPU VM (g2-standard-8): ~$2.40/hour = $19.20
- Storage: ~$3.11/day
- **Total**: ~$22-25 per session

**Savings**: ~$16-18 per session using CPU VM for TFRecords creation

## Prerequisites

### 1. GCP Project Setup
- GCP project with billing enabled
- Service account with required permissions:
  - Compute Engine Admin
  - Storage Admin
  - Service Account User
- APIs enabled:
  - Compute Engine API
  - Cloud Storage API

### 2. Local Requirements
- Terraform >= 1.0 installed
- gcloud CLI installed and authenticated
- Service account JSON key file downloaded

### 3. GCS Bucket
- Existing GCS bucket with forecast/truth data, OR
- Set `create_gcs_bucket = true` to create a new bucket

## Setup Instructions

### 1. Configure Variables

```bash
cd tf_vm_tfrecords/

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

**Required variables:**
```hcl
project_id           = "your-gcp-project-id"
credentials_file     = "path/to/service-account.json"
training_bucket_name = "your-data-bucket-name"
```

**Optional but recommended:**
```hcl
machine_type = "n2-standard-8"  # Adjust based on data size
preemptible  = false            # Set true for cost savings
temp_disk_size = 500            # Adjust based on data volume
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review Planned Changes

```bash
terraform plan
```

This will show:
- VM instance to be created
- Two disks to be created (50GB persistent + 500GB temporary)
- Estimated costs

### 4. Deploy Infrastructure

```bash
terraform apply
```

Review the plan and type `yes` to proceed.

**What gets created:**
- CPU VM instance (n2-standard-8 by default)
- 50GB persistent SSD disk (with destroy protection)
- 500GB temporary SSD disk
- Network configurations
- Optional: GCS bucket (if `create_gcs_bucket = true`)

## VM Setup and Usage

### First-Time Setup

After deployment, SSH into the VM:

```bash
# Get the SSH command from Terraform output
terraform output ssh_command

# Or manually:
gcloud compute ssh tfrecords-vm --zone=europe-west2-b
```

#### Mount and Format Disks (First Time Only)

```bash
# Create mount points
sudo mkdir -p /mnt/python-env /mnt/training-data

# Check attached disks
lsblk
# Expected: nvme0n1 (boot), nvme0n2 (50GB), nvme0n3 (500GB)

# ONE-TIME: Format persistent disk (WARNING: Only do this once!)
sudo mkfs.ext4 /dev/nvme0n2

# Mount persistent disk
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env

# Format and mount temporary disk (do this every session)
sudo mkfs.ext4 /dev/nvme0n3
sudo mount /dev/nvme0n3 /mnt/training-data
sudo chown -R $USER:$USER /mnt/training-data

# Create symbolic links
ln -sf /mnt/python-env ~/python-env
ln -sf /mnt/training-data ~/training-data

# Verify mounts
df -h /mnt/python-env /mnt/training-data
```

#### Install Python Environment (First Time Only)

```bash
# Install Micromamba to persistent disk
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
export MAMBA_EXE="$HOME/python-env/.local/bin/micromamba"

"${SHELL}" <(curl -L micro.mamba.pm/install.sh) -b -p "$HOME/python-env/.local"

# Add to PATH
echo 'export PATH="$HOME/python-env/.local/bin:$PATH"' >> ~/.bashrc
echo 'export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"' >> ~/.bashrc
source ~/.bashrc

# Create TensorFlow environment
micromamba create -n tf215 python=3.11 -y
micromamba activate tf215

# Install TensorFlow (CPU version is sufficient for TFRecords)
pip install tensorflow==2.15

# Install required libraries
pip install numba matplotlib seaborn numpy pandas scipy \
    cartopy xarray netcdf4 cfgrib iris regionmask xesmf \
    scikit-learn dask jupyter tqdm properscoring climlab \
    ecmwf-api-client flake8 schedule joblib google-cloud-storage

# Create activation script
cat > ~/python-env/activate-tf.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/python-env/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
micromamba activate tf215
echo "TensorFlow CPU environment activated!"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
EOF

chmod +x ~/python-env/activate-tf.sh
```

### Subsequent Sessions

For all future sessions after the first setup:

```bash
# SSH into VM
gcloud compute ssh tfrecords-vm --zone=europe-west2-b

# Mount disks
sudo mkdir -p /mnt/python-env /mnt/training-data
sudo mount /dev/nvme0n2 /mnt/python-env  # Already formatted
sudo chown -R $USER:$USER /mnt/python-env

sudo mkfs.ext4 /dev/nvme0n3  # Fresh format each time
sudo mount /dev/nvme0n3 /mnt/training-data
sudo chown -R $USER:$USER /mnt/training-data

# Activate environment
source ~/python-env/activate-tf.sh
```

## TFRecords Creation Workflow

### 1. Download Data from GCS

```bash
# Activate environment
source ~/python-env/activate-tf.sh

# Clone repository (if not already done)
git clone https://github.com/snath-xoc/cGAN_tutorial.git ~/cGAN_tutorial

# Download data using the GCS download script
python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://YOUR-BUCKET/forecast-data/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/FORECAST/ \
  --skip-existing \
  --workers 16

python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://YOUR-BUCKET/truth-data/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/TRUTH/ \
  --skip-existing \
  --workers 16

python ~/cGAN_tutorial/tf_gpu/tensorflow-env-data-setup/gcs_folder_download.py \
  gs://YOUR-BUCKET/constants/ \
  --creds ~/service-account-key.json \
  --dest /mnt/training-data/CONSTANTS/ \
  --skip-existing \
  --workers 16
```

### 2. Configure Data Paths

Edit `~/cGAN_tutorial/config/data_paths.yaml`:

```yaml
VM_SESSION:
  GENERAL:
    TRUTH_PATH: '/mnt/training-data/TRUTH/'
    FORECAST_PATH: '/mnt/training-data/FORECAST/'
    CONSTANTS_PATH: '/mnt/training-data/CONSTANTS/'

  TFRecords:
    tfrecords_path: '/mnt/training-data/tfrecords/'
```

Edit `~/cGAN_tutorial/config/local_config.yaml`:

```yaml
MACHINE: 'VM_SESSION'
```

### 3. Create TFRecords

```bash
cd ~/cGAN_tutorial

python << 'EOFPYTHON'
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(1, "./")
from data import write_data, gen_fcst_norm

# Generate normalization constants
gen_fcst_norm(year="2018")

# Create TFRecords for multiple years
for year in [2018, 2019, 2020]:
    print(f"Processing year {year}...")
    write_data(year)
    print(f"Completed year {year}")
EOFPYTHON
```

### 4. Upload Results to GCS

```bash
# Upload TFRecords
gsutil -m cp -r /mnt/training-data/tfrecords/* gs://YOUR-BUCKET/tfrecords/

# Upload normalization constants
gsutil -m cp /mnt/training-data/CONSTANTS/FCSTNorm*.pkl gs://YOUR-BUCKET/constants/
```

## Resource Management

### Safe Cleanup (Recommended)

Preserve the Python environment while removing expensive compute:

```bash
# Exit VM
exit

# From local machine, in tf_vm_tfrecords/
terraform destroy -target=google_compute_instance.tfrecords_vm
terraform destroy -target=google_compute_disk.temp_training_data
```

**Preserves:**
- 50GB persistent disk (~$8.50/month)
- GCS bucket and data

**Deletes:**
- VM instance (no compute costs)
- 500GB temporary disk (saves ~$85/month)

### Restart for Next Session

```bash
terraform apply
```

The persistent disk automatically reattaches with your Python environment intact.

### Complete Cleanup

**WARNING**: Deletes everything including Python environment!

```bash
# First, disable lifecycle protection
# Edit main.tf: Comment out the lifecycle block in persistent_python_env

terraform destroy
```

## Monitoring and Optimization

### Check Disk Usage

```bash
# Overall usage
df -h /mnt/training-data

# Detailed breakdown
du -sh /mnt/training-data/*/
```

### Monitor Processing

```bash
# CPU usage
top

# Memory usage
free -h

# Disk I/O
iostat -x 1
```

### Cost Optimization Tips

1. **Use Preemptible VMs**: Set `preemptible = true` for 60-80% savings
   - Risk: VM can be terminated after 24 hours or during high demand
   - Good for: Non-urgent batch processing

2. **Right-size Machine Type**:
   - Small datasets (<100GB): `n2-standard-4` (4 vCPUs, 16GB)
   - Medium datasets (100-300GB): `n2-standard-8` (8 vCPUs, 32GB)
   - Large datasets (>300GB): `n2-standard-16` (16 vCPUs, 64GB)

3. **Adjust Disk Sizes**:
   - `temp_disk_size`: Set to 1.2-1.5x your raw data size
   - `persistent_disk_size`: 50GB is usually sufficient

4. **Use Lifecycle Rules**:
   - Set `gcs_bucket_lifecycle_days` to auto-delete old data

## Troubleshooting

### VM Creation Fails

```bash
# Check quotas
gcloud compute project-info describe --project=YOUR-PROJECT

# Check available machine types in zone
gcloud compute machine-types list --zones=europe-west2-b --filter="name:n2-standard"
```

### Disk Not Mounting

```bash
# List attached disks
lsblk

# Check disk status
sudo fdisk -l

# Manual mount
sudo mount /dev/nvme0n2 /mnt/python-env
```

### Out of Disk Space

```bash
# Check usage
df -h /mnt/training-data

# Increase temp disk size in terraform.tfvars
temp_disk_size = 750  # Increase from 500

# Apply changes
terraform apply
```

### Python Packages Missing

```bash
# Reactivate environment
source ~/python-env/activate-tf.sh

# Reinstall packages
pip install google-cloud-storage netcdf4 xarray
```

## Advanced Configuration

### Use Spot VMs (Even Cheaper)

```bash
# In terraform.tfvars
preemptible = true
```

### Custom Network

```hcl
# In terraform.tfvars
network = "projects/YOUR-PROJECT/global/networks/custom-network"
```

### Multiple Processing VMs

```bash
# Create separate terraform.tfvars files
cp terraform.tfvars tfrecords-vm-1.tfvars
cp terraform.tfvars tfrecords-vm-2.tfvars

# Apply with specific var file
terraform apply -var-file=tfrecords-vm-1.tfvars
```

## Comparison: CPU VM vs GPU VM

| Feature | CPU VM (this setup) | GPU VM (tf_gpu/) |
|---------|-------------------|------------------|
| **Purpose** | TFRecords creation | Model training |
| **Machine Type** | n2-standard-8 | g2-standard-8 |
| **Compute Cost** | ~$0.40/hour | ~$2.40/hour |
| **GPU** | None | NVIDIA L4 |
| **Session Cost (8h)** | ~$6-7 | ~$22-25 |
| **Best For** | Data preprocessing | Deep learning |
| **Preemptible Option** | Yes | Limited |

**Recommendation**: Use this CPU VM setup for TFRecords creation, then use the GPU VM (`tf_gpu/`) only for actual model training.

---

## Quick Command Reference

```bash
# Deploy infrastructure
cd tf_vm_tfrecords/ && terraform apply

# SSH to VM
gcloud compute ssh tfrecords-vm --zone=europe-west2-b

# Mount disks (subsequent sessions)
sudo mkdir -p /mnt/python-env /mnt/training-data
sudo mount /dev/nvme0n2 /mnt/python-env
sudo mkfs.ext4 /dev/nvme0n3 && sudo mount /dev/nvme0n3 /mnt/training-data

# Activate environment
source ~/python-env/activate-tf.sh

# Download data
python gcs_folder_download.py gs://BUCKET/path --creds key.json --dest /mnt/training-data/

# Create TFRecords
cd ~/cGAN_tutorial
python -c "from data import write_data; write_data(2018)"

# Upload results
gsutil -m cp -r /mnt/training-data/tfrecords/* gs://BUCKET/tfrecords/

# Cleanup (safe - preserves Python environment)
exit
terraform destroy -target=google_compute_instance.tfrecords_vm

# Restart next time
terraform apply
```

---

**Related Documentation:**
- Complete workflow guide: `../docs/tfrecords_creation_workflow.md`
- Python environment setup: `../tf_gpu/tensorflow-env-data-setup/README.md`
- GPU training setup: `../tf_gpu/README.md`

**Repository**: https://github.com/snath-xoc/cGAN_tutorial
