# cGAN Training Infrastructure

This Terraform configuration creates a cost-optimized Google Cloud Platform infrastructure for cGAN training with dual SSD setup.

## Architecture Overview

- **GPU VM**: G2-standard-8 with NVIDIA L4 GPU
- **Persistent SSD**: 50GB for Python environment (preserved across training sessions)
- **Temporary SSD**: 500GB for training data (deleted after training)
- **GCS Integration**: Multi-region bucket for training datasets

## Cost Optimization Strategy

### Dual SSD Approach
1. **Persistent Disk (50GB)**: Stores Python environment, libraries, and configurations
   - Preserved between training sessions
   - Prevents reinstallation overhead
   - Lifecycle protection enabled

2. **Temporary Disk (500GB)**: Stores training datasets during training
   - Created only when needed
   - Deleted immediately after training
   - Reduces storage costs significantly

### Economic Benefits
- **Storage Cost Savings**: ~85% reduction by deleting 500GB temporary disk after training
- **Setup Time Reduction**: Persistent Python environment eliminates reinstallation
- **Data Transfer Optimization**: GCS to local SSD transfer for faster training

## Prerequisites

1. **GCP Project** with billing enabled
2. **Service Account** with required permissions:
   - Compute Engine Admin
   - Storage Admin
   - Service Account User
3. **APIs Enabled**:
   - Compute Engine API
   - Cloud Storage API
4. **GCS Bucket** with training datasets (NetCDF files)

## Setup Instructions

### 1. Configure Variables
```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 2. Initialize Terraform
```bash
terraform init
```

### 3. Plan Infrastructure
```bash
terraform plan
```

### 4. Deploy Infrastructure
```bash
terraform apply
```

### 5. Setup Disks (First Time Only)
After VM creation, SSH into the VM and setup the disks:

```bash
# SSH to VM
gcloud compute ssh [VM_NAME] --zone=[ZONE]

# Check attached disks
lsblk
# You should see:
# nvme0n1 (50G) - Boot disk (already mounted at /)
# nvme0n2 (50G) - Persistent Python environment disk
# nvme0n3 (500G) - Temporary training data disk

# Create mount points
sudo mkdir -p /mnt/python-env
sudo mkdir -p /mnt/training-data

# ONE-TIME SETUP: Format persistent Python environment disk (50GB)
# WARNING: Only do this on first setup - this will erase data!
sudo mkfs.ext4 /dev/nvme0n2

# Mount persistent disk
sudo mount /dev/nvme0n2 python-env
sudo chown -R $USER:$USER /mnt/python-env

# Add to fstab for persistent mounting
echo '/dev/nvme0n2 /mnt/python-env ext4 defaults 0 2' | sudo tee -a /etc/fstab

# EVERY SESSION: Format temporary training data disk (500GB)
sudo mkfs.ext4 /dev/nvme0n3
sudo mount /dev/nvme0n3 training-data
sudo chown -R $USER:$USER training-data

# Create symbolic links for easy access
ln -sf /mnt/python-env ~/python-env
ln -sf /mnt/training-data ~/training-data

# Install Python packages to persistent disk (first time only)
if [ ! -f /mnt/python-env/.setup_complete ]; then
    echo "Setting up Python environment on persistent disk..."
    mkdir -p /mnt/python-env/lib/python3.10/site-packages
    pip install --target /mnt/python-env/lib/python3.10/site-packages \
        tensorflow matplotlib netcdf4 xarray pandas numpy scikit-learn

    # Mark setup as complete
    touch /mnt/python-env/.setup_complete

    # Add to bashrc for permanent Python path
    echo 'export PYTHONPATH="/mnt/python-env/lib/python3.10/site-packages:$PYTHONPATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Download training data from GCS
gsutil -m cp -r gs://[YOUR_BUCKET_NAME]/* /mnt/training-data/

echo "Disk setup completed!"
echo "Python environment: /mnt/python-env (persistent)"
echo "Training data: /mnt/training-data (temporary)"
df -h /mnt/python-env /mnt/training-data
```

## Training Workflow

### 1. First Training Session
```bash
# Deploy infrastructure
terraform apply

# SSH to VM and setup disks (see step 5 above)
gcloud compute ssh [VM_NAME] --zone=[ZONE]
# Follow the disk setup commands above

# Training data available at: /mnt/training-data
# Python environment at: /mnt/python-env
```

### 1b. Subsequent Training Sessions
```bash
# Deploy infrastructure (persistent disk will reattach automatically)
terraform apply

# SSH to VM
gcloud compute ssh [VM_NAME] --zone=[ZONE]

# Quick setup for subsequent sessions
sudo mkdir -p python-env training-data

# Mount persistent Python environment (already formatted)
sudo mount /dev/nvme0n2 python-env
sudo chown -R $USER:$USER python-env

# Format and mount temporary training data disk (fresh each time)
sudo mkfs.ext4 /dev/nvme0n3
sudo mount /dev/nvme0n3 training-data
sudo chown -R $USER:$USER training-data

# Recreate symbolic links
#ln -sf /mnt/python-env ~/python-env
#ln -sf /mnt/training-data ~/training-data
```

### 2. During Training
- Training datasets are downloaded from GCS to `/mnt/training-data`
- Python environment is mounted from `/mnt/python-env`
- GPU acceleration available for TensorFlow/PyTorch

### 3. Complete Training Session
```bash
# Save results to GCS or download locally
gsutil cp /path/to/results gs://your-results-bucket/

# Destroy temporary resources
terraform destroy
```

## Resource Management

### Persistent Resources (Preserved)
- Python environment disk (50GB)
- GCS buckets
- Service accounts

### Temporary Resources (Deleted)
- GPU VM instance
- Training data disk (500GB)
- Ephemeral IP addresses

## Manual Disk Management

### Recommended: Delete Only Temporary Resources
```bash
# SAFE: Remove only VM and temporary 500GB disk, preserve 50GB Python environment
terraform destroy -target=google_compute_instance.cgan_training_vm
terraform destroy -target=google_compute_disk.temp_training_data
```

### Restart Training Session
```bash
# The 50GB persistent disk will automatically reattach
terraform apply
```

### ⚠️ DANGER: Delete All Resources
```bash
# This will delete EVERYTHING including the 50GB persistent disk
# Only use when completely done with the project
terraform destroy
```

### Accidental Protection
- The 50GB persistent disk has `prevent_destroy = true`
- `terraform destroy` will fail unless you use `-force`
- Always use targeted destroys for normal cleanup

## Cost Estimates (Europe-West2)

### Training Session (8 hours)
- GPU VM (g2-standard-8): ~$2.40/hour = $19.20
- Temporary SSD (500GB): ~$0.170/GB/month = $2.83/day
- Persistent SSD (50GB): ~$0.170/GB/month = $0.28/day
- **Total per session**: ~$22-25

### Monthly Storage (No Training)
- Persistent SSD (50GB): ~$8.50/month
- GCS Storage (500GB, multi-region): ~$10/month
- **Storage only**: ~$18.50/month

## Security Best Practices

1. **Service Account Keys**: Store securely, never commit to version control
2. **Network Security**: Default VPC with minimal required access
3. **Disk Encryption**: Google-managed encryption by default
4. **Access Control**: OS Login enabled for SSH access

## Troubleshooting

### VM Creation Issues
```bash
# Check quotas
gcloud compute project-info describe --project=[PROJECT_ID]

# Verify GPU availability
gcloud compute accelerator-types list --zones=[ZONE]
```

### Disk Mounting Issues
```bash
# Check disk attachment
lsblk

# Manual mount
sudo mkdir -p /mnt/training-data
sudo mount /dev/sdb /mnt/training-data
```

### GCS Access Issues
```bash
# Test GCS access
gsutil ls gs://[BUCKET_NAME]

# Check service account permissions
gcloud projects get-iam-policy [PROJECT_ID]
```

## Monitoring and Alerts

### Cost Monitoring
- Set up billing alerts for unexpected costs
- Monitor disk usage: `df -h`
- Track GPU utilization: `nvidia-smi`

### Training Progress
- Use TensorBoard for training visualization
- Log training metrics to Cloud Logging
- Set up Cloud Monitoring for GPU metrics

## Advanced Configurations

### Preemptible Instances
```hcl
# In main.tf, set preemptible = true for 60-90% cost savings
scheduling {
  preemptible = true
}
```

### Auto-scaling Training
- Use Cloud Functions to trigger training jobs
- Implement job queues with Cloud Tasks
- Auto-cleanup with scheduled Cloud Functions

## Support and Maintenance

### Regular Tasks
1. **Weekly**: Review and optimize disk usage
2. **Monthly**: Audit GCS storage costs
3. **Quarterly**: Update base images and dependencies

### Backup Strategy
- Python environment: Snapshot persistent disk monthly
- Training code: Version control in Git
- Results: Regular backup to GCS

---

## Quick Commands Reference

```bash
# Deploy infrastructure
terraform apply

# SSH to training VM
gcloud compute ssh cgan-training-vm --zone=europe-west2-b

# Monitor GPU usage
nvidia-smi -l 1

# Check disk space
df -h /mnt/training-data /mnt/python-env

# Download training data
gsutil -m cp -r gs://[BUCKET]/* /mnt/training-data/

# SAFE cleanup after training (preserves 50GB Python environment)
terraform destroy -target=google_compute_instance.cgan_training_vm
terraform destroy -target=google_compute_disk.temp_training_data

# Next training session - disk will reattach automatically
terraform apply
```
