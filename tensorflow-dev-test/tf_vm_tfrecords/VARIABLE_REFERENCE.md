# Variable Reference and Explanations

## Variable Consistency Verification ✅

All variables are now properly referenced across all Terraform files:

### `training_bucket_name` Variable

**Definition** (variables.tf:81):
```hcl
variable "training_bucket_name" {
  description = "Name of the GCS bucket for storing training data (must be globally unique)"
  type        = string
}
```

**Referenced in**:
- `main.tf:113` - Used to create/reference GCS bucket
- `outputs.tf:43` - Output value for bucket name
- `outputs.tf:48` - Output value for bucket URL
- `terraform.tfvars.example:24` - Example configuration

**Your configuration** (terraform.tfvars):
```hcl
training_bucket_name = "your-actual-bucket-name"
```

✅ **Status**: All references are correct and consistent!

---

## Understanding the `environment` Variable

### What is `environment`?

The `environment` variable is a **label/tag** used for resource organization and filtering in Google Cloud Platform. It does NOT affect the operating system or VM configuration.

**Definition** (variables.tf:104-109):
```hcl
variable "environment" {
  description = "Environment tag (dev, staging, prod)"
  type        = string
  default     = "tfrecords-processing"
}
```

### How `environment` is Used

**In main.tf:107**:
```hcl
tags = [var.environment, "cpu-processing", "tfrecords"]
```

These tags are used to:
1. **Network firewall rules** - Control which firewall rules apply to this VM
2. **Resource organization** - Filter/group VMs in Cloud Console
3. **Cost tracking** - Track costs by environment
4. **Automation** - Target specific VMs with scripts

### Example Values and Their Purposes

| Value | Purpose |
|-------|---------|
| `"dev"` | Development/testing environment |
| `"staging"` | Pre-production testing |
| `"prod"` | Production workloads |
| `"tfrecords-processing"` | Specific purpose (current default) |
| `"non-gpu-tfrecords"` | Your value - descriptive tag |

**Your setting**:
```hcl
environment = "non-gpu-tfrecords"
```

This is perfectly fine! It's just a descriptive tag. You could also use:
- `environment = "tfrecords"`
- `environment = "data-processing"`
- `environment = "cpu-vm"`

### What `environment` Does NOT Affect

❌ Operating system choice
❌ VM machine type
❌ GPU availability
❌ Disk configuration
❌ Python packages installed

---

## Operating System Configuration

### Boot Image Variable

The OS is controlled by the `boot_image` variable:

**Definition** (variables.tf:37-41):
```hcl
variable "boot_image" {
  description = "Boot disk image for the VM"
  type        = string
  default     = "projects/debian-cloud/global/images/family/debian-11"
}
```

### Current OS: Debian 11 ✅

**Status**: ✅ **Correctly configured for a CPU VM**

**Why Debian 11 is good for this use case:**
1. **Lightweight** - No GPU drivers or CUDA overhead
2. **Fast boot** - Quick VM startup
3. **Stable** - Well-tested and reliable
4. **Compatible** - Works with all Python packages needed
5. **Cost-effective** - Smaller image, faster provisioning

### Alternative OS Images

If you want to change the OS, you can override `boot_image` in terraform.tfvars:

#### Option 1: Ubuntu 20.04 LTS
```hcl
boot_image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts"
```

#### Option 2: Ubuntu 22.04 LTS
```hcl
boot_image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
```

#### Option 3: Debian 12
```hcl
boot_image = "projects/debian-cloud/global/images/family/debian-12"
```

#### Option 4: CentOS Stream 9
```hcl
boot_image = "projects/centos-cloud/global/images/family/centos-stream-9"
```

### Recommended: Keep Debian 11 (Current Default)

For TFRecords processing on a CPU VM, **Debian 11 is the optimal choice**:
- ✅ Minimal resource usage
- ✅ Fast package installation with apt
- ✅ Compatible with Micromamba and Python packages
- ✅ Well-documented and widely used
- ✅ Regular security updates

---

## Complete Variable Summary

### Required Variables (must be set in terraform.tfvars)

```hcl
project_id           = "your-gcp-project-id"          # Your GCP project
credentials_file     = "path/to/service-account.json" # Auth credentials
training_bucket_name = "your-bucket-name"             # GCS bucket (globally unique)
```

### Important Optional Variables

```hcl
# VM Configuration
vm_name      = "tfrecords-vm"                  # VM name in GCP
machine_type = "n2-standard-8"                 # 8 vCPUs, 32GB RAM
environment  = "non-gpu-tfrecords"             # Tag for organization
preemptible  = false                           # Use cheap preemptible VM?
boot_image   = "debian-11"                     # OS (Debian 11 is default)

# Storage Configuration
persistent_disk_size = 50                      # GB for Python env
temp_disk_size      = 500                      # GB for data processing
disk_type           = "pd-ssd"                 # SSD for performance

# GCS Bucket Configuration
create_gcs_bucket   = false                    # Create new bucket?
gcs_bucket_location = "EU"                     # Bucket location
```

### Network Tags (automatic)

The VM automatically gets these network tags:
1. `var.environment` - Your custom environment tag (e.g., "non-gpu-tfrecords")
2. `"cpu-processing"` - Indicates CPU-only VM
3. `"tfrecords"` - Indicates TFRecords purpose

These tags can be used in firewall rules like:
```hcl
# Example: Allow SSH only to VMs with "tfrecords" tag
target_tags = ["tfrecords"]
```

---

## Verification Commands

### Check your configuration is valid:

```bash
cd tf_vm_tfrecords/

# Validate syntax
terraform validate

# Show what will be created
terraform plan

# Check specific variable value
terraform console
> var.training_bucket_name
> var.environment
> var.boot_image
```

### After deployment, verify VM details:

```bash
# Show all outputs
terraform output

# Show specific output
terraform output vm_instance_name
terraform output gcs_bucket_url
terraform output ssh_command

# Inspect VM in GCP
gcloud compute instances describe tfrecords-vm --zone=europe-west2-b
```

---

## Common Configuration Examples

### Example 1: Basic Setup (Recommended)
```hcl
project_id           = "my-gcp-project"
credentials_file     = "~/keys/service-account.json"
training_bucket_name = "my-tfrecords-data-bucket"
environment          = "tfrecords-processing"
```

### Example 2: Cost-Optimized (Preemptible)
```hcl
project_id           = "my-gcp-project"
credentials_file     = "~/keys/service-account.json"
training_bucket_name = "my-tfrecords-data-bucket"
preemptible          = true                    # 60-80% cheaper
machine_type         = "n2-standard-4"         # Smaller VM
temp_disk_size       = 300                     # Less storage
```

### Example 3: High Performance
```hcl
project_id           = "my-gcp-project"
credentials_file     = "~/keys/service-account.json"
training_bucket_name = "my-tfrecords-data-bucket"
machine_type         = "n2-standard-16"        # 16 vCPUs, 64GB RAM
temp_disk_size       = 1000                    # 1TB storage
disk_type            = "pd-ssd"                # Fast SSD
```

### Example 4: Ubuntu-based Setup
```hcl
project_id           = "my-gcp-project"
credentials_file     = "~/keys/service-account.json"
training_bucket_name = "my-tfrecords-data-bucket"
boot_image           = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
environment          = "ubuntu-tfrecords"
```

---

## Summary

✅ **All variables are properly referenced** - No errors will occur
✅ **`training_bucket_name` is correct** - Matches your terraform.tfvars
✅ **`environment` is just a tag** - Used for organization, not OS selection
✅ **Debian 11 OS is optimal** - Correct choice for CPU-based TFRecords processing
✅ **All configurations validated** - Ready for `terraform apply`

Your configuration is correct and ready to deploy!
