# Project Configuration
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "europe-west2"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "europe-west2-b"
}

variable "credentials_file" {
  description = "Path to the service account key file"
  type        = string
}

# VM Configuration
variable "vm_name" {
  description = "Name of the CPU VM instance for TFRecords creation"
  type        = string
  default     = "tfrecords-vm"
}

variable "machine_type" {
  description = "Machine type for the CPU VM (e.g., n1-standard-8, n2-standard-16)"
  type        = string
  default     = "n2-standard-8"  # 8 vCPUs, 32 GB memory
}

variable "boot_image" {
  description = "Boot disk image for the VM"
  type        = string
  default     = "projects/debian-cloud/global/images/family/debian-11"
}

variable "preemptible" {
  description = "Whether to use a preemptible VM instance for cost savings"
  type        = bool
  default     = false
}

# Storage Configuration
variable "persistent_disk_name" {
  description = "Name of the persistent SSD for Python environment"
  type        = string
  default     = "tfrecords-python-env-disk"
}

variable "persistent_disk_size" {
  description = "Size of the persistent disk in GB"
  type        = number
  default     = 50
}

variable "temp_disk_name" {
  description = "Name of the temporary SSD for training data"
  type        = string
  default     = "tfrecords-data-disk"
}

variable "temp_disk_size" {
  description = "Size of the temporary training data disk in GB"
  type        = number
  default     = 500
}

variable "disk_type" {
  description = "Type of disk (pd-ssd for SSD, pd-standard for HDD)"
  type        = string
  default     = "pd-ssd"
}

# GCS Configuration
variable "training_bucket_name" {
  description = "Name of the GCS bucket for storing training data (must be globally unique)"
  type        = string
}

variable "create_gcs_bucket" {
  description = "Whether to create a new GCS bucket (set to false if bucket already exists)"
  type        = bool
  default     = false
}

variable "gcs_bucket_location" {
  description = "Location for the GCS bucket (e.g., EU, US, ASIA)"
  type        = string
  default     = "EU"
}

variable "gcs_bucket_lifecycle_days" {
  description = "Number of days before objects are deleted (0 to disable)"
  type        = number
  default     = 0  # Disabled by default
}

# Network Configuration
variable "network" {
  description = "Network for the VM"
  type        = string
  default     = "default"
}

# Tags and Labels
variable "environment" {
  description = "Environment tag (dev, staging, prod)"
  type        = string
  default     = "tfrecords-processing"
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default = {
    purpose    = "tfrecords-creation"
    managed-by = "terraform"
  }
}
