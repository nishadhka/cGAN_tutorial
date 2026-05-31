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
  description = "Name of the GPU VM instance"
  type        = string
  default     = "cgan-training-vm"
}

variable "machine_type" {
  description = "Machine type for the GPU VM"
  type        = string
  default     = "g2-standard-4"
}

variable "boot_image" {
  description = "Boot disk image for deep learning"
  type        = string
  default     = "projects/deeplearning-platform-release/global/images/tf-ent-2-15-cu121-v20240417-debian-11-py310"
}

# Storage Configuration
variable "persistent_disk_name" {
  description = "Name of the persistent SSD for Python environment"
  type        = string
  default     = "cgan-python-env-disk"
}

variable "persistent_disk_size" {
  description = "Size of the persistent disk in GB"
  type        = number
  default     = 50
}

variable "temp_disk_name" {
  description = "Name of the temporary SSD for training data"
  type        = string
  default     = "cgan-training-data-disk"
}

variable "temp_disk_size" {
  description = "Size of the temporary training data disk in GB"
  type        = number
  default     = 500
}

variable "disk_type" {
  description = "Type of disk (pd-ssd for SSD)"
  type        = string
  default     = "pd-ssd"
}

# GCS Configuration
variable "training_bucket_name" {
  description = "Name of the GCS bucket containing training data"
  type        = string
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
  default     = "training"
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default = {
    purpose = "cgan-training"
    managed-by = "terraform"
  }
}
