terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  credentials = file(var.credentials_file)
  project     = var.project_id
  region      = var.region
}

# Reference existing persistent SSD for Python environment (not creating a new one)
data "google_compute_disk" "persistent_python_env" {
  name = var.persistent_disk_name
  zone = var.zone
}

# Temporary SSD for training data (to be deleted after TFRecords creation)
resource "google_compute_disk" "temp_training_data" {
  name  = var.temp_disk_name
  type  = var.disk_type
  zone  = var.zone
  size  = var.temp_disk_size

  labels = merge(var.labels, {
    disk-purpose = "tfrecords-data"
    temporary    = "true"
  })
}

# CPU VM Instance for TFRecords Creation
resource "google_compute_instance" "tfrecords_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  labels = var.labels

  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = 50  # Boot disk size
      type  = "pd-standard"  # Standard disk for boot is sufficient
    }
  }

  # Attach persistent Python environment disk (existing disk)
  attached_disk {
    source      = data.google_compute_disk.persistent_python_env.id
    device_name = "python-env"
    mode        = "READ_WRITE"
  }

  # Attach temporary training data disk
  attached_disk {
    source      = google_compute_disk.temp_training_data.id
    device_name = "training-data"
    mode        = "READ_WRITE"
  }

  # No GPU configuration for CPU-only VM

  scheduling {
    on_host_maintenance = "MIGRATE"  # Can use MIGRATE for CPU instances
    automatic_restart   = true
    preemptible        = var.preemptible  # Can set to true for cost savings
  }

  network_interface {
    network = var.network
    access_config {
      # Ephemeral IP
    }
  }

  metadata = {
    enable-oslogin = "TRUE"
  }

  # Enhanced service account permissions for GCS access
  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  tags = [var.environment, "cpu-processing", "tfrecords"]
}

# Optional: GCS bucket for data storage (if needed)
resource "google_storage_bucket" "training_data_bucket" {
  count    = var.create_gcs_bucket ? 1 : 0
  name     = var.training_bucket_name
  location = var.gcs_bucket_location

  uniform_bucket_level_access = true

  labels = var.labels

  lifecycle_rule {
    condition {
      age = var.gcs_bucket_lifecycle_days
    }
    action {
      type = "Delete"
    }
  }
}
