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

# Persistent SSD for Python environment (not to be deleted)
resource "google_compute_disk" "persistent_python_env" {
  name  = var.persistent_disk_name
  type  = var.disk_type
  zone  = var.zone
  size  = var.persistent_disk_size

  labels = merge(var.labels, {
    disk-purpose = "python-environment"
    persistent   = "true"
  })

  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      labels,
      zone
    ]
  }
}

# Temporary SSD for training data (to be deleted after training)
resource "google_compute_disk" "temp_training_data" {
  name  = var.temp_disk_name
  type  = var.disk_type
  zone  = var.zone
  size  = var.temp_disk_size

  labels = merge(var.labels, {
    disk-purpose = "training-data"
    temporary    = "true"
  })
}

# GPU VM Instance
resource "google_compute_instance" "cgan_training_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  labels = var.labels

  boot_disk {
    initialize_params {
      image = var.boot_image
      size  = 50  # Minimum required for deep learning image
      type  = "pd-standard"  # Standard disk for boot is sufficient
    }
  }

  # Attach persistent Python environment disk
  attached_disk {
    source      = google_compute_disk.persistent_python_env.id
    device_name = "python-env"
    mode        = "READ_WRITE"
  }

  # Attach temporary training data disk
  attached_disk {
    source      = google_compute_disk.temp_training_data.id
    device_name = "training-data"
    mode        = "READ_WRITE"
  }

  # GPU configuration
  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible        = false  # Set to true for cost savings if acceptable
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

  tags = [var.environment, "gpu-training", "cgan"]
}
