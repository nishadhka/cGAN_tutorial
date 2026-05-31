output "vm_instance_name" {
  description = "Name of the created VM instance"
  value       = google_compute_instance.tfrecords_vm.name
}

output "vm_external_ip" {
  description = "External IP address of the VM"
  value       = google_compute_instance.tfrecords_vm.network_interface[0].access_config[0].nat_ip
}

output "vm_internal_ip" {
  description = "Internal IP address of the VM"
  value       = google_compute_instance.tfrecords_vm.network_interface[0].network_ip
}

output "vm_zone" {
  description = "Zone where the VM is located"
  value       = google_compute_instance.tfrecords_vm.zone
}

output "persistent_disk_name" {
  description = "Name of the persistent Python environment disk (existing)"
  value       = data.google_compute_disk.persistent_python_env.name
}

output "persistent_disk_size" {
  description = "Size of the persistent disk in GB"
  value       = data.google_compute_disk.persistent_python_env.size
}

output "temp_disk_name" {
  description = "Name of the temporary training data disk"
  value       = google_compute_disk.temp_training_data.name
}

output "temp_disk_size" {
  description = "Size of the temporary disk in GB"
  value       = google_compute_disk.temp_training_data.size
}

output "gcs_bucket_name" {
  description = "Name of the GCS bucket for training data"
  value       = var.training_bucket_name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${var.training_bucket_name}"
}

output "ssh_command" {
  description = "Command to SSH into the VM"
  value       = "gcloud compute ssh ${google_compute_instance.tfrecords_vm.name} --zone=${var.zone} --project=${var.project_id}"
}

output "cost_estimate_per_hour" {
  description = "Estimated cost per hour (USD) for running resources"
  value       = "~$0.50-0.80 (VM: $0.40-0.70, Storage: $0.10)"
}

output "mount_points" {
  description = "Disk mount points on the VM"
  value = {
    python_env    = "/mnt/python-env"
    training_data = "/mnt/training-data"
  }
}

output "setup_commands" {
  description = "Quick reference setup commands"
  value = <<-EOT
    # SSH into VM:
    ${google_compute_instance.tfrecords_vm.name} --zone=${var.zone} --project=${var.project_id}

    # Mount disks (first time):
    sudo mkdir -p /mnt/python-env /mnt/training-data
    sudo mkfs.ext4 /dev/nvme0n2  # Only first time!
    sudo mount /dev/nvme0n2 /mnt/python-env
    sudo mkfs.ext4 /dev/nvme0n3
    sudo mount /dev/nvme0n3 /mnt/training-data

    # Mount disks (subsequent times):
    sudo mkdir -p /mnt/python-env /mnt/training-data
    sudo mount /dev/nvme0n2 /mnt/python-env
    sudo mkfs.ext4 /dev/nvme0n3 && sudo mount /dev/nvme0n3 /mnt/training-data
  EOT
}
