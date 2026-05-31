output "vm_instance_name" {
  description = "Name of the created VM instance"
  value       = google_compute_instance.cgan_training_vm.name
}

output "vm_external_ip" {
  description = "External IP address of the VM"
  value       = google_compute_instance.cgan_training_vm.network_interface[0].access_config[0].nat_ip
}

output "vm_internal_ip" {
  description = "Internal IP address of the VM"
  value       = google_compute_instance.cgan_training_vm.network_interface[0].network_ip
}

output "persistent_disk_name" {
  description = "Name of the persistent Python environment disk"
  value       = google_compute_disk.persistent_python_env.name
}

output "temp_disk_name" {
  description = "Name of the temporary training data disk"
  value       = google_compute_disk.temp_training_data.name
}

output "ssh_command" {
  description = "Command to SSH into the VM"
  value       = "gcloud compute ssh ${google_compute_instance.cgan_training_vm.name} --zone=${var.zone} --project=${var.project_id}"
}

output "cost_estimate_per_hour" {
  description = "Estimated cost per hour (USD) for running resources"
  value       = "~$2.50 (VM: $2.40, Storage: $0.10)"
}

output "mount_points" {
  description = "Disk mount points on the VM"
  value = {
    python_env    = "/mnt/python-env"
    training_data = "/mnt/training-data"
  }
}