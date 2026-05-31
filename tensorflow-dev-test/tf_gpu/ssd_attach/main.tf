provider "google" {
  credentials = file("../../key-nka-terraform-access.json")
  project     = "sewaa-416306"
  region      = "europe-west2"
}

resource "google_compute_attached_disk" "attach_ssd" {
  disk     = "t1-cgan-ssd-disk-large"
  instance = "tfrecords-cgan-store-nka-t2"
  zone     = "europe-west2-b"
}
