"""
Register the cGAN TF+Torch image as a Coiled software environment.

Run once after `gcloud builds submit` has pushed the image to Artifact Registry:

    pip install coiled
    python coiled_register.py

Coiled will pull the container at cluster spin-up. The image must be readable
by the Coiled GCP service account; usually granting `roles/artifactregistry.reader`
on the `coiled` repo to Coiled's SA is enough.
"""
import coiled

coiled.create_software_environment(
    name="cgan-tf-torch-v1",
    workspace="geosfm",  # change to your Coiled workspace if different
    container="us-east1-docker.pkg.dev/sewaa-416306/coiled/cgan-tf-torch:v1.0",
)
