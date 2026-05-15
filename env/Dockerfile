# syntax=docker/dockerfile:1.6
#
# Two-framework GPU image for cGAN training:
#   /opt/conda/envs/tf     -> TensorFlow 2.15  (cGAN_tutorial)
#   /opt/conda/envs/torch  -> PyTorch 2.1 + Lightning 1.9 (CGAN_extreme_precipitation)
#
# Build:   docker build -t cgan-gpu:latest .
# Run:     docker run --gpus all --rm -it -p 8888:8888 \
#                     -v /scratch/notebook:/workspace cgan-gpu:latest
# Or:      docker compose up
#
# Switch envs at runtime:
#   pytf <script.py>      # TensorFlow env
#   pytorch <script.py>   # PyTorch env

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG MAMBA_VERSION=1.5.8
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/condabin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Build hosts (Cloud Build, plain CI runners) have no GPU, so conda-forge's
    # __cuda virtual package is absent and `tensorflow=...=cuda120*` /
    # `pytorch=...=cuda120*` builds become unsolvable. This override tells
    # the solver to pretend a CUDA 12.2 runtime is available, matching the
    # base image's NVIDIA libs. The image still runs fine on real GPU hosts.
    CONDA_OVERRIDE_CUDA=12.2

# Minimal apt deps. tini for clean PID 1; git for pip-from-git installs;
# bzip2/curl for the micromamba install.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 tini git \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (single static binary, no Python required)
RUN mkdir -p /opt/conda && \
    curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION}" \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Solve and create both envs. Keeping them in separate `RUN` layers so a
# failed solve in one doesn't invalidate the other on rebuild.
COPY env-tf.yaml /tmp/env-tf.yaml
RUN micromamba create -y -n tf -f /tmp/env-tf.yaml && \
    micromamba clean -a -y && rm /tmp/env-tf.yaml

COPY env-torch.yaml /tmp/env-torch.yaml
RUN micromamba create -y -n torch -f /tmp/env-torch.yaml && \
    micromamba clean -a -y && rm /tmp/env-torch.yaml

# Use the torch env as the host for Jupyter, and register both as kernels
# so a single `jupyter lab` exposes both frameworks side-by-side.
RUN /opt/conda/envs/tf/bin/python    -m ipykernel install \
        --prefix=/opt/conda/envs/torch \
        --name=tf    --display-name="cGAN (TF 2.15)" && \
    /opt/conda/envs/torch/bin/python -m ipykernel install \
        --prefix=/opt/conda/envs/torch \
        --name=torch --display-name="cGAN (Torch 2.1)" && \
    /opt/conda/envs/torch/bin/pip install --no-cache-dir jupyterlab

# Convenience launchers
RUN ln -s /opt/conda/envs/tf/bin/python    /usr/local/bin/pytf && \
    ln -s /opt/conda/envs/torch/bin/python /usr/local/bin/pytorch

# Sanity test on build: import both frameworks (CPU-only at build time)
RUN /opt/conda/envs/tf/bin/python    -c "import tensorflow as tf; print('TF', tf.__version__)" && \
    /opt/conda/envs/torch/bin/python -c "import torch; print('Torch', torch.__version__)"

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/conda/envs/torch/bin/jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--allow-root"]
