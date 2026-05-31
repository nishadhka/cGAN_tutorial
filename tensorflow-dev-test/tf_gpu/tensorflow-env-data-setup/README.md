# TensorFlow Environment Setup with Micromamba

This guide details the setup of a persistent Python environment using Micromamba on the 50GB persistent SSD disk for cGAN training.

## Overview

The setup creates a persistent Python environment that survives VM destruction/recreation cycles:
- **Location**: `~/python-env/` (mounted from 50GB persistent SSD)
- **Micromamba installation**: `~/python-env/.local/bin/`
- **Environment storage**: `~/python-env/micromamba/`
- **Python version**: 3.11
- **TensorFlow version**: 2.15 (GPU-enabled)

## Prerequisites

1. VM created with Terraform (50GB persistent disk attached as `/dev/nvme0n2`)
2. Disk mounted at `/mnt/python-env` or `~/python-env`
3. SSH access to the GPU VM

## Step-by-Step Setup

### 1. Prepare Persistent Disk Mount
```bash
# Create mount points and mount persistent disk
sudo mkdir -p /mnt/python-env
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env

# Create symbolic link for easy access
ln -sf /mnt/python-env ~/python-env
```

### 2. Install Micromamba to Persistent Disk
```bash
# Install micromamba to persistent location
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# During installation, specify these paths:
# - Micromamba binary location: ~/python-env/.local/bin
# - Environment location: ~/python-env/micromamba

# Reload shell configuration
source ~/.bashrc

# Update micromamba
micromamba self-update -c conda-forge
```

### 3. Setup PATH for Persistent Access
```bash
# Add micromamba to PATH permanently
echo 'export PATH="/home/nkalladath_icpac_net/python-env/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
micromamba --version
```

### 4. Create TensorFlow GPU Environment
```bash
# Create Python 3.11 environment for TensorFlow 2.15
micromamba create -n tf215gpu python=3.11
micromamba activate tf215gpu

# Install TensorFlow 2.15 with GPU support
python -m pip install tensorflow==2.15
```

### 5. Install Required Data Science Libraries
```bash
# Core scientific libraries
pip install numba
pip install matplotlib
pip install seaborn
pip install numpy
pip install pandas
pip install scipy

# Geospatial and climate data libraries
pip install cartopy
pip install xarray
pip install netcdf4
pip install cfgrib
pip install iris
pip install regionmask
pip install xesmf

# Machine learning libraries
pip install scikit-learn
pip install dask

# Development and utility libraries
pip install jupyter
pip install tqdm
pip install flake8
pip install schedule

# Climate modeling libraries
pip install properscoring
pip install climlab
pip install ecmwf-api-client
```

### 6. Verify GPU Access
```bash
# Test TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Test CUDA version compatibility
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## Complete Installation Script

Here's the complete setup script for first-time environment creation:

```bash
#!/bin/bash
# tensorflow-env-setup.sh

set -e

echo "Setting up persistent TensorFlow environment with Micromamba..."

# 1. Mount persistent disk
sudo mkdir -p /mnt/python-env
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env
ln -sf /mnt/python-env ~/python-env

# 2. Install Micromamba (if not already installed)
if [ ! -f ~/python-env/.local/bin/micromamba ]; then
    echo "Installing Micromamba..."
    # Set environment variables for automated installation
    export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
    export MAMBA_EXE="$HOME/python-env/.local/bin/micromamba"

    # Download and install
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh) -b -p "$HOME/python-env/.local"

    # Add to PATH
    echo 'export PATH="$HOME/python-env/.local/bin:$PATH"' >> ~/.bashrc
    echo 'export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"' >> ~/.bashrc
    source ~/.bashrc
fi

# 3. Update micromamba
$HOME/python-env/.local/bin/micromamba self-update -c conda-forge

# 4. Create TensorFlow environment (if doesn't exist)
if ! $HOME/python-env/.local/bin/micromamba env list | grep -q tf215gpu; then
    echo "Creating tf215gpu environment..."
    $HOME/python-env/.local/bin/micromamba create -n tf215gpu python=3.11 -y
fi

# 5. Activate environment and install packages
echo "Installing Python packages..."
$HOME/python-env/.local/bin/micromamba run -n tf215gpu pip install \
    tensorflow==2.15 \
    numba \
    matplotlib \
    seaborn \
    cartopy \
    jupyter \
    xarray \
    netcdf4 \
    scikit-learn \
    cfgrib \
    dask \
    tqdm \
    properscoring \
    climlab \
    iris \
    ecmwf-api-client \
    xesmf \
    flake8 \
    regionmask \
    schedule

# 6. Create activation script
cat > ~/python-env/activate-tf.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/python-env/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"
source $HOME/python-env/.local/bin/activate
micromamba activate tf215gpu
echo "TensorFlow GPU environment activated!"
echo "Python location: $(which python)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU devices: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')"
EOF

chmod +x ~/python-env/activate-tf.sh

echo "Setup completed!"
echo "To activate environment: source ~/python-env/activate-tf.sh"
```

## Subsequent VM Sessions

For subsequent training sessions (after VM recreation), use this quick setup:

```bash
#!/bin/bash
# quick-setup.sh

# Mount persistent disk
sudo mkdir -p /mnt/python-env
sudo mount /dev/nvme0n2 /mnt/python-env
sudo chown -R $USER:$USER /mnt/python-env
ln -sf /mnt/python-env ~/python-env

# Set PATH
export PATH="$HOME/python-env/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/python-env/micromamba"

# Activate environment
source ~/python-env/activate-tf.sh
```

## Environment Management

### List Environments
```bash
micromamba env list
```

### Activate Environment
```bash
micromamba activate tf215gpu
```

### Deactivate Environment
```bash
micromamba deactivate
```

### Install Additional Packages
```bash
micromamba activate tf215gpu
pip install package_name
```

### Export Environment
```bash
micromamba activate tf215gpu
micromamba env export > environment.yml
```

## Troubleshooting

### Micromamba Not Found
```bash
# Add to PATH manually
export PATH="$HOME/python-env/.local/bin:$PATH"
source ~/.bashrc
```

### Environment Activation Issues
```bash
# Check environment exists
micromamba env list

# Recreate if corrupted
micromamba env remove -n tf215gpu
micromamba create -n tf215gpu python=3.11
```

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check CUDA version
nvcc --version
```

### Disk Space Issues
```bash
# Check space on persistent disk
df -h ~/python-env

# Clean conda cache
micromamba clean --all
```

## Directory Structure

```
~/python-env/ (50GB persistent SSD)
├── .local/
│   └── bin/
│       └── micromamba          # Micromamba executable
├── micromamba/                 # Environment storage
│   ├── envs/
│   │   └── tf215gpu/          # TensorFlow environment
│   ├── pkgs/                  # Package cache
│   └── conda-meta/
├── activate-tf.sh             # Environment activation script
└── .setup_complete           # Setup completion marker
```

## Cost Optimization Notes

- **Persistent Environment**: All packages and environments survive VM destruction
- **No Reinstallation**: Skip package installation on subsequent sessions
- **Faster Startup**: Environment ready in minutes vs hours
- **Version Consistency**: Same package versions across all training sessions

## Integration with Training Workflow

1. **First Session**: Run complete setup script
2. **Training Session**: Activate environment and start training
3. **Cleanup**: Destroy VM but preserve persistent disk
4. **Next Session**: Quick mount and activate existing environment

This setup ensures your Python environment is always ready for cGAN training while optimizing costs through the persistent disk strategy.