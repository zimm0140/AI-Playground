
# Installation Guide {#installation-guide}

This guide provides detailed instructions for installing and configuring AI-Playground for different environments and hardware configurations.

## System Requirements {#system-requirements}

### Minimum Requirements {#minimum-requirements}

- Python 3.10 or higher

- 8GB RAM

- 10GB disk space

### Recommended Requirements {#recommended-requirements}

- Python 3.10 or 3.11

- 16GB+ RAM

- NVIDIA GPU with 6GB+ VRAM or Intel Arc GPU

- SSD with 20GB+ free space

### Supported Operating Systems {#supported-operating-systems}

- Windows 10/11 (64-bit)

- Ubuntu 20.04/22.04 LTS

- macOS 11.0 or newer (limited GPU acceleration)

## Installation Methods {#installation-methods}

### Method 1: Automatic Setup (Recommended) {#method-1-automatic-setup-recommended}

The `setup_hardware_env.py` script automatically detects your hardware and sets up the appropriate environment:

\`\`\`text\`bash

## Clone the repository {#clone-the-repository}

git clone <https://github.com/intel/AI-Playground.git>
cd AI-Playground

## Run the automatic setup {#run-the-automatic-setup}

python setup_hardware_env.py

````

#### Additional Options {#additional-options}

``
`b
ash

## Include development dependencies {#include-development-dependencies}

python setup_hardware_env.py --dev

## Force a specific hardware configuration {#force-a-specific-hardware-configuration}

python setup_hardware_env.py --hardware acm  # For Intel Arc GPUs

python setup_hardware_env.py --hardware mtl  # For Intel Meteor Lake CPUs

## Clean existing environment before setup {#clean-existing-environment-before-setup}

python setup_hardware_env.py --clean

```

#

##

 Method 2: Manual Setup {#method-2-manual-setup}

If you prefer to set up the environment manually:

1. Create a virtual environment:

   ```bash

   ## Using venv

   python -m venv .venv

   ## Activate the environment

   ## On Windows

   .venv\Scripts\activate

   ## On Linux/macOS

   source .venv/bin/activate

   ```

1. Install dependencies based on your hardware:

   ```bash

   ## For basic CPU setup

   pip install -r requirements.txt

   ## For Intel Arc GPUs

   pip install -r requirements-hardware-acm.txt

   ## For development

   pip install -r requirements-dev.txt

   ```

## Docker Installation {#docker-installation}

For containerized deployment:

``
`bash

## Build the Docker image {#build-the-docker-image}

docker build -t ai-playground .

## Run the container {#run-the-container}

docker run -p 8000:8000 ai-playground

```

#

# Troubleshooting Installation {#troubleshooting-installation}

### Common Issues {#common-issues}

1. *_Package installation failures__

   Try updating pip and setuptools:

   ```bash

   pip install --upgrade pip setuptools wheel

   ```

1. **GPU not detected**

   Ensure you have the latest GPU drivers installed for your hardware.

1. **Python version compatibility**

   If you encounter compatibility issues, we strongly recommend using Python 3.10.

For more troubleshooting help, see the [Troubleshooting Guide](../reference/troubleshooting.md).

## Verifying Installation {#verifying-installation}

To verify that your installation is working correctly:

```bash

## Activate the virtual environment if not already activated {#activate-the-virtual-environment-if-not-already-activated}

## Windows {#windows}

.venv\Scripts\activate

## Linux/macOS {#linuxmacos}

source .venv/bin/activate

## Run the verification script {#run-the-verification-script}

python test_venv.py

```

This will check that all required dependencies are installed and that your hardware is properly detected.

---
**Previous**: [Quick Start Guide](quickstart.md) | **Next**: [Migration Guide](migration.md) | __See also_*: [Hardware Compatibility](../hardware/compatibility.md)


````

````
