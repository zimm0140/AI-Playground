# Installation Guide

This guide provides detailed instructions for installing and configuring AI-Playground for different environments and hardware configurations.

## System Requirements

### Minimum Requirements

- Python 3.10 or higher
- 8GB RAM
- 10GB disk space

### Recommended Requirements

- Python 3.10 or 3.11
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM or Intel Arc GPU
- SSD with 20GB+ free space

### Supported Operating Systems

- Windows 10/11 (64-bit)
- Ubuntu 20.04/22.04 LTS
- macOS 11.0 or newer (limited GPU acceleration)

## Installation Methods

### Method 1: Automatic Setup (Recommended)

The `setup_hardware_env.py` script automatically detects your hardware and sets up the appropriate environment:

```bash
# Clone the repository
git clone https://github.com/intel/AI-Playground.git
cd AI-Playground

# Run the automatic setup
python setup_hardware_env.py
```

#### Additional Options

```bash
# Include development dependencies
python setup_hardware_env.py --dev

# Force a specific hardware configuration
python setup_hardware_env.py --hardware acm  # For Intel Arc GPUs
python setup_hardware_env.py --hardware mtl  # For Intel Meteor Lake CPUs

# Clean existing environment before setup
python setup_hardware_env.py --clean
```

### Method 2: Manual Setup

If you prefer to set up the environment manually:

1. Create a virtual environment:

   ```bash
   # Using venv
   python -m venv .venv
   
   # Activate the environment
   # On Windows
   .venv\Scripts\activate
   # On Linux/macOS
   source .venv/bin/activate
   ```

2. Install dependencies based on your hardware:

   ```bash
   # For basic CPU setup
   pip install -r requirements.txt
   
   # For Intel Arc GPUs
   pip install -r requirements-hardware-acm.txt
   
   # For development
   pip install -r requirements-dev.txt
   ```

## Docker Installation

For containerized deployment:

```bash
# Build the Docker image
docker build -t ai-playground .

# Run the container
docker run -p 8000:8000 ai-playground
```

## Troubleshooting Installation

### Common Issues

1. **Package installation failures**

   Try updating pip and setuptools:

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **GPU not detected**

   Ensure you have the latest GPU drivers installed for your hardware.

3. **Python version compatibility**

   If you encounter compatibility issues, we strongly recommend using Python 3.10.

For more troubleshooting help, see the [Troubleshooting Guide](../reference/troubleshooting.md).

## Verifying Installation

To verify that your installation is working correctly:

```bash
# Activate the virtual environment if not already activated
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Run the verification script
python test_venv.py
```

This will check that all required dependencies are installed and that your hardware is properly detected.

---
**Previous**: [Quick Start Guide](quickstart.md) | **Next**: [Migration Guide](migration.md) | **See also**: [Hardware Compatibility](../hardware/compatibility.md)
