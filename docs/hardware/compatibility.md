# Hardware Compatibility

This guide provides detailed information about hardware compatibility in AI-Playground, including recommended configurations, driver requirements, and known limitations.

## Supported Hardware Platforms

AI-Playground supports and has been tested on the following hardware platforms:

### Intel Platforms

| Platform | Minimum Version | Recommended Version | Notes |
|----------|----------------|-------------------|-------|
| Intel Arc GPUs | A380 | A770 | Requires driver 31.0.101.4255 or newer |
| Intel Battlemage GPUs | Any | Any | Early support, requires latest drivers |
| Intel Meteor Lake | Core Ultra 5 | Core Ultra 7 | Requires driver 31.0.101.4255 or newer |
| Intel Lunar Lake | Any | Any | Early support |
| Intel Core (non-Ultra) | 10th Gen | 12th Gen+ | Basic performance on older generations |

### NVIDIA Platforms

| GPU Series | Compute Capability | CUDA Version | Notes |
|------------|-------------------|-------------|-------|
| RTX 40 Series | 8.9 | 11.8+ | Fully supported |
| RTX 30 Series | 8.6 | 11.0+ | Fully supported |
| RTX 20 Series | 7.5 | 10.0+ | Fully supported |
| GTX 16 Series | 7.5 | 10.0+ | Fully supported |
| GTX 10 Series | 6.1 | 9.0+ | Supported, reduced performance |

### Other Platforms

| Platform | Support Level | Notes |
|----------|--------------|-------|
| AMD GPUs | Basic | Limited support via ROCm |
| Apple Silicon | Basic | CPU-only mode, no GPU acceleration |
| CPU-only | Full | Reduced performance for compute-intensive tasks |

## Driver Requirements

### Intel Graphics Drivers

| Hardware | Windows | Linux | macOS |
|----------|---------|-------|-------|
| Arc GPUs | 31.0.101.4255+ | Mesa 23.1+ | Not supported |
| Meteor Lake | 31.0.101.4255+ | Mesa 23.1+ | Not supported |
| Battlemage | 31.0.101.4521+ | Mesa 24.0+ | Not supported |

### NVIDIA Drivers

| CUDA Version | Minimum Driver | Recommended Driver |
|--------------|----------------|-------------------|
| CUDA 11.8 | 450.80.02 | 520.61.05+ |
| CUDA 11.0 | 450.36.06 | 455.23.05+ |
| CUDA 10.0 | 410.48 | 440.33+ |

## Software Requirements

The following software requirements apply based on the hardware platform:

| Hardware | Python Version | PyTorch Version | Notes |
|----------|---------------|----------------|-------|
| Intel Arc GPUs | 3.10+ | 2.0.0+ | Requires Intel Extension for PyTorch |
| Intel Meteor Lake | 3.10+ | 2.0.0+ | Requires Intel Extension for PyTorch |
| NVIDIA GPUs | 3.8+ | 1.10.0+ | CUDA 10.0+ required |
| CPU-only | 3.8+ | 1.10.0+ | No special requirements |

## Hardware-Specific Setup

### Intel Arc GPUs

\`\`\`text\`bash

## Install Intel GPU driver (Windows)

## Download from intel.com/graphics/drivers

## Setup environment with Arc optimizations

python setup_hardware_env.py --hardware acm

````text

### NVIDIA GPUs

```bash

## Install NVIDIA driver and CUDA toolkit

## Download from nvidia.com/drivers

## Setup environment with standard setup

python setup_hardware_env.py

```text

## Known Issues and Limitations

### Intel Arc GPUs

- Some operations may be slower than on NVIDIA counterparts
- Requires specific driver versions for optimal performance
- Memory usage may be higher than on NVIDIA GPUs

### NVIDIA GPUs

- Older GTX series GPUs have limited performance with larger models
- CUDA compatibility issues with some Python packages

### General Issues

- Mixed precision training may require platform-specific settings
- Large models (>6GB VRAM) may not work on lower-end GPUs

## Compatibility Testing

To verify hardware compatibility on your system:

```bash

## Run hardware detection

python hardware_detection.py

## Run compatibility test

python service/tools/hardware_compatibility_check.py

```text

This will generate a report about your hardware configuration and any potential compatibility issues.

---
**Previous**: [Hardware Overview](overview.md) | **Next**: [Hardware Optimization](optimization.md) | **See also**: [Intel Arc Guide](device-specific/intel-arc.md)
```text`
````
