# Hardware Tools

This directory contains scripts for hardware detection and configuration in the AI Playground project.

## Contents

- `hardware_detection.py`: Script to detect and report hardware capabilities
- `setup_hardware_env.py`: Script to set up the hardware environment
- `intel_extension_for_pytorch.py`: Helper module for Intel extensions for PyTorch

## Usage

### Hardware Detection

To detect and report hardware capabilities:

```

python tools/hardware/hardware_detection.py

```

This script will:

- Detect Intel Arc GPUs and other Intel hardware
- Report hardware capabilities
- Check for driver compatibility
- Recommend optimizations based on hardware

### Hardware Environment Setup

To set up the hardware environment:

```

python tools/hardware/setup_hardware_env.py

```

This script will:

- Install required drivers and libraries
- Configure the environment for optimal performance
- Set up hardware-specific dependencies

## Related Documentation

For more information on hardware support, refer to:

- [Hardware Compatibility Guide](docs/hardware/compatibility.md)
- [Hardware Optimization Guide](docs/hardware/optimization.md)
