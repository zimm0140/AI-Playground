
# Hardware Tools

This directory contains scripts for hardware detection and configuration in the AI Playground project.

## Contents

- `hardware_detection.py`: Script to detect and report hardware capabilities
- `setup_hardware_env.py`: Script to set up the hardware environment
- `intel_extension_for_pytorch.py`: Helper module for Intel extensions for PyTorch

## Usage

### Hardware Detection

To detect and report hardware capabilities:

```text`text

python tools/hardware/hardware_detection.py

```text

This script will:

- Detect Intel Arc GPUs and other Intel hardware
- Report hardware capabilities
- Check for driver compatibility
- Recommend optimizations based on hardware

### Hardware Environment Setup

To set up the hardware environment:

```text

python tools/hardware/setup_hardware_env.py

```text

This script will:

- Install required drivers and libraries
- Configure the environment for optimal performance
- Set up hardware-specific dependencies

## Related Documentation

For more information on hardware support, refer to:

- [Hardware Compatibility Guide](docs/hardware/compatibility.md)
- [Hardware Optimization Guide](docs/hardware/optimization.md)

## Hardware Detection Module

This module provides functionality to detect and identify hardware, particularly Intel GPUs and specialized processors that require specific Python packages for optimal
performance.

## Overview

The hardware detection module is designed to:

1. Detect available hardware (GPUs, CPUs)

1. Identify specific hardware types (Intel Arc GPUs, OpenVINO-compatible devices, etc.)

1. Provide information about hardware capabilities

1. Support CI/CD testing with simulated hardware environments

## Usage

### Basic Usage

```python

from tools.hardware.hardware_detection import detect_hardware_type, get_hardware_info

## Get the detected hardware type

hw_type = detect_hardware_type()
print(f"Detected hardware type: {hw_type}")

## Get detailed hardware information

info = get_hardware_info()
print(f"System: {info['system']}")
print(f"Python version: {info['python_version']}")
print(f"GPUs: {info['gpus']}")
print(f"CPU: {info['cpu']}")
print(f"OpenVINO available: {info['openvino_available']}")

```text

### Command Line Usage

You can also run the module directly to print hardware information:

```bash

python tools/hardware/hardware_detection.py

```text

Add the `--verbose` or `-v` flag to see more detailed information:

```bash

python tools/hardware/hardware_detection.py --verbose

```text

## Hardware Types

The module supports the following hardware types:

- `base`: Default hardware with no specialized acceleration
- `acm`: Intel Arc GPUs (Alchemist architecture)
- `bmg`: Intel Battlemage GPUs
- `mtl`: Intel Meteor Lake integrated graphics
- `lnl`: Intel Lunar Lake integrated graphics
- `ovino`: Systems with OpenVINO runtime
- `arl_h`: Intel Arrow Lake high-end integrated graphics

## CI/CD Testing

The module supports CI/CD testing with simulated hardware environments. This allows testing hardware-specific code without requiring the actual hardware.

### Simulating Hardware in CI

To simulate specific hardware in CI, set the `SIMULATED_HARDWARE` environment variable:

```bash

export SIMULATED_HARDWARE=acm
python tools/hardware/hardware_detection.py

```text

You can also use the provided setup script:

```bash

python .github/workflows/scripts/hardware_env_setup.py acm

```text

### Mock Files

The module can also use mock files to simulate hardware. Create a directory with the following structure:

```text

.uvfast/mock/
  ├── gpu_info.txt
  └── cpu_info.txt

```text

Then set the `UVFAST_MOCK_DIR` environment variable:

```bash

export UVFAST_MOCK_DIR=.uvfast/mock

```text

## Configuration

The module uses a configuration file (`uvfast.json`) to customize hardware detection. The configuration file should be placed in the project root directory.

Example configuration:

```json

{
  "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
  "default_hardware": "base",
  "detection": {
    "acm": {
      "gpu_name_pattern": "Intel.*Arc"
    },
    "ovino": {
      "cpu_name_pattern": "Intel.*Core.*i\\d-\\d{4}"
    }
  }
}

```text

```text`
