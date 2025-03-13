
# Hardware-Aware Python Environment Management

This project implements a hardware-aware environment management system for Python projects, with a focus on Intel hardware acceleration for machine learning workloads.

## Overview

The implementation combines modern Python packaging practices with hardware detection to provide optimized environments for different hardware configurations:

- Intel Arc GPUs (A-Series)
- Intel Battlemage GPUs (B-Series)
- Intel Meteor Lake processors
- Intel Lunar Lake processors
- OpenVINO acceleration
- Fallback to standard configurations

## Components

### Core Components

1. *_uvfast.py__: Main CLI tool for environment management

   - Setup environments for specific hardware
   - Run commands with appropriate hardware configurations
   - Generate lockfiles for reproducible environments
   - Provide information about detected hardware

1. **hardware_detection.py**: Hardware detection module

   - Detect Intel GPUs and processors
   - Identify appropriate dependencies based on hardware
   - Check for hardware-specific features

1. **pyproject.toml**: Modern PEP 621 configuration

   - Core dependencies for all environments
   - Optional dependencies for specific hardware types
   - Tool configuration for linting, testing, etc.

### Additional Components

1. **GitHub Actions Workflow**: CI/CD pipeline for testing

   - Matrix testing across multiple Python versions
   - Simulated hardware environments
   - Lockfile validation

1. **Docker Configuration**: Containerized development

   - Hardware-specific container targets
   - Optimized for different acceleration types

1. __XPU Integration Example_*: Demo of hardware-aware configuration

   - Shows how to use hardware detection with PyTorch
   - Configures backends based on available hardware

## Installation and Usage

### Quick Start

1. Clone the repository and navigate to the project directory:

\`\`\`text\`bash
git clone <repository-url>
cd <project-directory>

```text`text

1. Run the setup command to create an environment for your hardware:

```bash

python uvfast.py setup --dev

```text

1. Activate the virtual environment:

```bash

## On Windows

.venv\Scripts\activate

## On Linux/macOS

source .venv/bin/activate

```text

1. Run commands in the optimized environment:

```bash

python uvfast.py run pytest

```text

### Manual Hardware Selection

If you want to specify a hardware type explicitly:

```bash

python uvfast.py setup --hardware acm --dev

```text

Available hardware types:

- `base`: Standard configuration
- `acm`: Intel Arc GPUs (A-Series)
- `bmg`: Intel Battlemage GPUs (B-Series)
- `mtl`: Intel Meteor Lake processors
- `lnl`: Intel Lunar Lake processors
- `ovino`: OpenVINO acceleration
- `arl_h`: Intel Arc Alchemist Hardware

### Traditional Installation

For traditional installation with pip (but accelerated with uv):

```bash

python uvfast.py legacy-install --dev

```text

Or for fully traditional installation:

```bash

pip install -e .
pip install -e ".[dev]"

```text

For hardware-specific dependencies:

```bash

pip install -e ".[acm]"  # For Intel Arc GPUs

pip install -e ".[ovino]"  # For OpenVINO

```text

## Lockfile Management

Generate lockfiles for reproducible environments:

```bash

## Generate lockfile for current hardware

python uvfast.py lock

## Generate lockfiles for all hardware types

python uvfast.py lock --all

```text

Install from lockfiles:

```bash

python uvfast.py setup --use-lockfile

```text

## Environment Information

Display information about the current hardware and environment:

```bash

python uvfast.py info

```text

For more detailed information:

```bash

python uvfast.py info --verbose

```text

## Hardware-Specific Development

The examples directory contains a sample script demonstrating how to use hardware detection with PyTorch:

```bash

python examples/xpu_integration.py

```text

This script automatically configures PyTorch for the appropriate backend (XPU for Intel GPUs, OpenVINO, or CPU) based on the detected hardware.

## CI/CD Integration

The GitHub Actions workflow in `.github/workflows/hardware-matrix.yml` demonstrates how to set up CI/CD for hardware-aware testing. It includes:

- Matrix testing across multiple Python versions and hardware configurations
- Hardware simulation for CI environments
- Lockfile validation

## Docker Usage

For containerized development:

```bash

## Build the container for your hardware

docker build --target acm -t my-project:acm .  # For Intel Arc GPUs

docker build --target openvino -t my-project:openvino .  # For OpenVINO

## Run the container

docker run -it my-project:acm

```text

Or using Docker Compose with profiles:

```bash

docker-compose --profile acm up
docker-compose --profile openvino up

```text

## Advanced Configuration

You can customize the hardware detection and environment configuration by modifying the `uvfast.json` file. This allows you to:

- Add new hardware types
- Customize detection patterns
- Change dependency paths
- Configure environment settings

```text`

```text`