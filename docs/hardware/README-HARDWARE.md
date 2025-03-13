
# Hardware-Aware Python Environment Management {#hardware-aware-python-environment-management}

This project implements a hardware-aware environment management system for Python projects, with a focus on Intel hardware acceleration for machine learning workloads.

## Overview {#overview}

The implementation combines modern Python packaging practices with hardware detection to provide optimized environments for different hardware configurations:

- Intel Arc GPUs (A-Series)

- Intel Battlemage GPUs (B-Series)

- Intel Meteor Lake processors

- Intel Lunar Lake processors

- OpenVINO acceleration

- Fallback to standard configurations

## Components {#components}

### Core Components {#core-components}

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

### Additional Components {#additional-components}

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

## Installation and Usage {#installation-and-usage}

### Quick Start {#quick-start}

1. Clone the repository and navigate to the project directory:

\`\`\`text\`bash
git clone <repository-url>
cd <project-directory>

````

1. Run the setup command to create an environment for your hardware:

```bash

python uvfast.p
y
setup --dev

```

1. Activate the virtual
 e

nvironment:

```bash

## On Window

s

{#on-windows}

.venv\Scripts\activate

## On Linux/macOS {#on-linuxmacos}

source .venv/bin/activate

```

1. Run commands in th
e

optimized environment:

```bash

python uvfa
st
.py run pytest

```

### Manual Hardware

 S {#manual-hardware-s}

election {#manual-hardware-selection}

If you want to specify a hardware type explicitly:

```bash

python uv
fa
st.py setup --hardware acm --dev

```

Available hardwar
e
types:

- `base`: Standard configuration

- `acm`: Intel Arc GPUs (A-Series)

- `bmg`: Intel Battlemage GPUs (B-Series)

- `mtl`: Intel Meteor Lake processors

- `lnl`: Intel Lunar Lake processors

- `ovino`: OpenVINO acceleration

- `arl_h`: Intel Arc Alchemist Hardware

### Traditional Installation {#traditional-installation}

For traditional installation with pip (but accelerated with uv):

```bash

python
uv
fast.py legacy-install --dev

```

Or for fully tr
ad
itional installation:

```bash

pip i
ns
tall -e .
pip install -e ".[dev]"

```

For hardware-
sp
ecific dependencies:

```bash

pip
 i
nstall -e ".[acm]"  # For Intel Arc GPUs

pip install -e ".[ovino]"  # For OpenVINO

```

## Lockfile

 M {#lockfile-m}

anagement {#lockfile-management}

Generate lockfiles for reproducible environments:

```bash

##

 Generate lockfile for current hardware {#generate-lockfile-for-current-hardware}

python uvfast.py lock

## Generate lockfiles for all hardware types {#generate-lockfiles-for-all-hardware-types}

python uvfast.py lock --all

```

Install f
ro
m lockfiles:

```bash

p
ython uvfast.py setup --use-lockfile

```

## Envi

ro {#enviro}

nment Information {#environment-information}

Display information about the current hardware and environment:

```bas
h

python uvfast.py info

```

For m
or
e detailed information:

```b
as
h

python uvfast.py info --verbose

```

##

Ha {#ha}

rdware-Specific Development {#hardware-specific-development}

The examples directory contains a sample script demonstrating how to use hardware detection with PyTorch:

``
`b
ash

python examples/xpu_integration.py

```

T
hi
s script automatically configures PyTorch for the appropriate backend (XPU for Intel GPUs, OpenVINO, or CPU) based on the detected hardware.

## CI/CD Integration {#cicd-integration}

The GitHub Actions workflow in `.github/workflows/hardware-matrix.yml` demonstrates how to set up CI/CD for hardware-aware testing. It includes:

- Matrix testing across multiple Python versions and hardware configurations

- Hardware simulation for CI environments

- Lockfile validation

## Docker Usage {#docker-usage}

For containerized development:

``
`bash

## Build the container for your hardware {#build-the-container-for-your-hardware}

docker build --target acm -t my-project:acm .  # For Intel Arc GPUs

docker build --target openvino -t my-project:openvino .  # For OpenVINO

## Run the container {#run-the-container}

docker run -it my-project:acm

```

O
r using Docker Compose with profiles:

```bash

docker-compose --profile acm up
docker-compose --profile openvino up

```

## Advanced Configuration {#advanced-configuration}

You can customize the hardware detection and environment configuration by modifying the `uvfast.json` file. This allows you to:

- Add new hardware types

- Customize detection patterns

- Change dependency paths

- Configure environment settings

````

````
