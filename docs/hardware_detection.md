# Hardware Detection Module

The hardware detection module provides a standardized way to detect and utilize available hardware accelerators
for machine learning workloads.

## Installation

The module is included in the project and requires no additional installation steps.

If using from another project, you can install directly:

````bash

## From the repository root

pip install -e .

```text

## Usage

### Recommended Usage

Import directly from the `hardware_detection` package:

```python

## Import from the hardware_detection package (recommended)

from hardware_detection import detect_hardware_type, get_gpu_info, get_hardware_info

## Check hardware type

hardware_type = detect_hardware_type()
if hardware_type == "bmg":  # NVIDIA GPU

    ## NVIDIA-specific code

    pass
elif hardware_type == "arl_h":  # AMD GPU

    ## AMD-specific code

    pass

```text

### Legacy Usage (Deprecated)

The module can also be imported from the legacy location, but this approach is deprecated:

```python

## Legacy import (deprecated)

from tools.hardware import hardware_detection

## Check hardware type

hardware_type = hardware_detection.detect_hardware_type()

```text

## API Reference

### Core Functions

#### `detect_hardware_type() -> str`

Detect the available hardware type.

*_Returns:__

- A string representing the hardware type:
  - `"base"`: CPU only
  - `"bmg"`: NVIDIA GPU
  - `"arl_h"`: AMD GPU
  - `"acm"`: Intel Arc GPU
  - `"mtl"`: Intel Meteor Lake
  - `"lnl"`: Intel Lunar Lake
  - `"ovino"`: OpenVINO-compatible

__Example:__

```python
from hardware_detection import detect_hardware_type
hardware = detect_hardware_type()
print(f"Detected hardware: {hardware}")

```text

#### `get_gpu_info() -> List[str]`

Get information about available GPUs.

__Returns:__

- A list of strings describing the detected GPUs, or an empty list if no GPUs are found.

__Example:__

```python
from hardware_detection import get_gpu_info
gpus = get_gpu_info()
print(f"Available GPUs: {gpus}")

```text

#### `get_cpu_info() -> Dict[str, Any]`

Get information about the CPU.

__Returns:__

- A dictionary containing CPU information:
  - `vendor`: CPU manufacturer
  - `name`: CPU model name
  - `cores`: Number of CPU cores
  - `features`: (Optional) List of CPU features

__Example:__

```python
from hardware_detection import get_cpu_info
cpu = get_cpu_info()
print(f"CPU: {cpu['name']} with {cpu['cores']} cores")

```text

#### `get_hardware_info() -> Dict[str, Any]`

Get comprehensive information about the system hardware.

__Returns:__

- A dictionary with hardware information:
  - `system`: Operating system
  - `python_version`: Python version
  - `gpus`: List of available GPUs
  - `cpu`: CPU information
  - `detected_hardware`: Detected hardware type
  - `openvino_available`: Whether OpenVINO is available

__Example:__

```python
from hardware_detection import get_hardware_info
info = get_hardware_info()
print(json.dumps(info, indent=2))

```text

#### `print_hardware_info(verbose: bool = False) -> None`

Print information about the system hardware.

__Arguments:__

- `verbose`: Whether to show additional details

__Example:_*

```python
from hardware_detection import print_hardware_info
print_hardware_info(verbose=True)

```text

## Environment Variables

The module supports these environment variables for testing and CI:

- `SIMULATED_HARDWARE`: Override hardware detection with a specific type
- `DEBUG_HARDWARE_DETECTION`: Enable detailed debug output

## Migration Guide

We are transitioning to a proper package structure. The legacy import path is
still supported but will display deprecation warnings.

### Why Migrate?

The new package structure provides:

1. Better dependency management
2. Proper namespacing
3. Type hints and documentation
4. Easier testing
5. Future extensibility

### Steps to Migrate

1. Replace import statements:

   ```python
   ## Old

   from tools.hardware import hardware_detection

   ## New

   from hardware_detection import detect_hardware_type, get_gpu_info
   ```text

1. Update function calls:

   ```python
   ## Old

   hardware_type = hardware_detection.detect_hardware_type()

   ## New

   hardware_type = detect_hardware_type()
   ```text

````

