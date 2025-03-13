
# Hardware Detection Module {#hardware-detection-module}

The hardware detection module provides a standardized way to detect and utilize available hardware accelerators
for machine learning workloads.

## Installation {#installation}

The module is included in the project and requires no additional installation steps.

If using from another project, you can install directly:

````bash

## From the repository root {#from-the-repository-root}

pip install -e .

```

## Usag

 {#usag}

e {#usage}

### Recommended Usage {#recommended-usage}

Import directly from the `hardware_detection` package:

```python

## Import f

 {#import-f}

rom the hardware_detection package (recommended) {#import-from-the-hardware_detection-package-recommended}

from hardware_detection import detect_hardware_type, get_gpu_info, get_hardware_info

## Check hardware type {#check-hardware-type}

hardware_type = detect_hardware_type()
if hardware_type == "bmg":  # NVIDIA GPU

    ## NVIDIA-specific code

    pass
elif hardware_type == "arl_h":  # AMD GPU

    ## AMD-specific code

    pass

```

### L

 {#l}

egacy Usage (Deprecated) {#legacy-usage-deprecated}

The module can also be imported from the legacy location, but this approach is deprecated:

```python

## Legacy

 {#legacy}

 import (deprecated) {#legacy-import-deprecated}

from tools.hardware import hardware_detection

## Check hardware type {#check-hardware-type}

hardware_type = hardware_detection.detect_hardware_type()

```

##

API Reference {#api-reference}

### Core Functions {#core-functions}

#### `detect_hardware_type() -> str` {#detect_hardware_type---str}

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

**Example:**

```python

from ha

rdware_detection import detect_hardware_type
hardware = detect_hardware_type()
print(f"Detected hardware: {hardware}")

```

#

### `get_gpu_info() -> List[str]` {#get_gpu_info---liststr}

Get information about available GPUs.

**Returns:**

- A list of strings describing the detected GPUs, or an empty list if no GPUs are found.

**Example:**

```python

from

hardware_detection import get_gpu_info
gpus = get_gpu_info()
print(f"Available GPUs: {gpus}")

```

#### `get_cpu_info() -> Dict[str, Any]` {#get_cpu_info---dictstr-any}

Get information about the CPU.

**Returns:**

- A dictionary containing CPU information:
  - `vendor`: CPU manufacturer
  - `name`: CPU model name
  - `cores`: Number of CPU cores
  - `features`: (Optional) List of CPU features

**Example:**

```python

fro

m hardware_detection import get_cpu_info
cpu = get_cpu_info()
print(f"CPU: {cpu['name']} with {cpu['cores']} cores")

``
`

#### `get_hardware_info() -> Dict[str, Any]` {#get_hardware_info---dictstr-any}

Get comprehensive information about the system hardware.

**Returns:**

- A dictionary with hardware information:
  - `system`: Operating system
  - `python_version`: Python version
  - `gpus`: List of available GPUs
  - `cpu`: CPU information
  - `detected_hardware`: Detected hardware type
  - `openvino_available`: Whether OpenVINO is available

**Example:**

``
`python

f
rom hardware_detection import get_hardware_info
info = get_hardware_info()
print(json.dumps(info, indent=2))

```

#

### `print_hardware_info(verbose: bool = False) -> None` {#print_hardware_infoverbose-bool-false---none}

Print information about the system hardware.

**Arguments:**

- `verbose`: Whether to show additional details

__Example:_*

```python

from hardware_detection import print_hardware_info
print_hardware_info(verbose=True)

```

## Environment Variables {#environment-variables}

The module supports these environment variables for testing and CI:

- `SIMULATED_HARDWARE`: Override hardware detection with a specific type

- `DEBUG_HARDWARE_DETECTION`: Enable detailed debug output

## Migration Guide {#migration-guide}

We are transitioning to a proper package structure. The legacy import path is
still supported but will display deprecation warnings.

### Why Migrate? {#why-migrate}

The new package structure provides:

1. Better dependency management

1. Proper namespacing

1. Type hints and documentation

1. Easier testing

1. Future extensibility

### Steps to Migrate {#steps-to-migrate}

1. Replace import statements:

   ```python

   ## Old

   from tools.hardware import hardware_detection

   ## New

   from hardware_detection import detect_hardware_type, get_gpu_info

   ```

1. Update function calls:

   ```python

   ## Old

   hardware_type = hardware_detection.detect_hardware_type()

   ## New

   hardware_type = detect_hardware_type()

   ```

````