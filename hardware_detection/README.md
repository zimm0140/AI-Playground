# Hardware Detection Package

A robust hardware detection package that identifies specialized hardware like GPUs, CPUs, and accelerators across different platforms. This package is designed for CI/CD environments and includes Docker-based testing capabilities.

## Features

- Cross-platform detection of GPUs and CPUs
- Hardware-specific package detection
- Configuration-based hardware type identification
- Mock environments for testing
- Docker containers for consistent CI/CD
- Comprehensive test suite

## Installation

```bash
# Basic installation
pip install -e .

# For development
pip install -e ".[dev]"

# With hardware-specific dependencies
pip install -e ".[intel,openvino]"
```

## Usage

### Basic Usage

```python
from hardware_detection import detect_hardware_type, get_hardware_info

# Get the detected hardware type
hw_type = detect_hardware_type()
print(f"Detected hardware: {hw_type}")

# Get detailed hardware information
hw_info = get_hardware_info()
print(f"GPUs: {hw_info['gpus']}")
print(f"CPU: {hw_info['cpu']}")
```

### Command-line Interface

```bash
# Show hardware information
python -m hardware_detection.cli info

# Get JSON output
python -m hardware_detection.cli info --json

# Just detect hardware type
python -m hardware_detection.cli detect

# Create mock environment
python -m hardware_detection.cli mock --hardware-type acm
```

## Docker Containers

The package includes pre-configured Docker containers for testing different hardware environments:

```bash
# Build and run the base container
docker-compose build base
docker-compose run base

# Build and run the Intel Arc container
docker-compose build acm
docker-compose run acm

# Build and run the OpenVINO container
docker-compose build ovino
docker-compose run ovino

# Run tests in the specified container
docker-compose run -e SIMULATED_HARDWARE=acm test
```

## Configuration

Hardware detection is configured via a `uvfast.json` file. Example:

```json
{
  "hardware_types": ["base", "acm", "ovino"],
  "default_hardware": "base",
  "detection": {
    "acm": {
      "gpu_name_pattern": "Intel.*Arc|Arc.*Graphics",
      "cpu_name_pattern": "Intel.*i9"
    },
    "ovino": {
      "cpu_name_pattern": "Intel.*i7",
      "package_check": "openvino"
    }
  }
}
```

## Development

### Running Tests

```bash
# Run tests with pytest
pytest hardware_detection/tests

# Run tests in Docker
docker-compose build test
docker-compose run test
```

### Mock Environments

For testing different hardware configurations, use the `SIMULATED_HARDWARE` environment variable:

```bash
# Simulate Intel Arc GPU
SIMULATED_HARDWARE=acm python -m hardware_detection.cli info

# Simulate OpenVINO environment
SIMULATED_HARDWARE=ovino python -m hardware_detection.cli info

# Create mock environment files
python -m hardware_detection.cli mock --hardware-type acm
```

## CI/CD Integration

Include the package in your CI/CD pipeline:

```yaml
jobs:
  hardware-test:
    runs-on: ubuntu-latest
    container:
      image: your-registry/hardware-detection:acm
    steps:
      - name: Test hardware detection
        run: python -m hardware_detection.cli info
```

## License

This package is licensed under the MIT License.
