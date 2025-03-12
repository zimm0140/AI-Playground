# Hardware Detection Package

A Python package for automatic hardware detection and configuration.

## Features

- Automatic detection of specialized hardware (GPUs, specialized processors)
- Support for Intel Arc GPUs and other hardware accelerators
- Configurable detection rules
- Mock environment support for CI/CD testing
- Cross-platform compatibility (Windows, Linux)

## Installation

````bash
pip install -e .

```text

## Usage

```python
from hardware_detection import detect_hardware_type, get_hardware_info

## Get detected hardware type

hardware_type = detect_hardware_type()
print(f"Detected hardware: {hardware_type}")

## Get detailed hardware information

hardware_info = get_hardware_info()
print(f"GPUs: {hardware_info['gpus']}")
print(f"CPU: {hardware_info['cpu']}")

```text

## Testing

```bash
python -m pytest hardware_detection/tests/

```text

## License

MIT License. See LICENSE file for details.

````
