# Hardware Detection

A Python package for detecting and managing hardware configurations.

## Features

- Hardware type detection
- GPU information retrieval
- CPU information retrieval
- Cross-platform support (Windows, Linux)
- Mock hardware support for testing

## Installation

````bash
pip install -e .

```text

## Usage

```python
from hardware_detection import detect_hardware_type, get_gpu_info

## Detect hardware type

hardware_type = detect_hardware_type()

## Get GPU information

gpu_info = get_gpu_info()

```text

## Development

1. Clone the repository
1. Install dependencies: `pip install -e ".[dev]"`
1. Run tests: `pytest`

## License

MIT

````
