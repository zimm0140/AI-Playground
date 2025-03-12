# Hardware Detection

A Python package for detecting and managing hardware configurations.

## Features

- Hardware type detection
- GPU information retrieval
- CPU information retrieval
- Cross-platform support (Windows, Linux)
- Mock hardware support for testing

## Installation

```bash
pip install -e .

```

## Usage

```python
from hardware_detection import detect_hardware_type, get_gpu_info

# Detect hardware type

hardware_type = detect_hardware_type()

# Get GPU information

gpu_info = get_gpu_info()

```

## Development

1. Clone the repository
2. Install dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`

## License

MIT
