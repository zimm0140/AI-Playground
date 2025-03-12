#!/usr/bin/env python3
"""Compatibility module for hardware detection.

This module provides backward compatibility with the previous file-based structure.
It imports all functions from the new package structure and re-exports them.
"""

import os
import sys
import warnings

# Try to add the root directory to the path if the package isn't installed
try:
    # Try importing from the new package structure
    from hardware_detection import (
        __version__,
        detect_hardware_type,
        get_cpu_info,
        get_gpu_info,
        get_hardware_info,
        is_openvino_available,
        print_hardware_info,
    )

    # Issue deprecation warning
    warnings.warn(
        "Using tools/hardware/hardware_detection.py is deprecated. "
        "Import directly from 'hardware_detection' package instead.",
        DeprecationWarning,
        stacklevel=2,
    )

except ImportError:
    # If package is not found, add helpful error message
    # and implement basic stubs for CI
    __version__ = "0.0.0-stub"

    def detect_hardware_type() -> str:
        """Stub function for hardware type detection."""
        # Default to 'base' for CI environments
        return os.environ.get("SIMULATED_HARDWARE", "base")

    def get_gpu_info() -> list:
        """Stub function for GPU info."""
        return ["Stub GPU for CI"]

    def get_cpu_info() -> dict:
        """Stub function for CPU info."""
        return {
            "vendor": "Stub",
            "name": "Stub CPU for CI",
            "cores": 2,
        }

    def is_openvino_available() -> bool:
        """Stub function for OpenVINO availability."""
        return False

    def get_hardware_info() -> dict:
        """Stub function for hardware info."""
        return {
            "system": "CI",
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "gpus": get_gpu_info(),
            "cpu": get_cpu_info(),
            "detected_hardware": detect_hardware_type(),
            "openvino_available": is_openvino_available(),
        }

    def print_hardware_info(verbose: bool = False) -> None:
        """Stub function to print hardware info."""
        info = get_hardware_info()
        print(f"System: {info['system']}")
        print(f"Python version: {info['python_version']}")
        print(f"Detected hardware type: {info['detected_hardware']}")
        print(f"GPUs: {info['gpus']}")
        print(f"CPU: {info['cpu']}")


# Re-export everything to maintain the same API
__all__ = [
    "__version__",
    "detect_hardware_type",
    "get_cpu_info",
    "get_gpu_info",
    "get_hardware_info",
    "is_openvino_available",
    "print_hardware_info",
]

# For CLI compatibility
if __name__ == "__main__":
    print(f"Hardware Detection Module v{__version__}")
    print_hardware_info(verbose=("-v" in sys.argv or "--verbose" in sys.argv))
