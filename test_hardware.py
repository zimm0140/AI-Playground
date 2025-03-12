#!/usr/bin/env python3
"""
Simple script to test hardware detection.

This script can be run directly to test the hardware detection module.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the hardware detection module
from tools.hardware.hardware_detection import (
    __version__,
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    is_openvino_available,
    print_hardware_info,
)


def main():
    """Run hardware detection tests."""
    print(f"Hardware Detection Module v{__version__}")
    print("\n" + "=" * 50)
    print("Hardware Detection Test")
    print("=" * 50)

    # Check for simulated hardware
    if "SIMULATED_HARDWARE" in os.environ:
        print(f"\nSimulated hardware: {os.environ['SIMULATED_HARDWARE']}")

    # Print hardware information
    print("\nHardware Information:")
    print_hardware_info(verbose=True)

    # Test hardware type detection
    print("\nDetected Hardware Type:")
    hw_type = detect_hardware_type()
    print(f"  {hw_type}")

    # Test GPU detection
    print("\nDetected GPUs:")
    gpus = get_gpu_info()
    if gpus:
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("  No GPUs detected")

    # Test CPU detection
    print("\nDetected CPU:")
    cpu_info = get_cpu_info()
    print(f"  Vendor: {cpu_info.get('vendor', 'Unknown')}")
    print(f"  Name: {cpu_info.get('name', 'Unknown')}")
    print(f"  Cores: {cpu_info.get('cores', 'Unknown')}")

    # Test OpenVINO availability
    print("\nOpenVINO availability:")
    openvino_available = is_openvino_available()
    print(f"  Available: {openvino_available}")

    print("\n" + "=" * 50)
    print("Test completed successfully")
    print("=" * 50)


if __name__ == "__main__":
    main()
