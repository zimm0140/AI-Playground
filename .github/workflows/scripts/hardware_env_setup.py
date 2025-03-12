#!/usr/bin/env python3
"""
Script to set up simulated hardware environments for CI testing.

This script creates mock files and environment variables for different hardware types 
to allow testing without actual hardware.
"""

import argparse
import os
import sys
from pathlib import Path


def setup_base_env():
    """Set up the base environment with no specialized hardware."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Create empty mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Generic GPU\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Generic\n")
        f.write("name: Generic CPU\n")
        f.write("cores: 4\n")

    # Set environment variables
    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir.absolute())
    print(f"Base environment set up in {mock_dir.absolute()}")


def setup_intel_arc_env():
    """Set up environment for Intel Arc GPU simulation."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Create mock files with Intel Arc GPU information
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) Arc(TM) A770 Graphics\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i9-13900K\n")
        f.write("cores: 24\n")

    # Set environment variables
    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir.absolute())
    os.environ["SIMULATED_HARDWARE"] = "acm"

    # Create a dummy intel-gpu-tools package file
    intel_gpu_dir = mock_dir / "intel_gpu"
    intel_gpu_dir.mkdir(exist_ok=True)
    with open(intel_gpu_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write("# Mock Intel GPU package\n")
        f.write("__version__ = '1.0.0'\n")

    print(f"Intel Arc environment set up in {mock_dir.absolute()}")


def setup_openvino_env():
    """Set up environment for OpenVINO simulation."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Create mock files with OpenVINO information
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) UHD Graphics 770\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i7-1370P\n")
        f.write("cores: 16\n")

    # Set environment variables
    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir.absolute())
    os.environ["SIMULATED_HARDWARE"] = "ovino"

    # Create a dummy OpenVINO package
    openvino_dir = mock_dir / "openvino"
    openvino_dir.mkdir(exist_ok=True)
    with open(openvino_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write("# Mock OpenVINO package\n")
        f.write("__version__ = '2023.1.0'\n")

    print(f"OpenVINO environment set up in {mock_dir.absolute()}")


def main():
    """Main function to parse arguments and set up the environment."""
    parser = argparse.ArgumentParser(description="Set up simulated hardware environment")
    parser.add_argument(
        "hardware_type",
        choices=["base", "acm", "ovino"],
        default="base",
        nargs="?",
        help="Type of hardware to simulate",
    )

    args = parser.parse_args()

    # Set up the environment based on the hardware type
    if args.hardware_type == "base":
        setup_base_env()
    elif args.hardware_type == "acm":
        setup_intel_arc_env()
    elif args.hardware_type == "ovino":
        setup_openvino_env()
    else:
        print(f"Unknown hardware type: {args.hardware_type}")
        sys.exit(1)

    print(f"Simulated hardware environment set up for: {args.hardware_type}")

    # Print environment variables for debugging
    print("\nEnvironment variables:")
    for var in sorted(os.environ):
        if var.startswith(("UVFAST", "SIMULATED", "PYTHON")):
            print(f"  {var}={os.environ[var]}")


if __name__ == "__main__":
    main()
