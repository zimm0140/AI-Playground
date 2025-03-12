#!/usr/bin/env python3
"""
Script to set up simulated hardware environments for CI testing.

This script creates mock files and environment variables for different hardware types 
to allow testing without actual hardware.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def debug_print(message, level="INFO"):
    """Print debug information with a timestamp."""
    print(f"[HARDWARE_ENV_SETUP:{level}] {message}")


def setup_base_env():
    """Set up the base environment with no specialized hardware."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    debug_print(f"Setting up base environment in {mock_dir.absolute()}")

    # Create empty mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Generic GPU\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Generic\n")
        f.write("name: Generic CPU\n")
        f.write("cores: 4\n")

    # Set environment variables
    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir.absolute())

    # Create environment file that can be sourced in CI
    with open(mock_dir / "env.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export UVFAST_MOCK_DIR='{mock_dir.absolute()}'\n")

    # Make the script executable
    try:
        os.chmod(mock_dir / "env.sh", 0o755)
    except Exception as e:
        debug_print(f"Warning: Could not make env.sh executable: {e}", "WARNING")

    # Create a simple JSON file with environment info
    env_info = {"UVFAST_MOCK_DIR": str(mock_dir.absolute()), "hardware_type": "base"}
    with open(mock_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    debug_print(f"Base environment set up in {mock_dir.absolute()}")
    return env_info


def setup_intel_arc_env():
    """Set up environment for Intel Arc GPU simulation."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    debug_print(f"Setting up Intel Arc environment in {mock_dir.absolute()}")

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

    # Create environment script
    with open(mock_dir / "env.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export UVFAST_MOCK_DIR='{mock_dir.absolute()}'\n")
        f.write("export SIMULATED_HARDWARE='acm'\n")

    # Make the script executable
    try:
        os.chmod(mock_dir / "env.sh", 0o755)
    except Exception as e:
        debug_print(f"Warning: Could not make env.sh executable: {e}", "WARNING")

    # Create a simple JSON file with environment info
    env_info = {
        "UVFAST_MOCK_DIR": str(mock_dir.absolute()),
        "SIMULATED_HARDWARE": "acm",
        "hardware_type": "acm",
    }
    with open(mock_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    debug_print(f"Intel Arc environment set up in {mock_dir.absolute()}")
    return env_info


def setup_openvino_env():
    """Set up environment for OpenVINO simulation."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    debug_print(f"Setting up OpenVINO environment in {mock_dir.absolute()}")

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

    # Create environment script
    with open(mock_dir / "env.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export UVFAST_MOCK_DIR='{mock_dir.absolute()}'\n")
        f.write("export SIMULATED_HARDWARE='ovino'\n")

    # Make the script executable
    try:
        os.chmod(mock_dir / "env.sh", 0o755)
    except Exception as e:
        debug_print(f"Warning: Could not make env.sh executable: {e}", "WARNING")

    # Create a simple JSON file with environment info
    env_info = {
        "UVFAST_MOCK_DIR": str(mock_dir.absolute()),
        "SIMULATED_HARDWARE": "ovino",
        "hardware_type": "ovino",
    }
    with open(mock_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    debug_print(f"OpenVINO environment set up in {mock_dir.absolute()}")
    return env_info


def ensure_package_structure():
    """Ensure package directory structure exists."""
    # Create essential directories if they don't exist
    for dir_path in ["tools", "tools/hardware", "tests", "tests/hardware"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py if it doesn't exist
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f"# Auto-generated {dir_path} package\n")

    debug_print("Package structure verified")


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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Ensure package structure
    ensure_package_structure()

    debug_print(f"Setting up environment for hardware type: {args.hardware_type}")

    # Set up the environment based on the hardware type
    try:
        if args.hardware_type == "base":
            env_info = setup_base_env()
        elif args.hardware_type == "acm":
            env_info = setup_intel_arc_env()
        elif args.hardware_type == "ovino":
            env_info = setup_openvino_env()
        else:
            debug_print(f"Unknown hardware type: {args.hardware_type}", "ERROR")
            sys.exit(1)
    except Exception as e:
        debug_print(f"Error setting up environment: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    debug_print(f"Simulated hardware environment set up for: {args.hardware_type}")

    # Print environment information
    debug_print("Environment variables that should be set:")
    for key, value in env_info.items():
        if key.isupper():  # Only print actual environment variables
            debug_print(f"  {key}={value}")

    # Print current environment variables for debugging
    if args.verbose:
        debug_print("Current environment variables:")
        for var in sorted(os.environ):
            if var.startswith(("UVFAST", "SIMULATED", "PYTHON")):
                debug_print(f"  {var}={os.environ[var]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
