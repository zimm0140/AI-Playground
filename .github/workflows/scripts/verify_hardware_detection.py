#!/usr/bin/env python3
"""
Hardware Detection Verification Script

This script verifies that hardware detection is working properly
and provides detailed diagnostics for troubleshooting CI issues.
"""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def debug_print(message, level="INFO"):
    """Print debug information with a timestamp."""
    print(f"[VERIFY_HW:{level}] {message}")


def check_environment():
    """Check environment variables and paths."""
    debug_print("Checking environment...")

    # Check environment variables
    env_vars = {
        "SIMULATED_HARDWARE": os.environ.get("SIMULATED_HARDWARE", "Not set"),
        "UVFAST_MOCK_DIR": os.environ.get("UVFAST_MOCK_DIR", "Not set"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
    }

    for var, value in env_vars.items():
        debug_print(f"{var}: {value}")

    # Check if UVFAST_MOCK_DIR exists and has expected files
    mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    if mock_dir and Path(mock_dir).exists():
        debug_print(f"Mock directory exists: {mock_dir}")
        files = list(Path(mock_dir).glob("*"))
        debug_print(f"Files in mock directory: {[str(f.name) for f in files]}")

        # Check for specific files
        for expected_file in ["gpu_info.txt", "cpu_info.txt", "env.json"]:
            file_path = Path(mock_dir) / expected_file
            if file_path.exists():
                debug_print(f"Found {expected_file}")
                if expected_file.endswith(".json"):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = json.load(f)
                            debug_print(f"{expected_file} content: {content}")
                    except json.JSONDecodeError:
                        debug_print(f"Error parsing {expected_file}", "ERROR")
                else:
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read().strip()
                            debug_print(f"{expected_file} content: {content}")
                    except Exception as e:
                        debug_print(f"Error reading {expected_file}: {e}", "ERROR")
            else:
                debug_print(f"{expected_file} not found", "WARNING")
    else:
        debug_print("Mock directory not found or not set", "WARNING")

    return env_vars


def check_python_packages():
    """Check for required Python packages."""
    debug_print("Checking Python packages...")

    packages = [
        "openvino",
        "intel_gpu",
        "pytest",
    ]

    for package_name in packages:
        spec = importlib.util.find_spec(package_name)
        if spec:
            debug_print(f"{package_name} found at: {spec.origin}")
            try:
                module = importlib.import_module(package_name)
                if hasattr(module, "__version__"):
                    debug_print(f"{package_name} version: {module.__version__}")
            except ImportError as e:
                debug_print(f"Error importing {package_name}: {e}", "ERROR")
        else:
            debug_print(f"{package_name} not found", "WARNING")

    return True


def check_hardware_detection_module():
    """Check if the hardware detection module can be imported and used."""
    debug_print("Checking hardware detection module...")

    try:
        from tools.hardware.hardware_detection import (
            detect_hardware_type,
            get_cpu_info,
            get_gpu_info,
            get_hardware_info,
            is_openvino_available,
        )

        debug_print("Successfully imported hardware detection module")

        # Check hardware detection functions
        try:
            hw_type = detect_hardware_type()
            debug_print(f"Detected hardware type: {hw_type}")

            gpus = get_gpu_info()
            debug_print(f"Detected GPUs: {gpus}")

            cpu_info = get_cpu_info()
            debug_print(f"Detected CPU: {cpu_info}")

            openvino_available = is_openvino_available()
            debug_print(f"OpenVINO available: {openvino_available}")

            hardware_info = get_hardware_info()
            debug_print(f"Hardware info: {hardware_info}")

            return True
        except Exception as e:
            debug_print(f"Error calling hardware detection functions: {e}", "ERROR")
            import traceback

            traceback.print_exc()
            return False
    except ImportError as e:
        debug_print(f"Error importing hardware detection module: {e}", "ERROR")

        # Check if the module file exists
        module_path = Path("tools/hardware/hardware_detection.py")
        if module_path.exists():
            debug_print(f"Module file exists at {module_path}")
            # Check Python path
            debug_print(f"sys.path: {sys.path}")
            # Check if tools is a package
            tools_init = Path("tools/__init__.py")
            if tools_init.exists():
                debug_print("tools/__init__.py exists")
            else:
                debug_print("tools/__init__.py does not exist", "WARNING")

            # Check if tools/hardware is a package
            hardware_init = Path("tools/hardware/__init__.py")
            if hardware_init.exists():
                debug_print("tools/hardware/__init__.py exists")
            else:
                debug_print("tools/hardware/__init__.py does not exist", "WARNING")
        else:
            debug_print(f"Module file does not exist at {module_path}", "ERROR")

        return False


def check_pytest():
    """Check if pytest can find and run the hardware tests."""
    debug_print("Checking pytest for hardware tests...")

    try:
        import pytest

        test_path = Path("tests/hardware")
        if test_path.exists():
            debug_print(f"Test directory exists: {test_path}")
            test_files = list(test_path.glob("test_*.py"))
            if test_files:
                debug_print(f"Found test files: {[f.name for f in test_files]}")
                # Don't actually run the tests, just list them
                debug_print("Collecting tests (without running)...")
                try:
                    # Use pytest to collect tests without running them
                    args = ["-xvs", "--collect-only", "tests/hardware/"]
                    exit_code = pytest.main(args)
                    debug_print(f"pytest collection exit code: {exit_code}")
                    return exit_code == 0
                except Exception as e:
                    debug_print(f"Error collecting tests: {e}", "ERROR")
                    return False
            else:
                debug_print("No test files found", "WARNING")
                return False
        else:
            debug_print(f"Test directory does not exist: {test_path}", "ERROR")
            return False
    except ImportError as e:
        debug_print(f"Error importing pytest: {e}", "ERROR")
        return False


def main():
    """Run verification checks."""
    debug_print("Starting hardware detection verification...")
    debug_print(f"Python version: {platform.python_version()}")
    debug_print(f"Platform: {platform.platform()}")
    debug_print(f"Current directory: {os.getcwd()}")

    # Run checks
    env_vars = check_environment()
    packages_ok = check_python_packages()
    module_ok = check_hardware_detection_module()
    pytest_ok = check_pytest()

    # Print summary
    debug_print("\nVerification Summary:")
    debug_print(
        f"Environment: {'OK' if env_vars.get('SIMULATED_HARDWARE') != 'Not set' else 'WARNING'}",
    )
    debug_print(f"Python Packages: {'OK' if packages_ok else 'WARNING'}")
    debug_print(f"Hardware Detection Module: {'OK' if module_ok else 'ERROR'}")
    debug_print(f"Pytest: {'OK' if pytest_ok else 'WARNING'}")

    # Exit with appropriate code
    if module_ok:
        debug_print("Verification completed successfully")
        return 0
    debug_print("Verification failed", "ERROR")
    return 1


if __name__ == "__main__":
    sys.exit(main())
