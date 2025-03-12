#!/usr/bin/env python3
"""Hardware detection module for uvfast.py.

This module provides functions to detect Intel hardware, especially GPUs and specialized
processors that require specific Python packages for optimal performance.
"""

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# Define hardware types
HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]


def load_config() -> dict:
    """Load uvfast configuration from uvfast.json."""
    config_path = Path("uvfast.json")
    if not config_path.exists():
        return {"hardware_types": HARDWARE_TYPES, "default_hardware": "base"}

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_gpu_info_windows() -> list[str]:
    """Get GPU information on Windows using WMI."""
    # Check for mock environment first
    mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    if mock_dir:
        mock_path = Path(mock_dir) / "gpu_info.txt"
        if mock_path.exists():
            with open(mock_path) as f:
                return [line.strip() for line in f.readlines() if line.strip()]

    try:
        import wmi  # type: ignore

        w = wmi.WMI()
        return [gpu.Name for gpu in w.Win32_VideoController()]
    except ImportError:
        # If wmi is not available, try using subprocess
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                universal_newlines=True,
            )
            lines = output.strip().split("\n")[1:]
            return [line.strip() for line in lines if line.strip()]
        except (subprocess.SubprocessError, FileNotFoundError):
            return []


def get_gpu_info_linux() -> list[str]:
    """Get GPU information on Linux using lspci."""
    # Check for mock environment first
    mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    if mock_dir:
        mock_path = Path(mock_dir) / "gpu_info.txt"
        if mock_path.exists():
            with open(mock_path) as f:
                return [line.strip() for line in f.readlines() if line.strip()]

    try:
        output = subprocess.check_output(["lspci", "-v"], universal_newlines=True)
        gpu_lines = [line for line in output.split("\n") if "VGA" in line or "Display" in line]
        return gpu_lines
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_gpu_info_macos() -> list[str]:
    """Get GPU information on macOS using system_profiler."""
    # Check for mock environment first
    mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    if mock_dir:
        mock_path = Path(mock_dir) / "gpu_info.txt"
        if mock_path.exists():
            with open(mock_path) as f:
                return [line.strip() for line in f.readlines() if line.strip()]

    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], universal_newlines=True
        )
        chip_lines = [line for line in output.split("\n") if "Chipset Model" in line]
        return [line.split(":")[1].strip() for line in chip_lines]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_gpu_info() -> list[str]:
    """Get GPU information for the current platform."""
    # Allow direct override through environment variable for CI/testing
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw == "acm":
            return ["Intel(R) Arc(TM) A770 Graphics"]
        elif sim_hw == "ovino":
            return ["Intel(R) UHD Graphics"]

    system = platform.system()
    if system == "Windows":
        return get_gpu_info_windows()
    elif system == "Linux":
        return get_gpu_info_linux()
    elif system == "Darwin":
        return get_gpu_info_macos()
    else:
        return []


def get_cpu_info() -> dict[str, str]:
    """Get CPU information."""
    info = {}

    system = platform.system()
    if system == "Windows":
        try:
            import wmi  # type: ignore

            w = wmi.WMI()
            for processor in w.Win32_Processor():
                info["name"] = processor.Name
                info["manufacturer"] = processor.Manufacturer
                break
        except ImportError:
            # Fall back to platform module
            info["name"] = platform.processor()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["name"] = line.split(":")[1].strip()
                        break
        except (OSError, FileNotFoundError):
            info["name"] = platform.processor()
    else:
        info["name"] = platform.processor()

    return info


def detect_hardware_type() -> str:
    """Detect the hardware type based on GPU and CPU information."""
    # Allow direct override through environment variable for CI/testing
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw in HARDWARE_TYPES:
            return sim_hw

    config = load_config()
    detection_config = config.get("detection", {})

    gpus = get_gpu_info()
    cpu_info = get_cpu_info()

    # Check for specific hardware types in order of priority
    for hw_type in HARDWARE_TYPES:
        if hw_type == "base":
            continue

        hw_config = detection_config.get(hw_type, {})

        # Check GPU name pattern
        gpu_pattern = hw_config.get("gpu_name_pattern")
        if gpu_pattern and gpus:
            for gpu in gpus:
                if re.search(gpu_pattern, gpu, re.IGNORECASE):
                    return hw_type

        # Check CPU name pattern
        cpu_pattern = hw_config.get("cpu_name_pattern")
        if cpu_pattern and "name" in cpu_info:
            if re.search(cpu_pattern, cpu_info["name"], re.IGNORECASE):
                return hw_type

    # Default to base if no specific hardware is detected
    return config.get("default_hardware", "base")


def is_openvino_available() -> bool:
    """Check if OpenVINO is installed and available."""
    # For simulated environments
    if os.environ.get("SIMULATED_HARDWARE") == "ovino":
        return True

    try:
        import openvino  # type: ignore

        return True
    except ImportError:
        return False


def get_hardware_info() -> dict:
    """Get detailed hardware information for reporting."""
    info = {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "gpus": get_gpu_info(),
        "cpu": get_cpu_info(),
        "detected_hardware": detect_hardware_type(),
        "openvino_available": is_openvino_available(),
    }
    return info


def get_requirements_file(hardware_type: str, dev: bool = False) -> str:
    """Get the appropriate requirements file path for the hardware type."""
    config = load_config()

    requirements_config = config.get("requirements", {})

    # Check for hardware-specific requirements
    if hardware_type != "base" and "hardware" in requirements_config:
        hw_req = requirements_config.get("hardware", {}).get(hardware_type)
        if hw_req:
            if dev and requirements_config.get("dev"):
                return [hw_req, requirements_config.get("dev")]
            return hw_req

    # Fall back to base requirements
    base_req = requirements_config.get("base", "requirements.txt")
    if dev and requirements_config.get("dev"):
        return [base_req, requirements_config.get("dev")]
    return base_req


def print_hardware_info(verbose: bool = False) -> None:
    """Print hardware information to the console."""
    info = get_hardware_info()

    print(f"System: {info['system']}")
    print(f"Python version: {info['python_version']}")
    print(f"Detected hardware type: {info['detected_hardware']}")

    print("GPUs:")
    if info["gpus"]:
        for gpu in info["gpus"]:
            print(f"  - {gpu}")
    else:
        print("  No GPUs detected")

    print(f"CPU: {info['cpu'].get('name', 'Unknown')}")
    print(f"OpenVINO available: {info['openvino_available']}")

    if verbose:
        config = load_config()
        print("\nConfiguration:")
        print(f"  Hardware types: {config.get('hardware_types', HARDWARE_TYPES)}")
        print(f"  Default hardware: {config.get('default_hardware', 'base')}")

        # Print environment variables for debugging
        if "UVFAST_MOCK_DIR" in os.environ:
            print(f"\nMock directory: {os.environ.get('UVFAST_MOCK_DIR')}")
        if "SIMULATED_HARDWARE" in os.environ:
            print(f"Simulated hardware: {os.environ.get('SIMULATED_HARDWARE')}")


if __name__ == "__main__":
    verbose_flag = "--verbose" in sys.argv or "-v" in sys.argv
    print_hardware_info(verbose=verbose_flag)
