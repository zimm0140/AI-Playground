#!/usr/bin/env python3
"""Hardware detection module for uvfast.py.

This module provides functions to detect Intel hardware, especially GPUs and specialized
processors that require specific Python packages for optimal performance.
"""

import contextlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

# Define hardware types
HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]

# Add version for easier debugging
__version__ = "1.0.1"


def load_config() -> dict:
    """Load uvfast configuration from uvfast.json."""
    # Try multiple locations for uvfast.json
    possible_paths = [
        Path("uvfast.json"),  # Current directory
        Path(__file__).parent / "uvfast.json",  # Same directory as this module
        Path(__file__).parent.parent / "uvfast.json",  # Parent directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                return config
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Error loading config from {config_path}: {e}")

    # Default config if no config file is found
    return {"hardware_types": HARDWARE_TYPES, "default_hardware": "base"}


def get_gpu_info() -> list:
    """Get GPU information."""
    # Check for simulated environment in CI
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw == "acm":
            return ["Intel(R) Arc(TM) A770 Graphics"]
        elif sim_hw == "ovino":
            return ["Intel(R) UHD Graphics"]

    # Check for mock files
    mock_dir = Path(os.environ.get("UVFAST_MOCK_DIR", ".uvfast/mock"))
    mock_gpu_file = mock_dir / "gpu_info.txt"
    if mock_gpu_file.exists():
        try:
            with open(mock_gpu_file) as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except OSError:
            pass

    # Platform-specific GPU detection
    system = platform.system()
    gpus = []

    try:
        if system == "Windows":
            # Use wmic on Windows
            output = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout
            gpus = [line.strip() for line in output.split("\n")[1:] if line.strip()]
        elif system == "Linux":
            # Try lspci on Linux
            output = subprocess.run(
                ["lspci", "-v"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout

            # Extract GPU names from lspci output
            gpu_lines = []
            for line in output.split("\n"):
                if "VGA" in line or "3D" in line or "Display" in line:
                    gpu_lines.append(line)

            gpus = [line.split(":")[2].strip() if len(line.split(":")) > 2 else line for line in gpu_lines]
        elif system == "Darwin":
            # Use system_profiler on macOS
            output = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout

            # Extract GPU names from system_profiler output
            gpu_lines = []
            for line in output.split("\n"):
                if "Chipset Model:" in line:
                    gpu_lines.append(line.split(":")[1].strip())

            gpus = gpu_lines
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
        pass

    return gpus


def get_cpu_info() -> dict:
    """Get CPU information."""
    info = {
        "vendor": "",
        "name": "",
        "cores": 0,
    }

    # Check for mock files
    mock_dir = Path(os.environ.get("UVFAST_MOCK_DIR", ".uvfast/mock"))
    mock_cpu_file = mock_dir / "cpu_info.txt"
    if mock_cpu_file.exists():
        try:
            with open(mock_cpu_file) as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in info:
                            info[key] = value
            return info
        except OSError:
            pass

    # Platform-specific CPU detection
    system = platform.system()

    try:
        if system == "Windows":
            # Use wmic on Windows
            for key, wmic_key in [
                ("vendor", "Manufacturer"),
                ("name", "Name"),
                ("cores", "NumberOfCores"),
            ]:
                output = subprocess.run(
                    ["wmic", "cpu", "get", wmic_key],
                    capture_output=True,
                    text=True,
                    timeout=5, check=False,
                ).stdout
                value = output.split("\n")[1].strip()
                info[key] = value
        elif system == "Linux":
            # Parse /proc/cpuinfo on Linux
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo") as f:
                    content = f.read()

                    # Extract vendor
                    vendor_match = re.search(r"vendor_id\s*:\s*([^\n]+)", content)
                    if vendor_match:
                        info["vendor"] = vendor_match.group(1).strip()

                    # Extract model name
                    name_match = re.search(r"model name\s*:\s*([^\n]+)", content)
                    if name_match:
                        info["name"] = name_match.group(1).strip()

                    # Count cores
                    cores = content.count("processor")
                    if cores > 0:
                        info["cores"] = cores
        elif system == "Darwin":
            # Use sysctl on macOS
            vendor_output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.vendor"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout
            info["vendor"] = vendor_output.strip()

            name_output = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout
            info["name"] = name_output.strip()

            cores_output = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5, check=False,
            ).stdout
            with contextlib.suppress(ValueError):
                info["cores"] = int(cores_output.strip())
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
        pass

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
        if cpu_pattern and "name" in cpu_info and re.search(cpu_pattern, cpu_info["name"], re.IGNORECASE):
            return hw_type

    # Default to base if no specific hardware is detected
    return config.get("default_hardware", "base")


def is_openvino_available() -> bool:
    """Check if OpenVINO is installed and available."""
    # For simulated environments
    if os.environ.get("SIMULATED_HARDWARE") == "ovino":
        return True

    try:
        # Use importlib.util to check for package without import warning
        import importlib.util

        return importlib.util.find_spec("openvino") is not None
    except (ImportError, AttributeError):
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


def print_hardware_info(verbose=False):
    """Print hardware information."""
    info = get_hardware_info()

    print(f"System: {info['system']}")
    print(f"Python version: {info['python_version']}")
    print(f"Detected hardware type: {info['detected_hardware']}")

    print("\nGPUs:")
    if info["gpus"]:
        for gpu in info["gpus"]:
            print(f"  - {gpu}")
    else:
        print("  No GPUs detected")

    print("\nCPU:")
    cpu = info["cpu"]
    print(f"  Vendor: {cpu.get('vendor', 'Unknown')}")
    print(f"  Model: {cpu.get('name', 'Unknown')}")
    print(f"  Cores: {cpu.get('cores', 'Unknown')}")

    print(f"\nOpenVINO available: {info['openvino_available']}")

    if verbose:
        print("\nEnvironment Variables:")
        for var in sorted(os.environ):
            if var.startswith(("PYTHON", "PATH", "SIMULATED", "UVFAST", "HARDWARE")):
                print(f"  {var}={os.environ[var]}")


# If run directly, print hardware information
if __name__ == "__main__":
    print_hardware_info(verbose=True)
