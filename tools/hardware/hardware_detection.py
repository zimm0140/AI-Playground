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
from typing import Any

# Define hardware types
HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]

# Add version for easier debugging
__version__ = "1.0.2"


def debug_print(message: str, level: str = "INFO") -> None:
    """Print debug information with a prefix."""
    print(f"[HARDWARE_DETECTION:{level}] {message}")


def safe_run_command(command: list[str], timeout: int = 5) -> str:
    """Safely run a command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # Don't raise on non-zero exit code
        )
        return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError) as e:
        debug_print(f"Error running command {command}: {e}", "WARNING")
        return ""


def load_config() -> dict[str, Any]:
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
                    debug_print(f"Loaded config from {config_path}")
                    return config
            except (json.JSONDecodeError, OSError) as e:
                debug_print(f"Error loading config from {config_path}: {e}", "WARNING")

    # Default config if no config file is found
    debug_print("Using default config", "WARNING")
    return {"hardware_types": HARDWARE_TYPES, "default_hardware": "base"}


def get_gpu_info() -> list[str]:
    """Get GPU information."""
    # Check for simulation in CI environments
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        debug_print(f"Using simulated hardware: {sim_hw}")
        if sim_hw == "acm":
            return ["Intel(R) Arc(TM) A770 Graphics (Simulated)"]
        elif sim_hw == "ovino":
            return ["Intel(R) UHD Graphics (Simulated)"]
        return []

    # Check for mock files in CI environments
    mock_dir = Path(os.environ.get("UVFAST_MOCK_DIR", ".uvfast/mock"))
    mock_gpu_file = mock_dir / "gpu_info.txt"

    if mock_gpu_file.exists():
        try:
            with open(mock_gpu_file, encoding="utf-8") as f:
                gpus = [line.strip() for line in f.readlines() if line.strip()]
                debug_print(f"Using mock GPU info: {gpus}")
                return gpus
        except OSError as e:
            debug_print(f"Error reading mock GPU file: {e}", "WARNING")

    # Platform-specific GPU detection
    system = platform.system()
    gpus = []

    if system == "Windows":
        # Windows: Use WMIC to get GPU information
        output = safe_run_command(["wmic", "path", "win32_VideoController", "get", "Name"])
        gpus = [line.strip() for line in output.split("\n")[1:] if line.strip()]

    elif system == "Linux":
        # Linux: Try lspci
        output = safe_run_command(["lspci", "-v"])
        gpu_lines = []

        for line in output.split("\n"):
            if any(term in line for term in ["VGA", "3D", "Display"]):
                gpu_lines.append(line)

        gpus = [
            line.split(":", 2)[2].strip() if len(line.split(":", 2)) > 2 else line
            for line in gpu_lines
        ]

    elif system == "Darwin":
        # macOS: Use system_profiler
        output = safe_run_command(["system_profiler", "SPDisplaysDataType"])
        gpu_lines = []

        for line in output.split("\n"):
            if "Chipset Model:" in line:
                gpu_lines.append(line.split(":", 1)[1].strip())

        gpus = gpu_lines

    debug_print(f"Detected GPUs: {gpus}")
    return gpus


def get_cpu_info() -> dict[str, Any]:
    """Get CPU information."""
    info = {
        "vendor": "",
        "name": "",
        "cores": 0,
    }

    # Check for simulation in CI environments
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw == "acm":
            info = {
                "vendor": "Intel",
                "name": "Intel(R) Core(TM) i9-13900K (Simulated)",
                "cores": 24,
            }
        elif sim_hw == "ovino":
            info = {
                "vendor": "Intel",
                "name": "Intel(R) Core(TM) i7-1370P (Simulated)",
                "cores": 16,
            }
        else:
            info = {"vendor": "Intel", "name": "Intel(R) Core(TM) i5-10400 (Simulated)", "cores": 6}
        debug_print(f"Using simulated CPU info: {info}")
        return info

    # Check for mock files in CI environments
    mock_dir = Path(os.environ.get("UVFAST_MOCK_DIR", ".uvfast/mock"))
    mock_cpu_file = mock_dir / "cpu_info.txt"

    if mock_cpu_file.exists():
        try:
            with open(mock_cpu_file, encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in info:
                            info[key] = value
            debug_print(f"Using mock CPU info: {info}")
            return info
        except OSError as e:
            debug_print(f"Error reading mock CPU file: {e}", "WARNING")

    # Platform-specific CPU detection
    system = platform.system()

    if system == "Windows":
        # Windows: Use WMIC
        vendor_output = safe_run_command(["wmic", "cpu", "get", "Manufacturer"])
        vendor = vendor_output.split("\n")[1].strip() if "\n" in vendor_output else ""

        name_output = safe_run_command(["wmic", "cpu", "get", "Name"])
        name = name_output.split("\n")[1].strip() if "\n" in name_output else ""

        cores_output = safe_run_command(["wmic", "cpu", "get", "NumberOfCores"])
        cores_str = cores_output.split("\n")[1].strip() if "\n" in cores_output else "0"
        cores = int(cores_str) if cores_str.isdigit() else 0

        info = {"vendor": vendor, "name": name, "cores": cores}

    elif system == "Linux":
        # Linux: Parse /proc/cpuinfo
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
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
            except OSError as e:
                debug_print(f"Error reading /proc/cpuinfo: {e}", "WARNING")

    elif system == "Darwin":
        # macOS: Use sysctl
        vendor_output = safe_run_command(["sysctl", "-n", "machdep.cpu.vendor"])
        info["vendor"] = vendor_output.strip()

        name_output = safe_run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["name"] = name_output.strip()

        cores_output = safe_run_command(["sysctl", "-n", "hw.physicalcpu"])
        try:
            info["cores"] = int(cores_output.strip())
        except ValueError:
            pass

    debug_print(f"Detected CPU info: {info}")
    return info


def detect_hardware_type() -> str:
    """Detect the hardware type based on GPU and CPU information."""
    # Allow direct override through environment variable for CI/testing
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw in HARDWARE_TYPES:
            debug_print(f"Using simulated hardware type: {sim_hw}")
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
                    debug_print(f"Detected hardware type from GPU: {hw_type}")
                    return hw_type

        # Check CPU name pattern
        cpu_pattern = hw_config.get("cpu_name_pattern")
        if cpu_pattern and "name" in cpu_info:
            if re.search(cpu_pattern, cpu_info["name"], re.IGNORECASE):
                debug_print(f"Detected hardware type from CPU: {hw_type}")
                return hw_type

    # Default to base if no specific hardware is detected
    default_hw = config.get("default_hardware", "base")
    debug_print(f"No specific hardware detected, using default: {default_hw}")
    return default_hw


def is_openvino_available() -> bool:
    """Check if OpenVINO is installed and available."""
    # For simulated environments
    if os.environ.get("SIMULATED_HARDWARE") == "ovino":
        debug_print("Simulated OpenVINO environment")
        return True

    # Direct import check
    try:
        debug_print("Checking for OpenVINO package")
        import openvino

        debug_print(f"OpenVINO found: {openvino.__file__}")
        return True
    except ImportError:
        debug_print("OpenVINO not found")
        return False


def get_hardware_info() -> dict[str, Any]:
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


def print_hardware_info(verbose: bool = False) -> None:
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


if __name__ == "__main__":
    # When run directly, print hardware information
    print(f"Hardware Detection Module v{__version__}")
    print_hardware_info(verbose=("-v" in sys.argv or "--verbose" in sys.argv))
