#!/usr/bin/env python3
"""Hardware detection module for identifying specialized hardware.

This module provides functions to detect hardware, especially GPUs and specialized
processors that require specific Python packages for optimal performance.
"""

import contextlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[HARDWARE] %(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define hardware types
HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]


def debug_print(message: str, level: str = "INFO") -> None:
    """Print debug information with a prefix.

    Args:
        message: The message to print
        level: Log level (INFO, WARNING, ERROR)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logger.log(level_map.get(level, logging.INFO), message)


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
    """Load configuration from uvfast.json."""
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
    """Get information about available GPUs.

    Detects NVIDIA, AMD, and Intel GPUs using various methods including
    subprocess calls to system utilities and Python package presence.

    Returns:
        List[str]: A list of GPU descriptions or empty list if no GPUs found

    Example:
        >>> get_gpu_info()
        ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3080']
    """
    # Check for simulation in CI environments
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        debug_print(f"Using simulated hardware: {sim_hw}")
        if sim_hw == "acm":
            return ["Intel(R) Arc(TM) A770 Graphics (Simulated)"]
        if sim_hw == "ovino":
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

        gpus = [line.split(":", 2)[2].strip() if len(line.split(":", 2)) > 2 else line for line in gpu_lines]

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
    """Get information about the CPU.

    Retrieves vendor, model name, core count, and additional information
    about the CPU using platform-specific methods.

    Returns:
        Dict[str, Any]: Dictionary containing CPU information with these keys:
            - vendor: CPU manufacturer (str)
            - name: CPU model name (str)
            - cores: Number of CPU cores (int)
            - features: CPU features (list, optional)

    Example:
        >>> get_cpu_info()
        {'vendor': 'Intel', 'name': 'Intel(R) Core(TM) i7-10700K', 'cores': 8}
    """
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
            info = {
                "vendor": "Intel",
                "name": "Intel(R) Core(TM) i5-10400 (Simulated)",
                "cores": 6,
            }
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
                        if key == "cores" and isinstance(info["cores"], str) and info["cores"].isdigit():
                            info["cores"] = int(info["cores"])

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
        with contextlib.suppress(ValueError):
            info["cores"] = int(cores_output.strip())

    debug_print(f"Detected CPU info: {info}")
    return info


def detect_hardware_type() -> str:
    """Detect the available hardware type.

    Analyzes the system to determine the primary hardware acceleration type
    available. Checks for NVIDIA GPUs, AMD GPUs, Intel accelerators,
    and OpenVINO compatibility.

    Returns:
        str: One of the hardware types defined in HARDWARE_TYPES
            ("base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h")

    Example:
        >>> detect_hardware_type()
        'bmg'  # For NVIDIA GPUs
    """
    # Check for simulated hardware type first
    simulated_hw = _get_simulated_hardware()
    if simulated_hw:
        return simulated_hw

    # Get hardware information
    config = load_config()
    detection_config = config.get("detection", {})
    gpus = get_gpu_info()
    cpu_info = get_cpu_info()

    # Check hardware types based on gathered information
    detected_hw = _detect_specific_hardware(detection_config, gpus, cpu_info)
    if detected_hw:
        return detected_hw

    # Default to base if no specific hardware is detected
    default_hw = config.get("default_hardware", "base")
    debug_print(f"No specific hardware detected, using default: {default_hw}")
    return default_hw


def _get_simulated_hardware() -> str | None:
    """Check if simulated hardware is specified in environment variables.

    Returns:
        Optional[str]: The simulated hardware type or None
    """
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw in HARDWARE_TYPES:
            debug_print(f"Using simulated hardware type: {sim_hw}")
            return sim_hw
    return None


def _detect_specific_hardware(
    detection_config: dict[str, Any],
    gpus: list[str],
    cpu_info: dict[str, Any],
) -> str | None:
    """Check for specific hardware types based on detection configuration.

    Args:
        detection_config: The detection configuration dictionary
        gpus: List of detected GPUs
        cpu_info: Dictionary of CPU information

    Returns:
        Optional[str]: Detected hardware type or None if not detected
    """
    # Check for specific hardware types in order of priority
    for hw_type in HARDWARE_TYPES:
        if hw_type == "base":
            continue

        hw_config = detection_config.get(hw_type, {})

        # Check GPU name pattern
        if _check_gpu_match(hw_type, hw_config, gpus):
            return hw_type

        # Check CPU name pattern
        if _check_cpu_match(hw_type, hw_config, cpu_info):
            return hw_type

        # Check for packages
        if _check_package_available(hw_type, hw_config):
            return hw_type

    return None


def _check_gpu_match(hw_type: str, hw_config: dict[str, Any], gpus: list[str]) -> bool:
    """Check if any GPU matches the pattern for this hardware type.

    Args:
        hw_type: The hardware type being checked
        hw_config: Configuration for this hardware type
        gpus: List of detected GPUs

    Returns:
        bool: True if a match is found, False otherwise
    """
    gpu_pattern = hw_config.get("gpu_name_pattern")
    if gpu_pattern and gpus:
        for gpu in gpus:
            if re.search(gpu_pattern, gpu, re.IGNORECASE):
                debug_print(f"Detected hardware type from GPU: {hw_type}")
                return True
    return False


def _check_cpu_match(hw_type: str, hw_config: dict[str, Any], cpu_info: dict[str, Any]) -> bool:
    """Check if CPU matches the pattern for this hardware type.

    Args:
        hw_type: The hardware type being checked
        hw_config: Configuration for this hardware type
        cpu_info: Dictionary of CPU information

    Returns:
        bool: True if a match is found, False otherwise
    """
    cpu_pattern = hw_config.get("cpu_name_pattern")
    if cpu_pattern and "name" in cpu_info and re.search(cpu_pattern, cpu_info["name"], re.IGNORECASE):
        debug_print(f"Detected hardware type from CPU: {hw_type}")
        return True
    return False


def _check_package_available(hw_type: str, hw_config: dict[str, Any]) -> bool:
    """Check if a specific package is available for this hardware type.

    Args:
        hw_type: The hardware type being checked
        hw_config: Configuration for this hardware type

    Returns:
        bool: True if the package is available, False otherwise
    """
    package_check = hw_config.get("package_check")
    if package_check:
        try:
            __import__(package_check)
            debug_print(f"Detected hardware type from package {package_check}: {hw_type}")
            return True
        except ImportError:
            pass
    return False


def is_openvino_available() -> bool:
    """Check if OpenVINO runtime is available.

    Attempts to import the OpenVINO runtime package and checks for
    required components.

    Returns:
        bool: True if OpenVINO is available and usable, False otherwise

    Example:
        >>> is_openvino_available()
        True
    """
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
    """Get comprehensive information about the system hardware.

    Collects information about the system, GPU, CPU, detected hardware type,
    and availability of specific acceleration libraries.

    Returns:
        Dict[str, Any]: Dictionary with hardware information including:
            - system: Operating system (str)
            - python_version: Python version (str)
            - gpus: List of available GPUs (List[str])
            - cpu: CPU information (Dict[str, Any])
            - detected_hardware: Detected hardware type (str)
            - openvino_available: Whether OpenVINO is available (bool)

    Example:
        >>> get_hardware_info()
        {
            'system': 'Linux-5.15.0-x86_64-with-glibc2.31',
            'python_version': '3.10.0',
            'gpus': ['NVIDIA GeForce RTX 3090'],
            'cpu': {'vendor': 'Intel', 'name': 'Intel(R) Core(TM) i7-10700K', 'cores': 8},
            'detected_hardware': 'bmg',
            'openvino_available': False
        }
    """
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
    """Print information about the system hardware.

    Displays system information, Python version, hardware type,
    GPU and CPU details in a human-readable format.

    Args:
        verbose: Whether to show additional details

    Example:
        >>> print_hardware_info(verbose=True)
        System: Linux-5.15.0-x86_64-with-glibc2.31
        Python version: 3.10.0
        Detected hardware type: bmg
        GPUs: ['NVIDIA GeForce RTX 3090']
        CPU: {'vendor': 'Intel', 'name': 'Intel(R) Core(TM) i7-10700K', 'cores': 8}
    """
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
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    debug = "-d" in sys.argv or "--debug" in sys.argv

    if debug:
        logger.setLevel(logging.DEBUG)

    print_hardware_info(verbose=verbose)
