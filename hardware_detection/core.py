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
from typing import Any, Dict, List, Optional, Union, cast

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


def safe_run_command(command: List[str], timeout: int = 5) -> str:
    """Safely run a command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            debug_print(f"Command {command} failed with code {result.returncode}", "WARNING")
            return ""
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        debug_print(f"Error running command {command}: {e}", "ERROR")
        return ""


def load_config() -> Dict[str, Any]:
    """Load configuration for hardware detection."""
    # Default config with hardware detection rules
    default_config = {
        "hardware_types": {
            "acm": {"gpu_patterns": ["Intel.*Arc"]},
            "bmg": {"gpu_patterns": ["Intel.*Graphics.*Software"]},
            "mtl": {"cpu_patterns": ["Core.*1xx"]},
            "lnl": {"cpu_patterns": ["Core.*2xx"]},
            "ovino": {"package": "openvino"},
            "arl_h": {"gpu_patterns": ["Intel.*Arc.*High"]},
        },
    }

    # Load from package directory first
    config_path = Path(__file__).parent / "hardware_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            debug_print(f"Error loading config from {config_path}: {e}", "WARNING")

    return default_config


def _get_simulated_gpu_info() -> List[str]:
    """Get simulated GPU info for testing."""
    if "SIMULATED_GPU" in os.environ:
        return [os.environ["SIMULATED_GPU"]]
    return []


def _get_mock_gpu_info() -> Optional[List[str]]:
    """Get mock GPU info for testing."""
    if "MOCK_HARDWARE_TEST" in os.environ:
        return ["Intel Arc Graphics (Simulated)"]
    if os.path.exists("mock_gpu_info.txt"):
        try:
            with open("mock_gpu_info.txt", "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except IOError:
            pass
    return None


def _get_windows_gpu_info() -> List[str]:
    """Get GPU info on Windows."""
    output = safe_run_command(["wmic", "path", "win32_VideoController", "get", "Name"])
    return [line.strip() for line in output.split("\n")[1:] if line.strip()]


def _get_linux_gpu_info() -> List[str]:
    """Get GPU info on Linux."""
    # Try lspci first
    output = safe_run_command(["lspci", "-v"])
    if output:
        return [line for line in output.split("\n") if "VGA" in line or "3D" in line]
    
    # Fallback to lshw
    output = safe_run_command(["lshw", "-C", "display"])
    return [line.strip() for line in output.split("\n") if "product" in line]


def _get_macos_gpu_info() -> List[str]:
    """Get GPU info on macOS."""
    output = safe_run_command(["system_profiler", "SPDisplaysDataType"])
    gpu_info = []
    
    for line in output.split("\n"):
        if "Chipset Model" in line:
            gpu_info.append(line.split(":", 1)[1].strip())
            
    return gpu_info


def get_gpu_info() -> List[str]:
    """Get information about available GPUs.
    
    Returns:
        List of GPU information strings
    """
    # Check if we're using simulated GPU for testing
    simulated = _get_simulated_gpu_info()
    if simulated:
        debug_print(f"Using simulated GPU: {simulated}")
        return simulated
    
    # Check if we're using mock GPU for testing
    mock_gpu = _get_mock_gpu_info()
    if mock_gpu is not None:
        debug_print(f"Using mock GPU: {mock_gpu}")
        return mock_gpu
    
    # Get GPU info based on platform
    system = platform.system()
    
    if system == "Windows":
        return _get_windows_gpu_info()
    elif system == "Linux":
        return _get_linux_gpu_info()
    elif system == "Darwin":  # macOS
        return _get_macos_gpu_info()
    else:
        debug_print(f"Unsupported platform: {system}", "WARNING")
        return []


def _get_simulated_cpu_info() -> Dict[str, Any]:
    """Get simulated CPU info for testing."""
    if "SIMULATED_CPU" in os.environ:
        cpu_name = os.environ["SIMULATED_CPU"]
        
        # Create a simulated CPU info
        return {
            "name": cpu_name,
            "cores": 16,
            "threads": 32,
            "architecture": "x86_64",
            "frequency_mhz": 3500,
            "features": ["avx2", "sse4", "aes"],
            "vendor": "Intel" if "Intel" in cpu_name else "AMD" if "AMD" in cpu_name else "Unknown",
        }
    
    return {"name": "Simulated CPU"}


def _get_mock_cpu_info() -> Dict[str, Any]:
    """Get mock CPU info for testing."""
    if "MOCK_HARDWARE_TEST" in os.environ:
        return {
            "name": "Intel Core i9-1000K (Simulated)",
            "cores": 16,
            "threads": 32,
            "architecture": "x86_64",
            "frequency_mhz": 3500,
            "features": ["avx2", "sse4", "aes"],
            "vendor": "Intel",
        }
    
    if os.path.exists("mock_cpu_info.json"):
        try:
            with open("mock_cpu_info.json", "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            debug_print(f"Error loading mock CPU info: {e}", "WARNING")
            return {}
        
    # If we're in a test environment but no specific mock data,
    # return None to indicate we should use real detection
    return {}


def _get_windows_cpu_info() -> Dict[str, Union[str, int]]:
    """Get CPU info on Windows."""
    output = safe_run_command(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors"])
    
    if not output:
        return {"name": "Unknown CPU", "error": "Failed to retrieve CPU info"}
    
    lines = [line for line in output.split("\n") if line.strip()]
    if len(lines) < 2:
        return {"name": "Unknown CPU", "error": "Invalid CPU info format"}
    
    # Parse the output (format: Name NumberOfCores NumberOfLogicalProcessors)
    cpu_info = lines[1].split("  ")
    cpu_info = [info for info in cpu_info if info.strip()]
    
    if not cpu_info:
        return {"name": "Unknown CPU", "error": "Empty CPU info"}
    
    result: Dict[str, Union[str, int]] = {"name": cpu_info[0].strip()}
    
    # Add cores and threads if available
    if len(cpu_info) > 1:
        result["cores"] = int(cpu_info[1].strip())
    if len(cpu_info) > 2:
        result["threads"] = int(cpu_info[2].strip())
    
    return result


def _get_linux_cpu_info() -> Dict[str, Union[str, int]]:
    """Get CPU info on Linux."""
    result: Dict[str, Union[str, int]] = {}
    
    # Try lscpu first for structured data
    output = safe_run_command(["lscpu"])
    if output:
        for line in output.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                
                if key == "model_name":
                    result["name"] = value
                elif key == "architecture":
                    result["architecture"] = value
                elif key == "cpu(s)":
                    result["threads"] = int(value)
                elif key == "core(s)_per_socket" and "socket(s)" in output:
                    # Try to calculate physical cores
                    for socket_line in output.split("\n"):
                        if "socket(s)" in socket_line:
                            sockets = int(socket_line.split(":", 1)[1].strip())
                            cores_per_socket = int(value)
                            result["cores"] = sockets * cores_per_socket
                            break
    
    # Fallback to /proc/cpuinfo if necessary
    if not result:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                cpuinfo = f.read()
                
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    result["name"] = line.split(":", 1)[1].strip()
                    break
        except IOError:
            pass
    
    # Ensure we have at least a CPU name
    if "name" not in result:
        result["name"] = "Unknown Linux CPU"
        
    return result


def _get_macos_cpu_info() -> Dict[str, Union[str, int]]:
    """Get CPU info on macOS."""
    output = safe_run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    
    result: Dict[str, Union[str, int]] = {"name": output if output else "Unknown macOS CPU"}
    
    # Get core count
    cores_output = safe_run_command(["sysctl", "-n", "hw.physicalcpu"])
    if cores_output and cores_output.isdigit():
        result["cores"] = int(cores_output)
    
    # Get thread count
    threads_output = safe_run_command(["sysctl", "-n", "hw.logicalcpu"])
    if threads_output and threads_output.isdigit():
        result["threads"] = int(threads_output)
    
    return result


def get_cpu_info() -> Dict[str, Any]:
    """Get information about the CPU.
    
    Returns:
        Dictionary with CPU information
    """
    # Check if we're using simulated CPU for testing
    if "SIMULATED_CPU" in os.environ:
        simulated = _get_simulated_cpu_info()
        debug_print(f"Using simulated CPU: {simulated['name']}")
        return simulated
    
    # Check if we're using mock CPU for testing
    mock_cpu = _get_mock_cpu_info()
    if mock_cpu:
        debug_print(f"Using mock CPU: {mock_cpu.get('name', 'Unknown')}")
        return mock_cpu
    
    # Get CPU info based on platform
    system = platform.system()
    
    if system == "Windows":
        cpu_info = _get_windows_cpu_info()
    elif system == "Linux":
        cpu_info = _get_linux_cpu_info()
    elif system == "Darwin":  # macOS
        cpu_info = _get_macos_cpu_info()
    else:
        debug_print(f"Unsupported platform: {system}", "WARNING")
        cpu_info = {"name": f"Unknown CPU ({system})"}
    
    # Extract vendor - typically Intel or AMD
    cpu_name = str(cpu_info.get("name", ""))
    if "Intel" in cpu_name:
        cpu_info["vendor"] = "Intel"
    elif "AMD" in cpu_name:
        cpu_info["vendor"] = "AMD"
    else:
        cpu_info["vendor"] = "Unknown"
    
    return cpu_info


def detect_hardware_type() -> str:
    """Detect the hardware type based on available hardware.
    
    Returns:
        Hardware type from HARDWARE_TYPES
    """
    # Load hardware detection configuration
    config = load_config()
    detection_config = config.get("hardware_types", {})
    
    # Get hardware information
    gpus = get_gpu_info()
    cpu_info = get_cpu_info()
    
    debug_print(f"Detected GPUs: {gpus}")
    debug_print(f"Detected CPU: {cpu_info.get('name', 'Unknown')}")
    
    # First check if we're using simulated hardware for testing/development
    simulated = _get_simulated_hardware()
    if simulated:
        debug_print(f"Using simulated hardware type: {simulated}")
        return simulated
    
    # Try to detect specific hardware based on configuration
    detected = _detect_specific_hardware(detection_config, gpus, cpu_info)
    if detected:
        debug_print(f"Detected hardware type: {detected}")
        return detected
    
    # Default to base hardware type
    debug_print("No specific hardware detected, using base configuration")
    return cast(str, "base")


def _get_simulated_hardware() -> Optional[str]:
    """Check if we're using simulated hardware."""
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw in HARDWARE_TYPES:
            return sim_hw
    return None


def _detect_specific_hardware(
    detection_config: Dict[str, Any],
    gpus: List[str],
    cpu_info: Dict[str, Any],
) -> Optional[str]:
    """Detect specific hardware based on configuration.
    
    Args:
        detection_config: Configuration for detecting hardware types
        gpus: List of GPU information strings
        cpu_info: Dictionary with CPU information
        
    Returns:
        Detected hardware type or None if no specific hardware detected
    """
    # Check each hardware type in order of priority
    for hw_type in ["arl_h", "acm", "lnl", "mtl", "bmg", "ovino"]:
        hw_config = detection_config.get(hw_type, {})
        
        # Check GPU match
        if _check_gpu_match(hw_type, hw_config, gpus):
            return hw_type
        
        # Check CPU match
        if _check_cpu_match(hw_type, hw_config, cpu_info):
            return hw_type
        
        # Check package availability
        if _check_package_available(hw_type, hw_config):
            return hw_type
    
    # No specific hardware detected
    return None


def _check_gpu_match(hw_type: str, hw_config: Dict[str, Any], gpus: List[str]) -> bool:
    """Check if GPUs match the patterns for a hardware type."""
    # If no GPU patterns specified or no GPUs detected, no match
    if not hw_config.get("gpu_patterns") or not gpus:
        return False
    
    # Check each pattern against each GPU
    for pattern in hw_config["gpu_patterns"]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        for gpu in gpus:
            if pattern_re.search(gpu):
                debug_print(f"GPU match for {hw_type}: {gpu} matches {pattern}")
                return True
    
    return False


def _check_cpu_match(hw_type: str, hw_config: Dict[str, Any], cpu_info: Dict[str, Any]) -> bool:
    """Check if CPU matches the patterns for a hardware type."""
    # If no CPU patterns specified or no CPU name, no match
    if not hw_config.get("cpu_patterns") or not cpu_info.get("name"):
        return False
    
    cpu_name = cpu_info["name"]
    
    # Check each pattern against CPU name
    for pattern in hw_config["cpu_patterns"]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        if pattern_re.search(cpu_name):
            debug_print(f"CPU match for {hw_type}: {cpu_name} matches {pattern}")
            return True
    
    return False


def _check_package_available(hw_type: str, hw_config: Dict[str, Any]) -> bool:
    """Check if required packages are available for a hardware type."""
    package = hw_config.get("package")
    if not package:
        return False
    
    # Check specific packages
    if package == "openvino":
        if is_openvino_available():
            debug_print(f"Package match for {hw_type}: {package} is available")
            return True
    else:
        # Generic package availability check
        try:
            __import__(package)
            debug_print(f"Package match for {hw_type}: {package} is available")
            return True
        except ImportError:
            pass
    
    return False


def is_openvino_available() -> bool:
    """Check if OpenVINO is available.
    
    Returns:
        True if OpenVINO is available, False otherwise
    """
    # First check for environment variable override for testing
    if "OPENVINO_AVAILABLE" in os.environ:
        return os.environ["OPENVINO_AVAILABLE"].lower() in ("1", "true", "yes")
    
    # Try to import OpenVINO
    try:
        # Handle missing openvino module in type checking
        import openvino  # type: ignore
        return True
    except ImportError:
        # Check for specific OpenVINO detection methods on different platforms
        system = platform.system()
        
        if system == "Windows":
            # Check for OpenVINO installation directory
            program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            openvino_dir = os.path.join(program_files, "Intel", "OpenVINO")
            return os.path.exists(openvino_dir)
        
        return False


def get_hardware_info() -> Dict[str, Any]:
    """Get comprehensive hardware information.
    
    Returns:
        Dictionary with hardware information
    """
    # Get system information
    system = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Get hardware information
    gpus = get_gpu_info()
    cpu_info = get_cpu_info()
    
    # Detect hardware type
    hardware_type = detect_hardware_type()
    
    # Check OpenVINO availability
    openvino_available = is_openvino_available()
    
    # Assemble hardware information
    hardware_info = {
        "system": system,
        "python_version": python_version,
        "platform": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "gpus": gpus,
        "cpu": cpu_info,
        "detected_hardware": hardware_type,
        "openvino_available": openvino_available,
    }
    
    return hardware_info


def print_hardware_info(verbose: bool = False) -> None:
    """Print hardware information to the console.
    
    Args:
        verbose: Whether to print verbose information
    """
    info = get_hardware_info()
    
    print("\n=== Hardware Information ===")
    print(f"System: {info['system']} {info['platform']['release']}")
    print(f"Python version: {info['python_version']}")
    print(f"Detected hardware type: {info['detected_hardware']}")
    
    print("\nGPUs:")
    if info["gpus"]:
        for gpu in info["gpus"]:
            print(f"  - {gpu}")
    else:
        print("  No GPUs detected")
    
    print(f"\nCPU: {info['cpu'].get('name', 'Unknown')}")
    
    if verbose:
        print("\nDetailed CPU Info:")
        for key, value in info["cpu"].items():
            if key != "name":  # Already printed above
                print(f"  {key}: {value}")
        
        print("\nOpenVINO available:", "Yes" if info["openvino_available"] else "No")
        
        print("\nPlatform Info:")
        for key, value in info["platform"].items():
            print(f"  {key}: {value}")
    
    print("\nEnvironment variables affecting hardware detection:")
    env_vars = [
        "SIMULATED_HARDWARE",
        "SIMULATED_GPU",
        "SIMULATED_CPU",
        "OPENVINO_AVAILABLE",
        "MOCK_HARDWARE_TEST",
    ]
    
    found_env = False
    for var in env_vars:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
            found_env = True
    
    if not found_env:
        print("  None")


if __name__ == "__main__":
    # When run directly, print hardware information
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    debug = "-d" in sys.argv or "--debug" in sys.argv

    if debug:
        logger.setLevel(logging.DEBUG)

    print_hardware_info(verbose=verbose)
