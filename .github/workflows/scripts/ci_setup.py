#!/usr/bin/env python3
"""
CI Environment Setup Script

This script performs necessary setup for CI environments:
1. Creates symlinks or copies of key files for backward compatibility
2. Fixes import paths to work with the new directory structure
3. Sets up mocked hardware detection for testing
"""

import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def debug_print(message, level="INFO"):
    """Print debug information with a timestamp."""
    print(f"[CI_SETUP:{level}] {message}", flush=True)


def ensure_file_in_root(source_path, target_name=None):
    """Copy a file to the root directory if it doesn't exist there."""
    root_dir = Path(".")
    source = Path(source_path)

    if not source.exists():
        debug_print(f"Warning: Source file {source} does not exist", "WARNING")
        return False

    target = root_dir / (target_name or source.name)

    if not target.exists():
        debug_print(f"Copying {source} to {target}")
        try:
            shutil.copy2(source, target)
            return True
        except Exception as e:
            debug_print(f"Error copying {source} to {target}: {e}", "ERROR")
            traceback.print_exc()
            return False
    else:
        debug_print(f"File already exists: {target}")
        return False


def setup_hardware_detection():
    """Ensure hardware detection module is available for imports."""
    debug_print("Setting up hardware detection module...")

    # First check for the new package structure
    if Path("hardware_detection").exists() and Path("hardware_detection").is_dir():
        debug_print("Found hardware_detection package, using package structure")

        # Create necessary directories
        for subdir in ["", "tests"]:
            init_file = Path("hardware_detection") / subdir / "__init__.py"
            if init_file.exists():
                debug_print(f"Found {init_file}")
            else:
                debug_print(f"Creating {init_file}")
                init_dir = init_file.parent
                init_dir.mkdir(parents=True, exist_ok=True)
                with open(init_file, "w") as f:
                    if subdir == "":
                        f.write(
                            '''"""Hardware detection package for identifying and utilizing specialized hardware."""

__version__ = "1.0.0"

from hardware_detection.core import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
    print_hardware_info,
)

__all__ = [
    "detect_hardware_type",
    "get_cpu_info",
    "get_gpu_info",
    "get_hardware_info",
    "is_openvino_available",
    "print_hardware_info",
]
'''
                        )
                    else:
                        f.write('"""Auto-generated hardware_detection package."""\n')

        # Create py.typed file for type hints
        py_typed_file = Path("hardware_detection/py.typed")
        if not py_typed_file.exists():
            debug_print(f"Creating {py_typed_file}")
            with open(py_typed_file, "w") as f:
                f.write("")  # Empty file is sufficient

        # Set up the compatibility layer
        tools_hw_dir = Path("tools/hardware")
        tools_hw_dir.mkdir(parents=True, exist_ok=True)

        # Create the compatibility layer if it doesn't exist
        compat_file = tools_hw_dir / "hardware_detection.py"
        if not compat_file.exists():
            debug_print(f"Creating compatibility layer at {compat_file}")
            with open(compat_file, "w") as f:
                f.write(
                    '''#!/usr/bin/env python3
"""Compatibility module for hardware detection.

This module provides backward compatibility with the previous file-based structure.
It imports all functions from the new package structure and re-exports them.

IMPORTANT: This module is maintained for backward compatibility only.
New code should import directly from the hardware_detection package.
"""

import os
import sys
import warnings
import importlib.util
from typing import Dict, List, Any, Optional, Union

# Check if hardware_detection package exists
PACKAGE_EXISTS = importlib.util.find_spec("hardware_detection") is not None

if PACKAGE_EXISTS:
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

else:
    # If package is not found, add helpful error message
    # and implement basic stubs for CI
    __version__ = "0.0.0-stub"
    
    # Log a more helpful message about the missing package
    print(
        "WARNING: hardware_detection package not found. "
        "Using stub implementations for CI environment. "
        "For production, please install the hardware_detection package."
    )

    def detect_hardware_type() -> str:
        """Stub function for hardware type detection."""
        # Default to 'base' for CI environments
        return os.environ.get("SIMULATED_HARDWARE", "base")

    def get_gpu_info() -> List[str]:
        """Stub function for GPU info."""
        return ["Stub GPU for CI"]

    def get_cpu_info() -> Dict[str, Any]:
        """Stub function for CPU info."""
        return {
            "vendor": "Stub",
            "name": "Stub CPU for CI",
            "cores": 2,
        }

    def is_openvino_available() -> bool:
        """Stub function for OpenVINO availability."""
        return False

    def get_hardware_info() -> Dict[str, Any]:
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
'''
                )

        # Create __init__.py in tools/hardware
        tools_hw_init = tools_hw_dir / "__init__.py"
        if not tools_hw_init.exists():
            debug_print(f"Creating {tools_hw_init}")
            with open(tools_hw_init, "w") as f:
                f.write('"""Hardware detection compatibility package."""\n')

        debug_print("Hardware detection setup complete")
        return True

    # Legacy setup for older structure
    # Copy from tools/hardware if it exists there
    if Path("tools/hardware/hardware_detection.py").exists():
        debug_print("Found hardware_detection.py in tools/hardware/")
        ensure_file_in_root("tools/hardware/hardware_detection.py")
    else:
        debug_print(
            "Warning: hardware_detection.py not found in tools/hardware/", "WARNING"
        )
        # Try to find the hardware_detection.py file elsewhere
        for path in [
            "hardware_detection.py",
            "tools/scripts/hardware_detection.py",
            "scripts/hardware_detection.py",
        ]:
            if Path(path).exists():
                debug_print(f"Found hardware_detection.py at {path}")
                ensure_file_in_root(path, "hardware_detection.py")
                break
        else:
            debug_print("Error: Could not find hardware_detection.py", "ERROR")
            return False

    return True


def setup_mock_hardware():
    """Set up mock hardware detection for testing."""
    debug_print("Setting up mock hardware...")

    mock_dir = Path(".uvfast/mock")
    try:
        mock_dir.mkdir(parents=True, exist_ok=True)

        # Create mock GPU info file
        with open(mock_dir / "gpu_info.txt", "w") as f:
            f.write("Intel(R) Arc(TM) A770 Graphics")

        # Create mock CPU info file
        with open(mock_dir / "cpu_info.txt", "w") as f:
            f.write("vendor: Intel\n")
            f.write("name: Intel(R) Core(TM) i9-13900K\n")
            f.write("cores: 24\n")

        debug_print(f"Created mock hardware files in {mock_dir}")
    except Exception as e:
        debug_print(f"Error setting up mock hardware: {e}", "ERROR")
        traceback.print_exc()


def create_requirements_files():
    """Create any missing requirements files referenced in uvfast.json."""
    debug_print("Creating requirements files...")

    requirements_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-hardware-base.txt",
        "requirements-hardware-ovino.txt",
        "requirements-hardware-acm.txt",
        "requirements-hardware-arl_h.txt",
        "requirements-hardware-bmg.txt",
        "requirements-hardware-mtl.txt",
        "requirements-hardware-lnl.txt",
        "service/requirements-acm.txt",
        "service/requirements-bmg.txt",
        "service/requirements-mtl.txt",
        "service/requirements-lnl.txt",
        "service/requirements-arl_h.txt",
        "service/requirements-ls_level_zero.txt",
    ]

    base_content = (
        "# Base requirements for testing\ntorch>=2.0.0\nnumpy>=1.24.0\npytest>=7.0.0\n"
    )

    for req_file in requirements_files:
        req_path = Path(req_file)
        try:
            if not req_path.exists():
                debug_print(f"Creating requirements file: {req_file}")
                req_path.parent.mkdir(parents=True, exist_ok=True)
                with open(req_path, "w") as f:
                    f.write(base_content)

                    # Add specialized content for specific files
                    if "ovino" in req_file:
                        f.write("\n# OpenVINO requirements\nopenvino-stub>=1.0.0\n")
                    elif "acm" in req_file or "Arc" in req_file:
                        f.write(
                            "\n# Intel Arc requirements\nintel-extension-for-pytorch>=2.0.0\n"
                        )
        except Exception as e:
            debug_print(f"Error creating requirements file {req_file}: {e}", "ERROR")
            traceback.print_exc()


def setup_openvino_stub():
    """Create a stub for OpenVINO to use in CI environments."""
    debug_print("Setting up OpenVINO stub...")

    openvino_stub_path = Path("openvino")

    # If openvino is not already installed, create a stub
    try:
        import openvino

        debug_print(f"OpenVINO already installed: {openvino.__file__}")
        return
    except ImportError:
        debug_print("OpenVINO not found, creating stub")

    try:
        # Create the stub module
        openvino_stub_path.mkdir(exist_ok=True)

        # Copy the stub implementation
        stub_source = Path(".github/workflows/scripts/openvino_stub.py")
        if stub_source.exists():
            shutil.copy2(stub_source, openvino_stub_path / "__init__.py")
            debug_print("Installed OpenVINO stub module from file")
        else:
            # Create a simple stub if the source file doesn't exist
            with open(openvino_stub_path / "__init__.py", "w") as f:
                f.write('"""OpenVINO stub module for CI testing."""\n\n')
                f.write('version = "STUB.2023.0.0"\n\n')
                f.write("def Core(*args, **kwargs):\n")
                f.write('    """Mock Core class."""\n')
                f.write(
                    '    return type("Runtime", (), {"compile_model": lambda *a, **k: None})()\n'
                )
            debug_print("Created simple OpenVINO stub module")

        # Create empty __pycache__ to avoid warnings
        (openvino_stub_path / "__pycache__").mkdir(exist_ok=True)

        # Ensure the stub is in the Python path
        sys.path.append(str(Path(".").absolute()))
        debug_print(f"Added {Path('.').absolute()} to Python path")
    except Exception as e:
        debug_print(f"Error setting up OpenVINO stub: {e}", "ERROR")
        traceback.print_exc()


def setup_intel_extension_stub():
    """Create a stub for Intel Extension for PyTorch to use in CI environments."""
    debug_print("Setting up Intel Extension for PyTorch stub...")

    intel_stub_path = Path("intel_extension_for_pytorch")

    # If intel_extension_for_pytorch is not already installed, create a stub
    try:
        import intel_extension_for_pytorch

        debug_print(
            f"Intel Extension for PyTorch already installed: {intel_extension_for_pytorch.__file__}"
        )
        return
    except ImportError:
        debug_print("Intel Extension for PyTorch not found, creating stub")

    try:
        # Create the stub module
        intel_stub_path.mkdir(exist_ok=True)

        # Copy the stub implementation
        stub_source = Path(
            ".github/workflows/scripts/intel_extension_for_pytorch_stub.py"
        )
        if stub_source.exists():
            shutil.copy2(stub_source, intel_stub_path / "__init__.py")
            debug_print("Installed Intel Extension for PyTorch stub module from file")
        else:
            # Create a simple stub if the source file doesn't exist
            with open(intel_stub_path / "__init__.py", "w") as f:
                f.write(
                    '"""Intel Extension for PyTorch stub module for CI testing."""\n\n'
                )
                f.write('__version__ = "2.0.110+mock"\n\n')
                f.write("def xpu_device_name():\n")
                f.write('    """Get the XPU device name."""\n')
                f.write('    return "Intel Arc A770 Graphics (Mock)"\n\n')
                f.write("def optimize():\n")
                f.write('    """Mock optimize function."""\n')
                f.write("    return None\n")
            debug_print("Created simple Intel Extension for PyTorch stub module")

        # Create empty __pycache__ to avoid warnings
        (intel_stub_path / "__pycache__").mkdir(exist_ok=True)

        # Create xpu subdirectory
        xpu_dir = intel_stub_path / "xpu"
        xpu_dir.mkdir(exist_ok=True)

        # Create xpu/__init__.py
        with open(xpu_dir / "__init__.py", "w") as f:
            f.write('"""Intel Extension for PyTorch XPU module stub."""\n\n')
            f.write("def device_count():\n    return 1\n\n")
            f.write("def is_available():\n    return True\n")

        debug_print(f"Created Intel Extension XPU module at {xpu_dir}")
    except Exception as e:
        debug_print(f"Error setting up Intel Extension stub: {e}", "ERROR")
        traceback.print_exc()


def ensure_module_structure():
    """Ensure all Python package directories have proper __init__.py files."""
    debug_print("Ensuring module structure...")

    module_structure_script = Path(".github/workflows/scripts/init_module_structure.py")

    if module_structure_script.exists():
        # Run the script directly
        try:
            debug_print("Running module structure initialization script...")
            result = subprocess.run(
                [sys.executable, str(module_structure_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            debug_print(result.stdout)
            return True
        except subprocess.SubprocessError as e:
            debug_print(f"Error running module structure script: {e}", "ERROR")
            if hasattr(e, "stdout"):
                debug_print(f"STDOUT: {e.stdout}")
            if hasattr(e, "stderr"):
                debug_print(f"STDERR: {e.stderr}")
    else:
        debug_print(
            "Module structure script not found, creating minimal structure...",
            "WARNING",
        )

        # Ensure tools directory structure
        for dir_path in ["tools", "tools/hardware", "tools/scripts", "service"]:
            try:
                dir_init = Path(dir_path) / "__init__.py"
                if not Path(dir_path).exists():
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    debug_print(f"Created directory: {dir_path}")

                if not dir_init.exists():
                    with open(dir_init, "w") as f:
                        f.write(f'"""Auto-generated {dir_path} package."""\n')
                    debug_print(f"Created {dir_init}")
            except Exception as e:
                debug_print(
                    f"Error setting up module structure for {dir_path}: {e}", "ERROR"
                )
                traceback.print_exc()

    return False


def setup_mock_modules():
    """Set up mock modules for hardware-dependent packages."""
    debug_print("Setting up mock modules...")

    mock_modules_script = Path(".github/workflows/scripts/setup_mock_modules.py")

    if mock_modules_script.exists():
        # Run the script directly
        try:
            debug_print("Running mock modules setup script...")
            result = subprocess.run(
                [sys.executable, str(mock_modules_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            debug_print(result.stdout)
            return True
        except subprocess.SubprocessError as e:
            debug_print(f"Error running mock modules script: {e}", "ERROR")
            if hasattr(e, "stdout"):
                debug_print(f"STDOUT: {e.stdout}")
            if hasattr(e, "stderr"):
                debug_print(f"STDERR: {e.stderr}")
            return False
    else:
        debug_print(
            "Mock modules script not found, using individual setup functions...",
            "WARNING",
        )
        return setup_openvino_stub() and setup_intel_extension_stub()


def setup_uvfast():
    """Ensure uvfast.py is properly configured for CI."""
    debug_print("Setting up uvfast...")

    uvfast_path = Path("uvfast.py")
    if not uvfast_path.exists():
        debug_print("Warning: uvfast.py not found!", "WARNING")
        return False

    try:
        # Make sure uvfast.json exists
        uvfast_json_path = Path("uvfast.json")
        if not uvfast_json_path.exists():
            debug_print("Creating minimal uvfast.json")
            with open(uvfast_json_path, "w") as f:
                f.write(
                    """{
    "project_name": "ai-playground",
    "python_version": "3.10",
    "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
    "default_hardware": "base",
    "requirements": {
        "base": "requirements.txt",
        "dev": "requirements-dev.txt",
        "hardware": {
            "base": "requirements.txt",
            "acm": "service/requirements-acm.txt",
            "bmg": "service/requirements-bmg.txt",
            "mtl": "service/requirements-mtl.txt",
            "lnl": "service/requirements-lnl.txt",
            "ovino": "requirements-hardware-ovino.txt",
            "arl_h": "service/requirements-arl_h.txt"
        }
    },
    "lockfiles": {
        "base": "requirements.lock",
        "dev": "requirements-dev.lock",
        "hardware": {
            "base": "requirements-hardware-base.lock",
            "acm": "requirements-hardware-acm.lock",
            "bmg": "requirements-hardware-bmg.lock",
            "mtl": "requirements-hardware-mtl.lock",
            "lnl": "requirements-hardware-lnl.lock",
            "ovino": "requirements-hardware-ovino.lock",
            "arl_h": "requirements-hardware-arl_h.lock"
        }
    }
}"""
                )

        return True
    except Exception as e:
        debug_print(f"Error setting up uvfast: {e}", "ERROR")
        traceback.print_exc()
        return False


def main():
    """Main entry point for CI setup."""
    debug_print("Starting CI setup...")

    # Set up hardware detection first
    if not setup_hardware_detection():
        debug_print("Failed to set up hardware detection", "ERROR")
        sys.exit(1)

    debug_print("CI setup completed successfully")


if __name__ == "__main__":
    main()
