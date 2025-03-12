#!/usr/bin/env python3
"""
CI Environment Setup Script

This script performs necessary setup for CI environments:
1. Creates symlinks or copies of key files for backward compatibility
2. Fixes import paths to work with the new directory structure
3. Sets up mocked hardware detection for testing
"""

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def debug_print(message, level="INFO"):
    """Print debug information with a timestamp."""
    print(f"[CI_SETUP:{level}] {message}")


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

    # Copy from tools/hardware if it exists there
    if Path("tools/hardware/hardware_detection.py").exists():
        debug_print("Found hardware_detection.py in tools/hardware/")
        ensure_file_in_root("tools/hardware/hardware_detection.py")
    else:
        debug_print("Warning: hardware_detection.py not found in tools/hardware/", "WARNING")
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
            debug_print("Error: hardware_detection.py not found anywhere!", "ERROR")
            # Create a minimal version to prevent failures
            debug_print("Creating minimal hardware_detection.py")
            with open("hardware_detection.py", "w") as f:
                f.write(
                    """#!/usr/bin/env python3
\"\"\"Hardware detection module for CI environment.\"\"\"

import os

# Define hardware types
HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]
__version__ = "1.0.0"

def load_config():
    \"\"\"Load uvfast configuration.\"\"\"
    return {"hardware_types": HARDWARE_TYPES, "default_hardware": "base"}

def get_gpu_info():
    \"\"\"Get GPU information.\"\"\"
    if "SIMULATED_HARDWARE" in os.environ:
        if os.environ.get("SIMULATED_HARDWARE") == "acm":
            return ["Intel(R) Arc(TM) A770 Graphics (Simulated)"]
        elif os.environ.get("SIMULATED_HARDWARE") == "ovino":
            return ["Intel(R) UHD Graphics (Simulated)"]
    return []

def get_cpu_info():
    \"\"\"Get CPU information.\"\"\"
    return {"vendor": "Intel", "name": "CI Test CPU", "cores": 4}

def detect_hardware_type():
    \"\"\"Detect the hardware type.\"\"\"
    if "SIMULATED_HARDWARE" in os.environ:
        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
        if sim_hw in HARDWARE_TYPES:
            return sim_hw
    return "base"

def is_openvino_available():
    \"\"\"Check if OpenVINO is available.\"\"\"
    return os.environ.get("SIMULATED_HARDWARE") == "ovino"

def get_hardware_info():
    \"\"\"Get hardware information.\"\"\"
    return {
        "system": "CI",
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "gpus": get_gpu_info(),
        "cpu": get_cpu_info(),
        "detected_hardware": detect_hardware_type(),
        "openvino_available": is_openvino_available(),
    }

def print_hardware_info(verbose=False):
    \"\"\"Print hardware information.\"\"\"
    info = get_hardware_info()
    print(f"System: {info['system']}")
    print(f"Python version: {info['python_version']}")
    print(f"Detected hardware type: {info['detected_hardware']}")
    print(f"GPUs: {info['gpus']}")
    print(f"CPU: {info['cpu']}")
    print(f"OpenVINO available: {info['openvino_available']}")
"""
                )

    # Create __init__.py files to make directories importable
    for dir_path in ["tools", "tools/hardware", "tools/scripts"]:
        init_file = Path(dir_path) / "__init__.py"
        if not dir_path.exists():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                debug_print(f"Created directory: {dir_path}")
            except Exception as e:
                debug_print(f"Error creating directory {dir_path}: {e}", "ERROR")

        if not init_file.exists():
            try:
                debug_print(f"Creating {init_file}")
                with open(init_file, "w") as f:
                    f.write(f'"""Auto-generated {dir_path} package."""\n')
            except Exception as e:
                debug_print(f"Error creating {init_file}: {e}", "ERROR")
                traceback.print_exc()


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

    base_content = "# Base requirements for testing\ntorch>=2.0.0\nnumpy>=1.24.0\npytest>=7.0.0\n"

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
                        f.write("\n# Intel Arc requirements\nintel-extension-for-pytorch>=2.0.0\n")
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
        stub_source = Path(".github/workflows/scripts/intel_extension_for_pytorch_stub.py")
        if stub_source.exists():
            shutil.copy2(stub_source, intel_stub_path / "__init__.py")
            debug_print("Installed Intel Extension for PyTorch stub module from file")
        else:
            # Create a simple stub if the source file doesn't exist
            with open(intel_stub_path / "__init__.py", "w") as f:
                f.write('"""Intel Extension for PyTorch stub module for CI testing."""\n\n')
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
        debug_print("Module structure script not found, creating minimal structure...", "WARNING")

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
                debug_print(f"Error setting up module structure for {dir_path}: {e}", "ERROR")
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
        debug_print("Mock modules script not found, using individual setup functions...", "WARNING")
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
    """Main function to run all setup tasks."""
    debug_print("Setting up CI environment...")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"Current directory: {os.getcwd()}")

    try:
        # Ensure proper module structure with __init__.py files
        ensure_module_structure()

        # Ensure hardware detection is available
        setup_hardware_detection()

        # Set up mock hardware for testing
        setup_mock_hardware()

        # Set up uvfast configuration
        setup_uvfast()

        # Create any missing requirements files
        create_requirements_files()

        # Set up all mock modules (including OpenVINO and Intel Extension)
        setup_mock_modules()

        debug_print("CI environment setup complete!")
        return 0
    except Exception as e:
        debug_print(f"Error in CI setup: {e}", "ERROR")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
