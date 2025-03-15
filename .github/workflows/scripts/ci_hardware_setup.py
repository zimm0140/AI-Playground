#!/usr/bin/env python3
"""
CI Hardware Setup Script

This script handles CI-specific hardware setup tasks:
1. Creates essential directories and __init__.py files
2. Installs mock packages for hardware detection
3. Copies test fixtures to standard locations
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def debug_print(message, level="INFO"):
    """Print debug information with a timestamp."""
    print(f"[CI_HARDWARE_SETUP:{level}] {message}")


def ensure_directory_structure():
    """Ensure all required directories exist with __init__.py files."""
    debug_print("Ensuring directory structure...")

    # Create essential directories
    dirs_to_create = [
        "tools",
        "tools/hardware",
        "tests",
        "tests/hardware",
        "tests/hardware/mocks",
        "tests/hardware/mocks/openvino-dummy",
        "tests/hardware/mocks/openvino-dummy/openvino",
        "tests/hardware/mocks/intel-gpu-stub",
        "tests/hardware/mocks/intel-gpu-stub/intel_gpu",
        ".uvfast",
        ".uvfast/mock",
    ]

    for dir_path in dirs_to_create:
        dir_obj = Path(dir_path)
        if not dir_obj.exists():
            dir_obj.mkdir(parents=True, exist_ok=True)
            debug_print(f"Created directory: {dir_path}")

        # Add __init__.py to Python package directories
        if dir_path.startswith("tools") or dir_path.startswith("tests"):
            init_file = dir_obj / "__init__.py"
            if not init_file.exists():
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write(f"# Auto-generated {dir_path} package\n")
                debug_print(f"Created {init_file}")

    return True


def create_mock_packages():
    """Create mock packages for hardware detection."""
    debug_print("Setting up mock packages...")

    # Create OpenVINO mock package
    openvino_dir = Path("tests/hardware/mocks/openvino-dummy/openvino")
    openvino_init = openvino_dir / "__init__.py"

    with open(openvino_init, "w", encoding="utf-8") as f:
        f.write('"""Mock OpenVINO package for testing."""\n\n')
        f.write('__version__ = "2023.1.0"\n\n\n')
        f.write("class Core:\n")
        f.write('    """Mock OpenVINO Core class."""\n\n')
        f.write("    def __init__(self, *args, **kwargs):\n")
        f.write('        """Initialize the mock Core."""\n')
        f.write('        self.devices = ["CPU"]\n\n')
        f.write("    def get_versions(self, device_name=None):\n")
        f.write('        """Get mock versions."""\n')
        f.write('        return {"CPU": {"major": "2023", "minor": "1", "patch": "0"}}\n\n')
        f.write('    def compile_model(self, model, device="CPU", *args, **kwargs):\n')
        f.write('        """Compile a mock model."""\n')
        f.write("        return CompiledModel()\n\n\n")
        f.write("class CompiledModel:\n")
        f.write('    """Mock CompiledModel class."""\n\n')
        f.write("    def __init__(self):\n")
        f.write('        """Initialize the mock CompiledModel."""\n')
        f.write("        pass\n\n")
        f.write("    def infer(self, inputs):\n")
        f.write('        """Run mock inference."""\n')
        f.write('        return {"output": [1.0, 2.0, 3.0]}\n\n\n')
        f.write("def get_available_devices():\n")
        f.write('    """Return mock available devices."""\n')
        f.write('    return ["CPU"]\n')

    # Create OpenVINO setup.py
    openvino_setup = Path("tests/hardware/mocks/openvino-dummy/setup.py")
    with open(openvino_setup, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""Setup script for openvino-dummy package."""\n\n')
        f.write("from setuptools import setup, find_packages\n\n")
        f.write("setup(\n")
        f.write('    name="openvino-dummy",\n')
        f.write('    version="2023.1.0",\n')
        f.write('    description="Dummy OpenVINO package for testing",\n')
        f.write('    author="Test Author",\n')
        f.write('    author_email="test@example.com",\n')
        f.write("    packages=find_packages(),\n")
        f.write('    python_requires=">=3.8",\n')
        f.write(")\n")

    # Create Intel GPU mock package
    intel_gpu_dir = Path("tests/hardware/mocks/intel-gpu-stub/intel_gpu")
    intel_gpu_init = intel_gpu_dir / "__init__.py"

    with open(intel_gpu_init, "w", encoding="utf-8") as f:
        f.write('"""Mock Intel GPU package for testing."""\n\n')
        f.write('__version__ = "1.0.0"\n\n\n')
        f.write("def get_device_info():\n")
        f.write('    """Return mock device information."""\n')
        f.write("    return {\n")
        f.write('        "name": "Intel(R) Arc(TM) A770 Graphics",\n')
        f.write('        "vendor": "Intel",\n')
        f.write('        "memory": 16384,  # MB\n')
        f.write('        "compute_units": 32,\n')
        f.write("    }\n\n\n")
        f.write("def is_available():\n")
        f.write('    """Check if Intel GPU is available (always returns True for mock)."""\n')
        f.write("    return True\n\n\n")
        f.write("def get_device_count():\n")
        f.write('    """Return mock device count."""\n')
        f.write("    return 1\n\n\n")
        f.write("class Device:\n")
        f.write('    """Mock Intel GPU Device class."""\n\n')
        f.write("    def __init__(self, device_id=0):\n")
        f.write('        """Initialize the mock Device."""\n')
        f.write("        self.id = device_id\n")
        f.write('        self.name = "Intel(R) Arc(TM) A770 Graphics"\n\n')
        f.write("    def get_info(self):\n")
        f.write('        """Get mock device info."""\n')
        f.write("        return get_device_info()\n\n")
        f.write("    def synchronize(self):\n")
        f.write('        """Mock synchronize method."""\n')
        f.write("        pass\n")

    # Create Intel GPU setup.py
    intel_gpu_setup = Path("tests/hardware/mocks/intel-gpu-stub/setup.py")
    with open(intel_gpu_setup, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""Setup script for intel-gpu-stub package."""\n\n')
        f.write("from setuptools import setup, find_packages\n\n")
        f.write("setup(\n")
        f.write('    name="intel-gpu-stub",\n')
        f.write('    version="1.0.0",\n')
        f.write('    description="Dummy Intel GPU package for testing",\n')
        f.write('    author="Test Author",\n')
        f.write('    author_email="test@example.com",\n')
        f.write("    packages=find_packages(),\n")
        f.write('    python_requires=">=3.8",\n')
        f.write(")\n")

    debug_print("Mock packages created successfully")
    return True


def install_mock_packages():
    """Install mock packages for hardware detection."""
    debug_print("Installing mock packages...")

    packages_to_install = [
        "tests/hardware/mocks/openvino-dummy",
        "tests/hardware/mocks/intel-gpu-stub",
    ]

    for package in packages_to_install:
        if Path(package).exists():
            try:
                debug_print(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", package],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                debug_print(f"Installed {package}: {result.stdout.strip()}")
            except subprocess.SubprocessError as e:
                debug_print(f"Error installing {package}: {e}", "ERROR")
                if e.stdout:
                    debug_print(f"STDOUT: {e.stdout}")
                if e.stderr:
                    debug_print(f"STDERR: {e.stderr}")
        else:
            debug_print(f"Package directory not found: {package}", "WARNING")

    # Verify installation
    try:
        debug_print("Verifying OpenVINO installation...")
        openvino_spec = importlib.util.find_spec("openvino")
        if openvino_spec:
            debug_print(f"OpenVINO found at: {openvino_spec.origin}")
            import openvino

            debug_print(f"OpenVINO version: {openvino.__version__}")
        else:
            debug_print("OpenVINO not found", "WARNING")

        debug_print("Verifying Intel GPU installation...")
        intel_gpu_spec = importlib.util.find_spec("intel_gpu")
        if intel_gpu_spec:
            debug_print(f"Intel GPU found at: {intel_gpu_spec.origin}")
            import intel_gpu

            debug_print(f"Intel GPU version: {intel_gpu.__version__}")
        else:
            debug_print("Intel GPU not found", "WARNING")
    except ImportError as e:
        debug_print(f"Error importing packages: {e}", "WARNING")

    return True


def setup_hardware_env(hardware_type="base"):
    """Set up the hardware environment for testing."""
    debug_print(f"Setting up hardware environment for: {hardware_type}")

    # Call the hardware environment setup script
    hw_setup_script = Path(".github/workflows/scripts/hardware_env_setup.py")
    if hw_setup_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(hw_setup_script), hardware_type, "--verbose"],
                check=True,
                capture_output=True,
                text=True,
            )
            debug_print(f"Hardware environment setup: {result.stdout.strip()}")

            # Set environment variables from the JSON file
            mock_dir = Path(".uvfast/mock")
            env_file = mock_dir / "env.json"
            if env_file.exists():
                with open(env_file, encoding="utf-8") as f:
                    env_info = json.load(f)
                    for key, value in env_info.items():
                        if key.isupper():  # Only set environment variables
                            os.environ[key] = str(value)
                            debug_print(f"Set environment variable: {key}={value}")
        except subprocess.SubprocessError as e:
            debug_print(f"Error setting up hardware environment: {e}", "ERROR")
            if e.stdout:
                debug_print(f"STDOUT: {e.stdout}")
            if e.stderr:
                debug_print(f"STDERR: {e.stderr}")
            return False
    else:
        debug_print(f"Hardware environment setup script not found: {hw_setup_script}", "ERROR")
        return False

    return True


def main():
    """Main function to run CI hardware setup tasks."""
    debug_print("Running CI hardware setup tasks...")

    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Set up hardware environment for CI")
    parser.add_argument(
        "--hardware-type",
        choices=["base", "acm", "ovino"],
        default=os.environ.get("SIMULATED_HARDWARE", "base"),
        help="Type of hardware to simulate",
    )

    args = parser.parse_args()

    # Run setup tasks
    if not ensure_directory_structure():
        debug_print("Failed to ensure directory structure", "ERROR")
        return 1

    if not create_mock_packages():
        debug_print("Failed to create mock packages", "ERROR")
        return 1

    if not install_mock_packages():
        debug_print("Failed to install mock packages", "ERROR")
        return 1

    if not setup_hardware_env(args.hardware_type):
        debug_print(f"Failed to set up hardware environment for {args.hardware_type}", "ERROR")
        return 1

    debug_print("CI hardware setup completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())