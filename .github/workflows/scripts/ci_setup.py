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
from pathlib import Path


def ensure_file_in_root(source_path, target_name=None):
    """Copy a file to the root directory if it doesn't exist there."""
    root_dir = Path(".")
    source = Path(source_path)

    if not source.exists():
        print(f"Warning: Source file {source} does not exist")
        return False

    target = root_dir / (target_name or source.name)

    if not target.exists():
        print(f"Copying {source} to {target}")
        shutil.copy2(source, target)
        return True
    else:
        print(f"File already exists: {target}")
        return False


def setup_hardware_detection():
    """Ensure hardware detection module is available for imports."""
    # Copy from tools/hardware if it exists there
    if Path("tools/hardware/hardware_detection.py").exists():
        print("Setting up hardware_detection.py from tools/hardware/")
        ensure_file_in_root("tools/hardware/hardware_detection.py")

    # Create __init__.py files to make directories importable
    for dir_path in ["tools", "tools/hardware"]:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            print(f"Creating {init_file}")
            init_file.touch()


def setup_mock_hardware():
    """Set up mock hardware detection for testing."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Create mock GPU info file
    with open(mock_dir / "gpu_info.txt", "w") as f:
        f.write("Intel(R) Arc(TM) A770 Graphics")

    print(f"Created mock hardware files in {mock_dir}")


def create_requirements_files():
    """Create any missing requirements files referenced in uvfast.json."""
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
        if not req_path.exists():
            print(f"Creating requirements file: {req_file}")
            req_path.parent.mkdir(parents=True, exist_ok=True)
            with open(req_path, "w") as f:
                f.write(base_content)

                # Add specialized content for specific files
                if "ovino" in req_file:
                    f.write("\n# OpenVINO requirements\nopenvino-stub>=1.0.0\n")
                elif "acm" in req_file or "Arc" in req_file:
                    f.write("\n# Intel Arc requirements\nintel-extension-for-pytorch>=2.0.0\n")


def setup_openvino_stub():
    """Create a stub for OpenVINO to use in CI environments."""
    openvino_stub_path = Path("openvino")

    # If openvino is not already installed, create a stub
    try:
        import openvino

        print("OpenVINO already installed:", openvino.__file__)
        return
    except ImportError:
        pass

    # Create the stub module
    openvino_stub_path.mkdir(exist_ok=True)

    # Copy the stub implementation
    stub_source = Path(".github/workflows/scripts/openvino_stub.py")
    if stub_source.exists():
        shutil.copy2(stub_source, openvino_stub_path / "__init__.py")
        print("Installed OpenVINO stub module")
    else:
        # Create a simple stub if the source file doesn't exist
        with open(openvino_stub_path / "__init__.py", "w") as f:
            f.write('"""OpenVINO stub module for CI testing."""\n\n')
            f.write('version = "STUB.2023.0.0"\n\n')
            f.write("def Core(*args, **kwargs):\n")
            f.write('    """Mock Core class."""\n')
            f.write('    return type("Runtime", (), {"compile_model": lambda *a, **k: None})()\n')
        print("Created simple OpenVINO stub module")

    # Create empty __pycache__ to avoid warnings
    (openvino_stub_path / "__pycache__").mkdir(exist_ok=True)

    # Ensure the stub is in the Python path
    sys.path.append(str(Path(".").absolute()))


def setup_intel_extension_stub():
    """Create a stub for Intel Extension for PyTorch to use in CI environments."""
    intel_stub_path = Path("intel_extension_for_pytorch")

    # If intel_extension_for_pytorch is not already installed, create a stub
    try:
        import intel_extension_for_pytorch

        print(
            "Intel Extension for PyTorch already installed:", intel_extension_for_pytorch.__file__
        )
        return
    except ImportError:
        pass

    # Create the stub module
    intel_stub_path.mkdir(exist_ok=True)

    # Copy the stub implementation
    stub_source = Path(".github/workflows/scripts/intel_extension_for_pytorch_stub.py")
    if stub_source.exists():
        shutil.copy2(stub_source, intel_stub_path / "__init__.py")
        print("Installed Intel Extension for PyTorch stub module")
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
        print("Created simple Intel Extension for PyTorch stub module")

    # Create empty __pycache__ to avoid warnings
    (intel_stub_path / "__pycache__").mkdir(exist_ok=True)


def ensure_module_structure():
    """Ensure all Python package directories have proper __init__.py files."""
    module_structure_script = Path(".github/workflows/scripts/init_module_structure.py")

    if module_structure_script.exists():
        # Run the script directly
        try:
            print("Running module structure initialization script...")
            result = subprocess.run(
                [sys.executable, str(module_structure_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            return True
        except subprocess.SubprocessError as e:
            print(f"Error running module structure script: {e}")
            if hasattr(e, "stdout"):
                print(e.stdout)
            if hasattr(e, "stderr"):
                print(e.stderr)
    else:
        print("Module structure script not found, creating minimal structure...")

        # Ensure tools directory structure
        for dir_path in ["tools", "tools/hardware", "tools/scripts"]:
            dir_init = Path(dir_path) / "__init__.py"
            if not dir_init.exists():
                dir_init.parent.mkdir(parents=True, exist_ok=True)
                with open(dir_init, "w") as f:
                    f.write(f'"""Auto-generated {dir_path} package."""\n')
                print(f"Created {dir_init}")

    return False


def setup_mock_modules():
    """Set up mock modules for hardware-dependent packages."""
    mock_modules_script = Path(".github/workflows/scripts/setup_mock_modules.py")

    if mock_modules_script.exists():
        # Run the script directly
        try:
            print("Running mock modules setup script...")
            result = subprocess.run(
                [sys.executable, str(mock_modules_script)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            return True
        except subprocess.SubprocessError as e:
            print(f"Error running mock modules script: {e}")
            if hasattr(e, "stdout"):
                print(e.stdout)
            if hasattr(e, "stderr"):
                print(e.stderr)
            return False
    else:
        print("Mock modules script not found, using individual setup functions...")
        return setup_openvino_stub() and setup_intel_extension_stub()


def main():
    """Main function to run all setup tasks."""
    print("Setting up CI environment...")

    # Ensure proper module structure with __init__.py files
    ensure_module_structure()

    # Ensure hardware detection is available
    setup_hardware_detection()

    # Set up mock hardware for testing
    setup_mock_hardware()

    # Create any missing requirements files
    create_requirements_files()

    # Set up all mock modules (including OpenVINO and Intel Extension)
    setup_mock_modules()

    print("CI environment setup complete")


if __name__ == "__main__":
    main()
