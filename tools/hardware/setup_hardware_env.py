#!/usr/bin/env python3
"""Setup script for hardware-specific environments.

This script detects the available hardware and sets up the appropriate Python
environment with the necessary dependencies for optimal performance.
"""

import argparse
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Import the hardware detection module
try:
    import hardware_detection
except ImportError:
    print("Error: hardware_detection.py not found in the current directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up a hardware-specific Python environment")
    parser.add_argument(
        "--hardware",
        choices=hardware_detection.HARDWARE_TYPES,
        help="Specify the hardware type manually (auto-detect if not specified)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Include development dependencies",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing environment before setup",
    )
    parser.add_argument(
        "--venv-dir",
        default=".venv",
        help="Virtual environment directory (default: .venv)",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Use uv for package installation (faster)",
    )
    parser.add_argument(
        "--skip-hardware-check",
        action="store_true",
        help="Skip hardware availability check",
    )
    return parser.parse_args()


def is_uv_available():
    """Check if uv is available."""
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def is_venv_available():
    """Check if venv module is available."""
    try:
        return True
    except ImportError:
        return False


def create_venv(venv_dir, clean=False):
    """Create a virtual environment."""
    venv_path = Path(venv_dir)

    # Clean existing environment if requested
    if clean and venv_path.exists():
        logging.info(f"Removing existing environment at {venv_path}")
        shutil.rmtree(venv_path)

    # Create virtual environment
    if not venv_path.exists():
        logging.info(f"Creating virtual environment at {venv_path}")
        import venv

        venv.create(venv_path, with_pip=True)
    else:
        logging.info(f"Using existing virtual environment at {venv_path}")

    return venv_path


def get_python_executable(venv_dir):
    """Get the Python executable path for the virtual environment."""
    if platform.system() == "Windows":
        return Path(venv_dir) / "Scripts" / "python.exe"
    return Path(venv_dir) / "bin" / "python"


def install_requirements(python_executable, hardware_type, dev=False, use_uv=False):
    """Install requirements for the specified hardware type."""
    # Get requirements file path based on hardware type
    requirements_files = hardware_detection.get_requirements_file(hardware_type, dev)

    if not isinstance(requirements_files, list):
        requirements_files = [requirements_files]

    # Check if all requirements files exist
    for req_file in requirements_files:
        if not Path(req_file).exists():
            logging.info(f"Error: Requirements file {req_file} not found")
            sys.exit(1)

    # Install requirements
    for req_file in requirements_files:
        logging.info(f"Installing requirements from {req_file}")

        if use_uv and is_uv_available():
            # Use uv for faster installation
            cmd = [str(python_executable), "-m", "pip", "install", "--upgrade", "uv"]
            subprocess.run(cmd, check=True)
        else:
            # Use pip
            cmd = [str(python_executable), "-m", "pip", "install", "-r", req_file]
            subprocess.run(cmd, check=True)

    logging.info("Requirements installation completed successfully")


def check_hardware_availability(hardware_type):
    """Check if the specified hardware is available."""
    if hardware_type == "base":
        return True

    # Get detected hardware type
    detected_type = hardware_detection.detect_hardware_type()

    if hardware_type != detected_type:
        logging.warning(
            f"Warning: Requested hardware type '{hardware_type}' does not match detected type '{detected_type}'",
        )
        logging.warning("This might cause issues with hardware-specific dependencies")
        return False

    return True


def main():
    """Main function."""
    args = parse_args()

    # Detect hardware type if not specified
    hardware_type = args.hardware or hardware_detection.detect_hardware_type()
    logging.info(f"Setting up environment for hardware type: {hardware_type}")

    # Check hardware availability
    if not args.skip_hardware_check and not check_hardware_availability(hardware_type):
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != "y":
            logging.info("Aborting setup")
            sys.exit(1)

    # Check venv availability
    if not is_venv_available():
        logging.error("Error: venv module not available. Please install it or use a Python version with venv support.")
        sys.exit(1)

    # Create virtual environment
    create_venv(args.venv_dir, args.clean)
    python_executable = get_python_executable(args.venv_dir)

    # Install requirements
    install_requirements(python_executable, hardware_type, args.dev, args.use_uv)

    # Print activation instructions
    logging.info("\nEnvironment setup complete!")
    logging.info("To activate the environment:")
    if platform.system() == "Windows":
        logging.info(f"    {args.venv_dir}\\Scripts\\activate")
    else:
        logging.info(f"    source {args.venv_dir}/bin/activate")

    # Print hardware info
    logging.info("\nHardware information:")
    hardware_detection.print_hardware_info()


if __name__ == "__main__":
    main()