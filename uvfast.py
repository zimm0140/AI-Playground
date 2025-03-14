#!/usr/bin/env python3
"""uvfast: Fast hardware-optimized Python environment manager.

This script provides tools for managing Python environments optimized for
different hardware configurations (Intel Arc GPUs, OpenVINO, etc.).
It leverages the 'uv' package manager for fast dependency installation.
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import hardware detection module, which should be in the same directory
try:
    # Update import path to use the module from tools/hardware
    import os
    import sys
    from pathlib import Path

    # Add tools directory to path if not already there
    tools_dir = Path("tools")
    if tools_dir.exists():
        tools_path = str(tools_dir.absolute())
        if tools_path not in sys.path:
            sys.path.append(tools_path)
            print(f"Added {tools_path} to Python path")

    # Try importing from tools.hardware first
    try:
        from tools.hardware import hardware_detection

        print("Imported hardware_detection from tools.hardware")
    except (ImportError, ModuleNotFoundError):
        # Then try importing from the root
        try:
            import hardware_detection

            print("Imported hardware_detection from root")
        except (ImportError, ModuleNotFoundError):
            print("hardware_detection.py not found, using fallback")

            # Define fallback hardware constants
            HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]

            # Create minimal fallback module for CI
            class FallbackHardwareDetection:
                def __init__(self):
                    self.HARDWARE_TYPES = HARDWARE_TYPES

                def detect_hardware_type(self):
                    """Detect hardware type based on environment variables."""
                    if "SIMULATED_HARDWARE" in os.environ:
                        sim_hw = os.environ.get("SIMULATED_HARDWARE", "").lower()
                        if sim_hw in HARDWARE_TYPES:
                            return sim_hw
                    return "base"

                def print_hardware_info(self, verbose=False):
                    """Print hardware info."""
                    print("System: CI Environment")
                    print(f"Python version: {sys.version}")
                    print(f"Detected hardware type: {self.detect_hardware_type()}")
                    print("GPUs: [Simulated]")
                    print("CPU: Simulated CI CPU")

                def get_hardware_info(self):
                    """Get hardware info."""
                    return {
                        "system": "CI",
                        "python_version": sys.version,
                        "gpus": [],
                        "cpu": {"name": "CI CPU"},
                        "detected_hardware": self.detect_hardware_type(),
                        "openvino_available": self.detect_hardware_type() == "ovino",
                    }

            # Create fallback module
            hardware_detection = FallbackHardwareDetection()
except Exception as e:
    logging.warning(f"Error importing hardware_detection: {e}")
    logging.warning("hardware_detection.py not found, some features will be limited")
    HARDWARE_TYPES = ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]


# Default config values
DEFAULT_CONFIG = {
    "project_name": "ai-playground",
    "python_version": "3.10",
    "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
    "default_hardware": "base",
    "requirements": {
        "base": "requirements.txt",
        "dev": "requirements-dev.txt",
        "hardware": {
            "base": "requirements-hardware-base.txt",
            "acm": "service/requirements-acm.txt",
            "bmg": "service/requirements-bmg.txt",
            "mtl": "service/requirements-mtl.txt",
            "lnl": "service/requirements-lnl.txt",
            "ovino": "requirements-hardware-ovino.txt",
            "arl_h": "service/requirements-arl_h.txt",
        },
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
            "arl_h": "requirements-hardware-arl_h.lock",
        },
    },
    "environment": {
        "venv_path": ".venv",
        "cache_dir": ".cache/uv",
    },
}


class UVFast:
    """Main class for uvfast functionality."""

    def __init__(self):
        """Initialize uvfast with configuration."""
        self.config = self._load_config()

        # Ensure we have hardware detection or use simple detection
        if "hardware_detection" in sys.modules:
            self.hardware_type = hardware_detection.detect_hardware_type()
        else:
            # Simple detection as fallback
            self.hardware_type = self._simple_hardware_detection()

    def _load_config(self) -> dict:
        """Load the configuration from the config file."""
        config_path = Path(".uvfast.json")
        if config_path.exists():
            try:
                with config_path.open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse config file: {config_path}")
                return {}
        else:
            # Create a default config
            return {
                "project_name": "ai-playground",
                "hardware_types": ["intel_arc", "intel_cpu", "openvino", "rocm", "cuda"],
                "requirements": {
                    "base": "requirements.txt",
                    "dev": "requirements-dev.txt",
                    "hardware": {
                        "intel_arc": "requirements-intel-arc.txt",
                        "intel_cpu": "requirements-intel-cpu.txt",
                        "openvino": "requirements-openvino.txt",
                        "rocm": "requirements-rocm.txt",
                        "cuda": "requirements-cuda.txt",
                    },
                },
                "lockfiles": {
                    "base": "requirements.lock",
                    "dev": "requirements-dev.lock",
                    "hardware": {
                        "intel_arc": "requirements-intel-arc.lock",
                        "intel_cpu": "requirements-intel-cpu.lock",
                        "openvino": "requirements-openvino.lock",
                        "rocm": "requirements-rocm.lock",
                        "cuda": "requirements-cuda.lock",
                    },
                },
            }

    def _simple_hardware_detection(self) -> str:
        """Simple hardware detection as a fallback when the module is not available."""
        system = platform.system()

        # Check for GPU presence on Windows
        if system == "Windows":
            try:
                output = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "Name"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                gpu_names = [line.strip() for line in output.split("\n")[1:] if line.strip()]

                # Check for Intel Arc GPU
                for gpu_name in gpu_names:
                    if (
                        "Arc" in gpu_name
                        or "Intel" in gpu_name
                        and any(f"A{num}" in gpu_name for num in range(300, 800))
                    ):
                        return "acm"

            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # Check for OpenVINO
        try:
            subprocess.run(
                [sys.executable, "-c", "import openvino"],
                capture_output=True,
                text=True,
                check=False,
            )
            return "ovino"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Default to base
        return self.config.get("default_hardware", "base")

    def _get_venv_path(self) -> Path:
        """Get the path to the virtual environment."""
        return Path(self.config.get("environment", {}).get("venv_path", ".venv"))

    def _get_python_executable(self) -> Path:
        """Get the path to the Python executable in the virtual environment."""
        venv_path = self._get_venv_path()
        if platform.system() == "Windows":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def _get_requirements_files(self, hardware_type: str, dev: bool = False) -> list[str]:
        """Get the requirements files for the specified hardware type."""
        req_config = self.config.get("requirements", {})
        result = []

        # Add hardware-specific requirements if available
        if hardware_type != "base" and "hardware" in req_config:
            hw_req = req_config.get("hardware", {}).get(hardware_type)
            if hw_req and Path(hw_req).exists():
                result.append(hw_req)

        # Add base requirements if not already included or if no hardware-specific requirements
        base_req = req_config.get("base", "requirements.txt")
        if base_req and Path(base_req).exists() and base_req not in result:
            result.append(base_req)

        # Add development requirements if requested
        if dev and "dev" in req_config:
            dev_req = req_config.get("dev")
            if dev_req and Path(dev_req).exists():
                result.append(dev_req)

        return result

    def _get_lockfile_path(self, hardware_type: str, dev: bool = False) -> str:
        """Get the path to the lockfile for the specified hardware type."""
        lock_config = self.config.get("lockfiles", {})

        # Get hardware-specific lockfile if available
        if hardware_type != "base" and "hardware" in lock_config:
            hw_lock = lock_config.get("hardware", {}).get(hardware_type)
            if hw_lock:
                return hw_lock

        # Default to base lockfile
        return lock_config.get("base", "requirements.lock")

    def _ensure_uv_installed(self) -> bool:
        """Ensure uv is installed."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logging.info("uv not found, attempting to install...")
            try:
                if platform.system() == "Windows":
                    # Install uv on Windows
                    subprocess.run(
                        [
                            "powershell.exe",
                            "-Command",
                            "(Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing)"
                            ".Content | powershell -",
                        ],
                        check=True,
                    )
                else:
                    # Install uv on Unix-like systems
                    subprocess.run(["curl", "-sSf", "https://astral.sh/uv/install.sh", "|", "sh"], check=True)
                return True
            except subprocess.SubprocessError:
                logging.error("Failed to install uv. Please install it manually from https://github.com/astral-sh/uv")
                return False

    def _create_venv(self, clean: bool = False) -> bool:
        """Create a virtual environment."""
        venv_path = self._get_venv_path()

        # Clean existing environment if requested
        if clean and venv_path.exists():
            logging.info(f"Removing existing environment at {venv_path}")
            shutil.rmtree(venv_path)

        # Create virtual environment if it doesn't exist
        if not venv_path.exists():
            logging.info(f"Creating virtual environment at {venv_path}")
            try:
                if self._ensure_uv_installed():
                    subprocess.run(["uv", "venv", str(venv_path)], check=True)
                else:
                    import venv

                    venv.create(venv_path, with_pip=True)
                return True
            except (subprocess.SubprocessError, ImportError) as e:
                logging.error(f"Error creating virtual environment: {e}")
                return False

        logging.info(f"Using existing virtual environment at {venv_path}")
        return True

    def setup(self, args: argparse.Namespace) -> int:
        """Set up the environment for the specified hardware."""
        hardware_type = args.hardware or self.hardware_type
        logging.info(f"Setting up environment for hardware type: {hardware_type}")

        # Create virtual environment
        if not self._create_venv(args.clean):
            return 1

        # Get requirements files
        req_files = self._get_requirements_files(hardware_type, args.dev)
        if not req_files:
            logging.error(f"No requirements files found for hardware type: {hardware_type}")
            return 1

        # Install requirements
        logging.info(f"Installing requirements from: {', '.join(req_files)}")
        if not self._ensure_uv_installed():
            logging.error("uv is required for installation")
            return 1

        try:
            for req_file in req_files:
                cmd = ["uv", "pip", "install", "-r", req_file]
                logging.info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
        except subprocess.SubprocessError as e:
            logging.error(f"Error installing requirements: {e}")
            return 1

        logging.info(f"Environment for {hardware_type} set up successfully")
        return 0

    def run(self, args: argparse.Namespace) -> int:
        """Run a command in the virtual environment."""
        venv_python = self._get_python_executable()
        if not venv_python.exists():
            logging.error(f"Python executable not found at {venv_python}")
            logging.error("Please run 'python uvfast.py setup' first")
            return 1

        # Run the command
        cmd = [str(venv_python)] + args.command
        logging.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=False)
            return 0
        except subprocess.SubprocessError as e:
            logging.error(f"Error running command: {e}")
            return 1

    def info(self, args: argparse.Namespace) -> int:
        """Display information about the environment."""
        # Show hardware information
        print("Hardware Information:")
        print(f"Project: {self.config.get('project_name', 'ai-playground')}")
        print(f"Detected hardware type: {self.hardware_type}")
        print(f"Available hardware types: {', '.join(self.config.get('hardware_types', []}")

        # Show requirements files
        print("\nRequirements files:")
        req_config = self.config.get("requirements", {})
        print(f"  Base: {req_config.get('base', 'requirements.txt')}")
        print(f"  Dev: {req_config.get('dev', 'requirements-dev.txt')}")
        print("  Hardware:")
        for hw_type, req_file in req_config.get("hardware", {}).items():
            print(f"    {hw_type}: {req_file}")

        # Show lockfiles
        print("\nLockfiles:")
        lock_config = self.config.get("lockfiles", {})
        print(f"  Base: {lock_config.get('base', 'requirements.lock')}")
        print(f"  Dev: {lock_config.get('dev', 'requirements-dev.lock')}")
        print("  Hardware:")
        for hw_type, lock_file in lock_config.get("hardware", {}).items():
            print(f"    {hw_type}: {lock_file}")

        # Show environment path
        venv_path = self._get_venv_path()
        print(f"\nVirtual environment: {venv_path}")
        if venv_path.exists():
            print("  Status: Installed")
            print(f"  Python: {self._get_python_executable()}")
        else:
            print("  Status: Not installed")

        return 0

    def update_lockfiles(self, args: argparse.Namespace) -> int:
        """Update lockfiles for the specified hardware types."""
        hardware_types = self.config.get(
            "hardware_types",
            [] if args.all else [args.hardware or self.hardware_type],
        )
        logging.info(f"Updating lockfiles for hardware types: {', '.join(hardware_types)}")

        if not self._ensure_uv_installed():
            logging.error("uv is required for updating lockfiles")
            return 1

        for hw_type in hardware_types:
            # Get requirements files
            req_files = self._get_requirements_files(hw_type, args.dev)
            if not req_files:
                logging.warning(f"No requirements files found for hardware type: {hw_type}")
                continue

            # Get lockfile path
            lock_file = self._get_lockfile_path(hw_type, args.dev)
            logging.info(f"Updating lockfile for {hw_type}: {lock_file}")

            # Update lockfile
            try:
                cmd = ["uv", "pip", "compile"]
                for req_file in req_files:
                    cmd.extend(["-r", req_file])
                cmd.extend(["-o", lock_file])

                logging.info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logging.info(f"Lockfile {lock_file} updated successfully")
            except subprocess.SubprocessError as e:
                logging.error(f"Error updating lockfile for {hw_type}: {e}")
                return 1

        return 0

    def lock(self, args: argparse.Namespace) -> int:
        """Generate lockfiles for dependencies."""
        if not self._ensure_uv_installed():
            logging.error("uv is required for lockfile generation")
            return 1

        # Get the hardware types to process
        hardware_types = self.config.get(
            "hardware_types",
            [] if args.all else [args.hardware or self.hardware_type],
        )

        for hw_type in hardware_types:
            logging.info(f"Generating lockfile for hardware type: {hw_type}")

            req_files = self._get_requirements_files(hw_type, args.dev)
            if not req_files:
                logging.error(f"No requirements files found for hardware type: {hw_type}")
                continue

            for req_file in req_files:
                lockfile = self._get_lockfile_path(hw_type, args.dev)
                logging.info(f"Generating lockfile: {lockfile}")

                try:
                    subprocess.run(
                        ["uv", "pip", "compile", req_file, "--output", lockfile],
                        check=True,
                    )
                except subprocess.SubprocessError as e:
                    logging.error(f"Error generating lockfile: {e}")
                    return 1

        logging.info("Lockfile generation complete")
        return 0


def main() -> int:
    """Main entry point for uvfast."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the environment")
    setup_parser.add_argument(
        "--hardware",
        choices=["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
        help="Hardware type to set up the environment for",
    )
    setup_parser.add_argument("--dev", action="store_true", help="Include development dependencies")
    setup_parser.add_argument("--clean", action="store_true", help="Clean existing environment")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a command in the environment")
    run_parser.add_argument("command", nargs="+", help="Command to run")

    # Info command
    subparsers.add_parser("info", help="Show information about the environment")

    # Update lockfiles command
    update_parser = subparsers.add_parser("update-lockfiles", help="Update lockfiles")
    update_parser.add_argument(
        "--hardware",
        choices=["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
        help="Hardware type to update lockfiles for",
    )
    update_parser.add_argument("--dev", action="store_true", help="Include development dependencies")
    update_parser.add_argument("--all", action="store_true", help="Generate lockfiles for all hardware types")

    # Lock command
    lock_parser = subparsers.add_parser("lock", help="Generate lockfiles for dependencies")
    lock_parser.add_argument(
        "--hardware",
        choices=["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
        help="Hardware type to generate lockfile for",
    )
    lock_parser.add_argument("--dev", action="store_true", help="Include development dependencies")
    lock_parser.add_argument("--all", action="store_true", help="Generate lockfiles for all hardware types")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Create the UVFast instance
    uvfast = UVFast()

    # Dictionary-based command dispatch
    command_handlers = {
        "setup": uvfast.setup,
        "run": uvfast.run,
        "info": uvfast.info,
        "update-lockfiles": uvfast.update_lockfiles,
        "lock": uvfast.lock,
    }

    # Get the appropriate handler and execute it
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)

    # This should never happen as argparse will validate the command
    return 1


if __name__ == "__main__":
    sys.exit(main())
