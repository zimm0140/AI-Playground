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
import textwrap
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import hardware detection module, which should be in the same directory
try:
    # Update import path to use the module from tools/hardware
    import sys
    from pathlib import Path

    # Add tools directory to path if needed
    tools_dir = Path(__file__).parent.parent
    sys.path.append(str(tools_dir.absolute()))

    from tools.hardware import hardware_detection
except ImportError:
    logging.warning("hardware_detection.py not found in tools/hardware, trying local import")
    try:
        import hardware_detection
    except ImportError:
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

    def _load_config(self) -> dict[str, list[str]]:
        """Load configuration from uvfast.json or use defaults."""
        config_path = Path("uvfast.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                return config
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"Error loading configuration: {e}")
                logging.info("Using default configuration")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def _simple_hardware_detection(self) -> str:
        """Simple hardware detection as a fallback when the module is not available."""
        system = platform.system()

        # Check for GPU presence on Windows
        if system == "Windows":
            try:
                output = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "Name"],
                    capture_output=True,
                    text=True, check=False,
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
                text=True, check=False,
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

        # Get Python executable
        python_executable = self._get_python_executable()

        # Install dependencies
        for req_file in req_files:
            logging.info(f"Installing dependencies from {req_file}")

            try:
                if args.use_lockfile:
                    # Use lockfile if available
                    lockfile = self._get_lockfile_path(hardware_type, args.dev)
                    if Path(lockfile).exists():
                        logging.info(f"Using lockfile: {lockfile}")
                        if self._ensure_uv_installed():
                            subprocess.run(["uv", "pip", "sync", lockfile], check=True)
                        else:
                            subprocess.run(
                                [str(python_executable), "-m", "pip", "install", "-r", lockfile],
                                check=True,
                            )
                    else:
                        logging.info(f"Lockfile not found: {lockfile}")
                        if self._ensure_uv_installed():
                            subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                        else:
                            subprocess.run(
                                [str(python_executable), "-m", "pip", "install", "-r", req_file],
                                check=True,
                            )
                # Install from requirements file
                elif self._ensure_uv_installed():
                    subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                else:
                    subprocess.run(
                        [str(python_executable), "-m", "pip", "install", "-r", req_file],
                        check=True,
                    )
            except subprocess.SubprocessError as e:
                logging.error(f"Error installing dependencies: {e}")
                return 1

        logging.info("\nEnvironment setup complete!")
        logging.info("To activate the environment:")
        if platform.system() == "Windows":
            logging.info(f"    {self._get_venv_path()}\\Scripts\\activate")
        else:
            logging.info(f"    source {self._get_venv_path()}/bin/activate")

        return 0

    def run(self, args: argparse.Namespace) -> int:
        """Run a command in the configured environment."""
        python_executable = self._get_python_executable()

        if not python_executable.exists():
            logging.error(f"Error: Python executable not found at {python_executable}")
            logging.info("Please run 'python uvfast.py setup' first")
            return 1

        logging.info(f"Running command with {python_executable}")
        cmd = [str(python_executable)] + args.command

        try:
            env = os.environ.copy()
            # Add XPU_VISIBLE_DEVICES=0 for acm hardware if not already set
            if self.hardware_type == "acm" and "XPU_VISIBLE_DEVICES" not in env:
                env["XPU_VISIBLE_DEVICES"] = "0"

            result = subprocess.run(cmd, env=env, check=False)
            return result.returncode
        except subprocess.SubprocessError as e:
            logging.error(f"Error running command: {e}")
            return 1

    def lock(self, args: argparse.Namespace) -> int:
        """Generate lockfiles for dependencies."""
        if not self._ensure_uv_installed():
            logging.error("uv is required for lockfile generation")
            return 1

        # Get the hardware types to process
        if args.all:
            hardware_types = HARDWARE_TYPES
        else:
            hardware_types = [args.hardware or self.hardware_type]

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

    def info(self, args: argparse.Namespace) -> int:
        """Display information about the environment."""
        # Show hardware information
        if hardware_detection:
            logging.info("Hardware Information:")
            detected_type = hardware_detection.detect_hardware_type()
            logging.info(f"  Detected hardware type: {detected_type}")

            is_intel_arc = hardware_detection.has_intel_arc()
            is_intel_gpu = hardware_detection.has_intel_gpu()
            is_igpu_capable = hardware_detection.is_igpu_capable()

            # Display detailed hardware info
            if args.verbose:
                if platform.system() == "Windows":
                    logging.info("\nGPU Details (Windows):")
                    gpu_info = hardware_detection.get_windows_gpu_info()
                    for gpu in gpu_info:
                        logging.info(f"  {gpu}")
                elif platform.system() == "Linux":
                    logging.info("\nGPU Details (Linux):")
                    gpu_info = hardware_detection.get_linux_gpu_info()
                    for line in gpu_info:
                        logging.info(f"  {line}")
                else:
                    logging.info("\nGPU Details (macOS):")
                    gpu_info = hardware_detection.get_mac_gpu_info()
                    for line in gpu_info:
                        logging.info(f"  {line}")

            logging.info("\nIntel GPU Capabilities:")
            logging.info(f"  Has Intel Arc GPU: {is_intel_arc}")
            logging.info(f"  Has Intel GPU: {is_intel_gpu}")
            logging.info(f"  Has Intel integrated GPU: {is_igpu_capable}")
        else:
            logging.warning("Hardware detection module not available")

        # Show environment information
        venv_path = self._get_venv_path()
        python_executable = self._get_python_executable()

        logging.info("\nEnvironment Information:")
        logging.info(f"  Environment path: {venv_path}")
        logging.info(f"  Python executable: {python_executable}")

        if python_executable.exists():
            try:
                # Get Python version
                result = subprocess.run(
                    [str(python_executable), "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logging.info(f"  Python version: {result.stdout.strip()}")

                # Get package list
                if args.packages:
                    logging.info("\nInstalled Packages:")
                    result = subprocess.run(
                        [str(python_executable), "-m", "pip", "list"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    for line in result.stdout.strip().split("\n"):
                        logging.info(f"  {line}")
            except subprocess.SubprocessError as e:
                logging.error(f"Error getting Python information: {e}")
        else:
            logging.warning(f"Python executable not found at {python_executable}")

        # Show configuration information
        logging.info("\nConfiguration:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for k, v in value.items():
                    logging.info(f"    {k}: {v}")
            else:
                logging.info(f"  {key}: {value}")

        return 0

    def legacy_install(self, args: argparse.Namespace) -> int:
        """Install dependencies using traditional pip (but accelerated with uv)."""
        hardware_type = args.hardware or self.hardware_type
        req_files = self._get_requirements_files(hardware_type, args.dev)

        if not req_files:
            logging.error(f"No requirements files found for hardware type: {hardware_type}")
            return 1

        for req_file in req_files:
            logging.info(f"Installing dependencies from {req_file}")

            try:
                if self._ensure_uv_installed():
                    subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                else:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)
            except subprocess.SubprocessError as e:
                logging.error(f"Error installing dependencies: {e}")
                return 1

        logging.info("Installation complete")
        return 0

    def hardware_check(self, args: argparse.Namespace) -> int:
        """Check if hardware-specific requirements are available for the current hardware."""
        if "hardware_detection" in sys.modules:
            hardware_detection.print_hardware_info(verbose=args.verbose)
        else:
            logging.info(f"Detected hardware type: {self.hardware_type}")
            logging.info("Note: hardware_detection.py not found, showing limited information")

        req_files = self._get_requirements_files(self.hardware_type)
        logging.info(f"\nRequirements files for {self.hardware_type}:")
        for req_file in req_files:
            exists = Path(req_file).exists()
            logging.info(f"  - {req_file} ({'exists' if exists else 'not found'})")

        return 0

    def main(self) -> int:
        """Main entry point."""
        parser = argparse.ArgumentParser(
            description="Fast hardware-optimized Python environment manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
                Examples:
                    python uvfast.py setup --hardware acm --dev
                    python uvfast.py run pytest tests/
                    python uvfast.py lock --all
                    python uvfast.py info --verbose
                """
            ),
        )

        # Main subcommands
        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # Setup command
        setup_parser = subparsers.add_parser("setup", help="Set up the environment for the specified hardware")
        setup_parser.add_argument(
            "--hardware",
            choices=HARDWARE_TYPES,
            help="Hardware type to set up environment for (default: auto-detect)",
        )
        setup_parser.add_argument("--clean", action="store_true", help="Clean existing environment before setup")
        setup_parser.add_argument("--dev", action="store_true", help="Install development dependencies")
        setup_parser.add_argument(
            "--no-lock",
            action="store_true",
            help="Don't use lockfiles (not recommended)",
        )

        # Run command
        run_parser = subparsers.add_parser("run", help="Run a command in the environment")
        run_parser.add_argument(
            "--hardware",
            choices=HARDWARE_TYPES,
            help="Hardware type to use (default: auto-detect)",
        )
        run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

        # Lock command
        lock_parser = subparsers.add_parser("lock", help="Generate lockfiles for dependencies")
        lock_parser.add_argument(
            "--hardware",
            choices=HARDWARE_TYPES,
            help="Hardware type to generate lockfile for (default: auto-detect)",
        )
        lock_parser.add_argument("--all", action="store_true", help="Generate lockfiles for all hardware types")
        lock_parser.add_argument("--dev", action="store_true", help="Include development dependencies")

        # Info command
        info_parser = subparsers.add_parser("info", help="Display information about the environment")
        info_parser.add_argument("--verbose", action="store_true", help="Show verbose information")
        info_parser.add_argument("--packages", action="store_true", help="Show installed packages")

        # Legacy install command
        legacy_parser = subparsers.add_parser("legacy-install", help="Install dependencies using pip instead of uv")
        legacy_parser.add_argument(
            "--hardware",
            choices=HARDWARE_TYPES,
            help="Hardware type to install dependencies for (default: auto-detect)",
        )
        legacy_parser.add_argument("--dev", action="store_true", help="Install development dependencies")

        # Hardware check command
        hardware_parser = subparsers.add_parser(
            "hardware-check", help="Check hardware and show compatibility information"
        )
        hardware_parser.add_argument("--verbose", action="store_true", help="Show verbose hardware information")

        args = parser.parse_args()

        # Default to info if no command specified
        if not args.command:
            logging.info("No command specified, showing environment information\n")
            return self.info(argparse.Namespace(verbose=False, packages=False))

        # Dispatch to appropriate command
        if args.command == "setup":
            return self.setup(args)
        elif args.command == "run":
            return self.run(args)
        elif args.command == "lock":
            return self.lock(args)
        elif args.command == "info":
            return self.info(args)
        elif args.command == "legacy-install":
            return self.legacy_install(args)
        elif args.command == "hardware-check":
            return self.hardware_check(args)
        else:
            logging.error(f"Unknown command: {args.command}")
            return 1


if __name__ == "__main__":
    uvfast = UVFast()
    sys.exit(uvfast.main())
