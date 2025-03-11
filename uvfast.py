#!/usr/bin/env python3
"""uvfast: Fast hardware-optimized Python environment manager.

This script provides tools for managing Python environments optimized for
different hardware configurations (Intel Arc GPUs, OpenVINO, etc.).
It leverages the 'uv' package manager for fast dependency installation.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Try to import hardware detection module, which should be in the same directory
try:
    import hardware_detection
except ImportError:
    print("Warning: hardware_detection.py not found, some features will be limited")
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
                print(f"Error loading configuration: {e}")
                print("Using default configuration")
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
                    text=True,
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
        """Ensure that uv is installed and available."""
        try:
            subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("uv not found, attempting to install...")
            try:
                if platform.system() == "Windows":
                    # Install uv on Windows using powershell
                    subprocess.run(
                        [
                            "powershell",
                            "-Command",
                            "(Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing).Content | powershell -",
                        ],
                        check=True,
                    )
                else:
                    # Install uv on Unix-like systems
                    subprocess.run(
                        ["curl", "-sSf", "https://astral.sh/uv/install.sh", "|", "sh"], check=True
                    )
                return True
            except subprocess.SubprocessError:
                print(
                    "Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
                )
                return False

    def _create_venv(self, clean: bool = False) -> bool:
        """Create a virtual environment."""
        venv_path = self._get_venv_path()

        # Clean existing environment if requested
        if clean and venv_path.exists():
            print(f"Removing existing environment at {venv_path}")
            shutil.rmtree(venv_path)

        # Create virtual environment if it doesn't exist
        if not venv_path.exists():
            print(f"Creating virtual environment at {venv_path}")
            try:
                if self._ensure_uv_installed():
                    subprocess.run(["uv", "venv", str(venv_path)], check=True)
                else:
                    import venv

                    venv.create(venv_path, with_pip=True)
                return True
            except (subprocess.SubprocessError, ImportError) as e:
                print(f"Error creating virtual environment: {e}")
                return False

        print(f"Using existing virtual environment at {venv_path}")
        return True

    def setup(self, args: argparse.Namespace) -> int:
        """Set up the environment for the specified hardware."""
        hardware_type = args.hardware or self.hardware_type
        print(f"Setting up environment for hardware type: {hardware_type}")

        # Create virtual environment
        if not self._create_venv(args.clean):
            return 1

        # Get requirements files
        req_files = self._get_requirements_files(hardware_type, args.dev)
        if not req_files:
            print(f"No requirements files found for hardware type: {hardware_type}")
            return 1

        # Get Python executable
        python_executable = self._get_python_executable()

        # Install dependencies
        for req_file in req_files:
            print(f"Installing dependencies from {req_file}")

            try:
                if args.use_lockfile:
                    # Use lockfile if available
                    lockfile = self._get_lockfile_path(hardware_type, args.dev)
                    if Path(lockfile).exists():
                        print(f"Using lockfile: {lockfile}")
                        if self._ensure_uv_installed():
                            subprocess.run(["uv", "pip", "sync", lockfile], check=True)
                        else:
                            subprocess.run(
                                [str(python_executable), "-m", "pip", "install", "-r", lockfile],
                                check=True,
                            )
                    else:
                        print(f"Lockfile not found: {lockfile}")
                        if self._ensure_uv_installed():
                            subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                        else:
                            subprocess.run(
                                [str(python_executable), "-m", "pip", "install", "-r", req_file],
                                check=True,
                            )
                else:
                    # Install from requirements file
                    if self._ensure_uv_installed():
                        subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                    else:
                        subprocess.run(
                            [str(python_executable), "-m", "pip", "install", "-r", req_file],
                            check=True,
                        )
            except subprocess.SubprocessError as e:
                print(f"Error installing dependencies: {e}")
                return 1

        print("\nEnvironment setup complete!")
        print("To activate the environment:")
        if platform.system() == "Windows":
            print(f"    {self._get_venv_path()}\\Scripts\\activate")
        else:
            print(f"    source {self._get_venv_path()}/bin/activate")

        return 0

    def run(self, args: argparse.Namespace) -> int:
        """Run a command in the configured environment."""
        python_executable = self._get_python_executable()

        if not python_executable.exists():
            print(f"Error: Python executable not found at {python_executable}")
            print("Please run 'python uvfast.py setup' first")
            return 1

        print(f"Running command with {python_executable}")
        cmd = [str(python_executable)] + args.command

        try:
            env = os.environ.copy()
            # Add XPU_VISIBLE_DEVICES=0 for acm hardware if not already set
            if self.hardware_type == "acm" and "XPU_VISIBLE_DEVICES" not in env:
                env["XPU_VISIBLE_DEVICES"] = "0"

            result = subprocess.run(cmd, env=env)
            return result.returncode
        except subprocess.SubprocessError as e:
            print(f"Error running command: {e}")
            return 1

    def lock(self, args: argparse.Namespace) -> int:
        """Generate lockfiles for dependencies."""
        if not self._ensure_uv_installed():
            print("uv is required for lockfile generation")
            return 1

        hardware_types = []
        if args.all:
            hardware_types = self.config.get("hardware_types", ["base"])
        else:
            hardware_types = [args.hardware or self.hardware_type]

        for hw_type in hardware_types:
            req_files = self._get_requirements_files(hw_type, args.dev)
            if not req_files:
                print(f"No requirements files found for hardware type: {hw_type}")
                continue

            lockfile = self._get_lockfile_path(hw_type, args.dev)
            print(f"Generating lockfile for {hw_type}: {lockfile}")

            cmd = ["uv", "pip", "compile"]
            for req_file in req_files:
                cmd.extend(["--requirement", req_file])
            cmd.extend(["--output-file", lockfile])

            try:
                subprocess.run(cmd, check=True)
                print(f"Lockfile generated: {lockfile}")
            except subprocess.SubprocessError as e:
                print(f"Error generating lockfile: {e}")
                return 1

        return 0

    def info(self, args: argparse.Namespace) -> int:
        """Display information about the environment and detected hardware."""
        print("uvfast v1.0.0")
        print(f"Python version: {platform.python_version()}")
        print(f"System: {platform.system()} {platform.release()}")

        print(f"\nDetected hardware type: {self.hardware_type}")

        # Get GPU information if available
        if "hardware_detection" in sys.modules:
            hardware_info = hardware_detection.get_hardware_info()

            print("\nGPUs:")
            if hardware_info["gpus"]:
                for gpu in hardware_info["gpus"]:
                    print(f"  - {gpu}")
            else:
                print("  No GPUs detected")

            print(f"\nCPU: {hardware_info['cpu'].get('name', 'Unknown')}")
            print(f"OpenVINO available: {hardware_info['openvino_available']}")
        else:
            # Simple fallback
            print("\nNote: hardware_detection.py not found, showing limited information")
            if platform.system() == "Windows":
                try:
                    output = subprocess.run(
                        ["wmic", "path", "win32_VideoController", "get", "Name"],
                        capture_output=True,
                        text=True,
                    ).stdout
                    gpu_names = [line.strip() for line in output.split("\n")[1:] if line.strip()]
                    print("\nGPUs:")
                    for gpu in gpu_names:
                        print(f"  - {gpu}")
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("  Unable to detect GPUs")

        # Display environment information
        venv_path = self._get_venv_path()
        if venv_path.exists():
            print(f"\nVirtual environment: {venv_path} (exists)")
            python_executable = self._get_python_executable()
            print(f"Python executable: {python_executable}")

            if python_executable.exists():
                # Get installed packages
                try:
                    output = subprocess.run(
                        [str(python_executable), "-m", "pip", "list"],
                        capture_output=True,
                        text=True,
                    ).stdout
                    package_count = len(output.split("\n")) - 2  # Subtract header rows
                    print(f"Installed packages: {package_count}")

                    # Check for key packages
                    key_packages = [
                        "torch",
                        "numpy",
                        "flask",
                        "intel-extension-for-pytorch",
                        "openvino",
                    ]
                    print("\nKey packages:")
                    for package in key_packages:
                        try:
                            result = subprocess.run(
                                [
                                    str(python_executable),
                                    "-c",
                                    f"import {package.replace('-', '_')}; print({package.replace('-', '_')}.__version__)",
                                ],
                                capture_output=True,
                                text=True,
                            )
                            if result.returncode == 0:
                                print(f"  {package}: {result.stdout.strip()}")
                            else:
                                print(f"  {package}: not installed")
                        except subprocess.SubprocessError:
                            print(f"  {package}: not installed")
                except subprocess.SubprocessError:
                    print("Unable to list installed packages")
        else:
            print(f"\nVirtual environment: {venv_path} (does not exist)")
            print("Run 'python uvfast.py setup' to create the environment")

        # Show configuration
        if args.verbose:
            print("\nConfiguration:")
            print(json.dumps(self.config, indent=2))

        return 0

    def legacy_install(self, args: argparse.Namespace) -> int:
        """Install dependencies using traditional pip (but accelerated with uv)."""
        hardware_type = args.hardware or self.hardware_type
        req_files = self._get_requirements_files(hardware_type, args.dev)

        if not req_files:
            print(f"No requirements files found for hardware type: {hardware_type}")
            return 1

        for req_file in req_files:
            print(f"Installing dependencies from {req_file}")

            try:
                if self._ensure_uv_installed():
                    subprocess.run(["uv", "pip", "install", "-r", req_file], check=True)
                else:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", req_file], check=True
                    )
            except subprocess.SubprocessError as e:
                print(f"Error installing dependencies: {e}")
                return 1

        print("Installation complete")
        return 0

    def hardware_check(self, args: argparse.Namespace) -> int:
        """Check if hardware-specific requirements are available for the current hardware."""
        if "hardware_detection" in sys.modules:
            hardware_detection.print_hardware_info(verbose=args.verbose)
        else:
            print(f"Detected hardware type: {self.hardware_type}")
            print("Note: hardware_detection.py not found, showing limited information")

        req_files = self._get_requirements_files(self.hardware_type)
        print(f"\nRequirements files for {self.hardware_type}:")
        for req_file in req_files:
            exists = Path(req_file).exists()
            print(f"  - {req_file} ({'exists' if exists else 'not found'})")

        return 0

    def main(self) -> int:
        """Main entry point for the uvfast CLI."""
        parser = argparse.ArgumentParser(
            description="Fast hardware-optimized Python environment manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
                Examples:
                  # Set up environment for detected hardware
                  python uvfast.py setup

                  # Set up environment for specific hardware with development dependencies
                  python uvfast.py setup --hardware acm --dev

                  # Generate lockfiles for all hardware types
                  python uvfast.py lock --all

                  # Run a command in the configured environment
                  python uvfast.py run pytest

                  # Show hardware and environment information
                  python uvfast.py info

                  # Install dependencies using traditional pip (but accelerated with uv)
                  python uvfast.py legacy-install
            """
            ),
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # setup command
        setup_parser = subparsers.add_parser(
            "setup", help="Set up the environment for the specified hardware"
        )
        setup_parser.add_argument(
            "--hardware",
            choices=self.config.get("hardware_types", ["base"]),
            help="Hardware type (detected by default)",
        )
        setup_parser.add_argument(
            "--dev", action="store_true", help="Include development dependencies"
        )
        setup_parser.add_argument(
            "--clean", action="store_true", help="Clean existing environment before setup"
        )
        setup_parser.add_argument(
            "--use-lockfile",
            action="store_true",
            help="Use lockfiles for reproducible environments",
        )

        # run command
        run_parser = subparsers.add_parser(
            "run", help="Run a command in the configured environment"
        )
        run_parser.add_argument("command", nargs="+", help="Command to run")

        # lock command
        lock_parser = subparsers.add_parser("lock", help="Generate lockfiles for dependencies")
        lock_parser.add_argument(
            "--hardware",
            choices=self.config.get("hardware_types", ["base"]),
            help="Hardware type (detected by default)",
        )
        lock_parser.add_argument(
            "--dev", action="store_true", help="Include development dependencies"
        )
        lock_parser.add_argument(
            "--all", action="store_true", help="Generate lockfiles for all hardware types"
        )

        # info command
        info_parser = subparsers.add_parser(
            "info", help="Display information about the environment and detected hardware"
        )
        info_parser.add_argument(
            "--verbose", "-v", action="store_true", help="Show verbose information"
        )

        # legacy-install command
        legacy_parser = subparsers.add_parser(
            "legacy-install",
            help="Install dependencies using traditional pip (but accelerated with uv)",
        )
        legacy_parser.add_argument(
            "--hardware",
            choices=self.config.get("hardware_types", ["base"]),
            help="Hardware type (detected by default)",
        )
        legacy_parser.add_argument(
            "--dev", action="store_true", help="Include development dependencies"
        )

        # hardware-check command
        hardware_parser = subparsers.add_parser(
            "hardware-check",
            help="Check if hardware-specific requirements are available for the current hardware",
        )
        hardware_parser.add_argument(
            "--verbose", "-v", action="store_true", help="Show verbose information"
        )

        args = parser.parse_args()

        # Default to info if no command specified
        if not args.command:
            print("No command specified, showing environment information\n")
            return self.info(argparse.Namespace(verbose=False))

        # Run the appropriate command
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
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(UVFast().main())
