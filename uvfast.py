#!/usr/bin/env python3
"""
uvfast.py - Simplified uv-based package management for AI projects with hardware acceleration

This script provides a streamlined approach to managing Python dependencies for projects
with hardware-specific requirements using the ultra-fast uv package manager.

Usage:
  python uvfast.py setup [--hardware TYPE] [--dev]  # Set up environment
  python uvfast.py info                            # Show environment info
  python uvfast.py lockfiles [--all]               # Generate lockfiles
  python uvfast.py run [command]                   # Run command in the environment

Features:
  - Automatic hardware detection (Intel Arc, Meteor Lake, etc.)
  - Fast dependency resolution with uv
  - Lockfile generation for reproducible environments
  - Backward compatibility with existing requirements files
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration - can be overridden with a uvfast.json file
DEFAULT_CONFIG = {
    "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino"],
    "lockfiles_dir": ".lockfiles",
    "venv_dir": ".venv",
    "cache_dir": ".uvcache",
    "parallel_jobs": 4,
}


def ensure_uv():
    """Ensure uv is installed, installing it if necessary."""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        print("Installing uv package manager...")

        if platform.system() == "Windows":
            install_cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
        else:
            install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"

        try:
            subprocess.run(install_cmd, shell=True, check=True)
            # Verify installation worked
            subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, check=True)
            return True
        except subprocess.SubprocessError:
            print(
                "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
            )
            return False


def detect_hardware():
    """Detect Intel hardware and return appropriate hardware type."""
    if platform.system() == "Windows":
        try:
            # Check for Intel GPU
            gpu_cmd = "wmic path win32_VideoController get name"
            gpu_output = subprocess.check_output(gpu_cmd, shell=True, text=True).lower()

            if "arc" in gpu_output or "intel graphics" in gpu_output:
                if "a770" in gpu_output or "a750" in gpu_output:
                    return "acm"  # Intel Arc A-Series
                elif any(x in gpu_output for x in ["battlemage", "bmg"]):
                    return "bmg"  # Intel Battlemage

            # Check for CPU type
            cpu_cmd = "wmic cpu get name"
            cpu_output = subprocess.check_output(cpu_cmd, shell=True, text=True).lower()

            if "meteor lake" in cpu_output:
                return "mtl"  # Intel Meteor Lake
            elif "lunar lake" in cpu_output:
                return "lnl"  # Intel Lunar Lake
        except Exception as e:
            print(f"Hardware detection error (Windows): {e}")

    else:  # Linux/macOS
        try:
            if shutil.which("lspci"):
                # Check for GPU
                gpu_cmd = "lspci | grep -i vga"
                gpu_output = subprocess.check_output(gpu_cmd, shell=True, text=True).lower()

                if "intel" in gpu_output:
                    if "arc" in gpu_output:
                        return "acm"  # Intel Arc

            # Check for CPU
            if shutil.which("lscpu"):
                cpu_cmd = "lscpu"
                cpu_output = subprocess.check_output(cpu_cmd, shell=True, text=True).lower()

                if "intel" in cpu_output:
                    if "meteor lake" in cpu_output:
                        return "mtl"  # Intel Meteor Lake
                    elif "lunar lake" in cpu_output:
                        return "lnl"  # Intel Lunar Lake
        except Exception as e:
            print(f"Hardware detection error (Unix): {e}")

    # Check for OpenVINO
    try:
        with open(os.devnull, "w") as devnull:
            subprocess.check_call(
                [sys.executable, "-c", "import openvino"], stdout=devnull, stderr=devnull
            )
            return "ovino"  # OpenVINO is installed
    except:
        pass

    return "base"  # Default fallback


def load_config():
    """Load configuration from uvfast.json if present, otherwise use defaults."""
    config = DEFAULT_CONFIG.copy()

    config_file = Path("uvfast.json")
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {config_file}. Using default config.")

    return config


def get_requirements_files(hardware_type, dev=False):
    """Get the appropriate requirements file paths for the hardware type."""
    # Start with base requirements
    files = ["requirements.txt"]

    # Add hardware-specific requirements if they exist and not base
    if hardware_type != "base":
        hw_file = Path(f"requirements-hardware-{hardware_type}.txt")
        if hw_file.exists():
            files.append(str(hw_file))

    # Add dev requirements if requested
    if dev:
        dev_file = Path("requirements-dev.txt")
        if dev_file.exists():
            files.append(str(dev_file))

    return files


def generate_lockfile(config, hardware_type, dev=False):
    """Generate a lockfile for the given hardware type."""
    lockdir = Path(config["lockfiles_dir"])
    lockdir.mkdir(exist_ok=True)

    # Determine lockfile name
    lockfile_suffix = "-dev" if dev else ""
    lockfile = lockdir / f"{hardware_type}{lockfile_suffix}.lock"

    # Get requirements files
    req_files = get_requirements_files(hardware_type, dev)

    # Check if requirements files exist
    existing_files = [f for f in req_files if os.path.exists(f)]
    if not existing_files:
        print(f"Error: No requirements files found for {hardware_type}")
        return None

    # Build command for generating lockfile
    cmd = ["uv", "pip", "compile"]
    for req_file in existing_files:
        cmd.extend(["-r", req_file])

    cmd.extend(["--output-file", str(lockfile)])

    # Execute command
    print(f"Generating lockfile: {lockfile}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Created lockfile: {lockfile}")
        return lockfile
    except subprocess.CalledProcessError as e:
        print(f"Error generating lockfile: {e}")
        return None


def setup_environment(config, hardware_type, dev=False, use_lockfile=True):
    """Set up a Python environment for the given hardware type."""
    # Ensure uv is installed
    if not ensure_uv():
        sys.exit(1)

    # Create virtual environment
    venv_dir = Path(config["venv_dir"])
    if not venv_dir.exists():
        print(f"Creating virtual environment in {venv_dir}")
        subprocess.run(["uv", "venv", "--path", str(venv_dir)], check=True)

    # Determine if we should use lockfiles
    if use_lockfile:
        lockdir = Path(config["lockfiles_dir"])
        lockfile_suffix = "-dev" if dev else ""
        lockfile = lockdir / f"{hardware_type}{lockfile_suffix}.lock"

        # Generate lockfile if it doesn't exist
        if not lockfile.exists():
            lockfile = generate_lockfile(config, hardware_type, dev)
            if lockfile is None:
                print("Falling back to direct installation without lockfile")
                use_lockfile = False

        if use_lockfile:
            # Install from lockfile
            print(f"Installing dependencies from lockfile: {lockfile}")
            cmd = [
                "uv",
                "pip",
                "sync",
                "--path",
                str(venv_dir),
                "--cache-dir",
                config["cache_dir"],
                "--jobs",
                str(config["parallel_jobs"]),
                str(lockfile),
            ]
            subprocess.run(cmd, check=True)

    if not use_lockfile:
        # Install directly from requirements files
        req_files = get_requirements_files(hardware_type, dev)
        existing_files = [f for f in req_files if os.path.exists(f)]

        if not existing_files:
            print(f"Error: No requirements files found for {hardware_type}")
            sys.exit(1)

        print(f"Installing dependencies from: {', '.join(existing_files)}")

        cmd = [
            "uv",
            "pip",
            "install",
            "--path",
            str(venv_dir),
            "--cache-dir",
            config["cache_dir"],
            "--jobs",
            str(config["parallel_jobs"]),
        ]

        for req_file in existing_files:
            cmd.extend(["-r", req_file])

        subprocess.run(cmd, check=True)

    # Support editable install if setup.py exists
    if Path("setup.py").exists():
        print("Installing package in development mode...")
        setup_cmd = ["uv", "pip", "install", "--path", str(venv_dir), "-e", "."]
        subprocess.run(setup_cmd, check=True)

    # Print activation instructions
    print("\nSetup complete! Activate your environment:")
    if platform.system() == "Windows":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")


def run_command(config, cmd_args):
    """Run a command in the virtual environment using uv run."""
    venv_dir = Path(config["venv_dir"])
    if not venv_dir.exists():
        print(f"Virtual environment not found at {venv_dir}")
        print("Run 'python uvfast.py setup' first")
        sys.exit(1)

    cmd = ["uv", "run", "--path", str(venv_dir)]
    cmd.extend(cmd_args)

    subprocess.run(cmd)


def show_info(config):
    """Show information about the environment and detected hardware."""
    print("=== Environment Information ===")

    # Python info
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")

    # Hardware detection
    hw_type = detect_hardware()
    print(f"Detected hardware: {hw_type}")

    # uv info
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
        print(f"uv version: {uv_version}")
    except:
        print("uv: Not installed")

    # Requirements files
    print("\nAvailable requirement files:")
    base_reqs = Path("requirements.txt")
    if base_reqs.exists():
        size = base_reqs.stat().st_size
        print(f"- requirements.txt ({size} bytes)")

    dev_reqs = Path("requirements-dev.txt")
    if dev_reqs.exists():
        size = dev_reqs.stat().st_size
        print(f"- requirements-dev.txt ({size} bytes)")

    for hw in config["hardware_types"]:
        if hw != "base":
            hw_file = Path(f"requirements-hardware-{hw}.txt")
            if hw_file.exists():
                size = hw_file.stat().st_size
                print(f"- requirements-hardware-{hw}.txt ({size} bytes)")

    # Check virtual environment
    venv_dir = Path(config["venv_dir"])
    if venv_dir.exists():
        print(f"\nVirtual environment: {venv_dir} (exists)")
        # Try to get installed packages
        if platform.system() == "Windows":
            pip_path = venv_dir / "Scripts" / "pip"
        else:
            pip_path = venv_dir / "bin" / "pip"

        if pip_path.with_suffix(".exe").exists() or pip_path.exists():
            try:
                cmd = [str(pip_path), "list"]
                output = subprocess.check_output(cmd, text=True)
                pkg_count = len(output.splitlines()) - 2  # Subtract header lines
                print(f"Installed packages: {pkg_count}")
            except:
                print("Could not get installed packages")
    else:
        print(f"\nVirtual environment: {venv_dir} (does not exist)")

    # Check lockfiles
    lockdir = Path(config["lockfiles_dir"])
    if lockdir.exists():
        lockfiles = list(lockdir.glob("*.lock"))
        if lockfiles:
            print(f"\nLockfiles ({len(lockfiles)}):")
            for lockfile in lockfiles:
                size = lockfile.stat().st_size
                print(f"- {lockfile.name} ({size} bytes)")
        else:
            print("\nNo lockfiles generated yet")
    else:
        print("\nNo lockfiles generated yet")


def main():
    """Main entry point."""
    config = load_config()

    parser = argparse.ArgumentParser(description="Fast uv-based setup for AI projects")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up a Python environment")
    setup_parser.add_argument(
        "--hardware",
        choices=config["hardware_types"],
        default="detect",
        help="Hardware type (default: auto-detect)",
    )
    setup_parser.add_argument("--dev", action="store_true", help="Include development dependencies")
    setup_parser.add_argument("--no-lockfile", action="store_true", help="Don't use lockfiles")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show environment information")

    # Lockfiles command
    lockfiles_parser = subparsers.add_parser("lockfiles", help="Generate lockfiles")
    lockfiles_parser.add_argument("--all", action="store_true", help="Generate all lockfiles")
    lockfiles_parser.add_argument(
        "--hardware", choices=config["hardware_types"], help="Hardware type for lockfile"
    )
    lockfiles_parser.add_argument(
        "--dev", action="store_true", help="Include development dependencies"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a command in the virtual environment")
    run_parser.add_argument(
        "cmd_args", nargs=argparse.REMAINDER, help="Command and arguments to run"
    )

    # Legacy install command
    legacy_parser = subparsers.add_parser(
        "legacy-install", help="Install using traditional requirements.txt (but faster)"
    )
    legacy_parser.add_argument(
        "--dev", action="store_true", help="Include development dependencies"
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "setup":
        hardware = args.hardware
        if hardware == "detect":
            hardware = detect_hardware()
            print(f"Detected hardware: {hardware}")

        setup_environment(config, hardware, args.dev, not args.no_lockfile)

    elif args.command == "info":
        show_info(config)

    elif args.command == "lockfiles":
        if args.all:
            for hw_type in config["hardware_types"]:
                generate_lockfile(config, hw_type, dev=False)
                generate_lockfile(config, hw_type, dev=True)
        elif args.hardware:
            generate_lockfile(config, args.hardware, args.dev)
        else:
            print("Error: Specify --all or --hardware")
            sys.exit(1)

    elif args.command == "run":
        if not args.cmd_args:
            print("Error: No command specified")
            sys.exit(1)
        run_command(config, args.cmd_args)

    elif args.command == "legacy-install":
        # Simulate the old workflow but with uv's better performance
        ensure_uv()

        venv_dir = Path(config["venv_dir"])
        if not venv_dir.exists():
            subprocess.run(["uv", "venv", "--path", str(venv_dir)], check=True)

        print("Installing using legacy workflow with uv acceleration...")
        legacy_req = Path("requirements.txt")
        legacy_dev = Path("requirements-dev.txt")

        if legacy_req.exists():
            subprocess.run(
                ["uv", "pip", "install", "--path", str(venv_dir), "-r", str(legacy_req)], check=True
            )

        if legacy_dev.exists() and args.dev:
            subprocess.run(
                ["uv", "pip", "install", "--path", str(venv_dir), "-r", str(legacy_dev)], check=True
            )

        print("\nLegacy installation complete! Activate your environment:")
        if platform.system() == "Windows":
            print(f"  {venv_dir}\\Scripts\\activate")
        else:
            print(f"  source {venv_dir}/bin/activate")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
