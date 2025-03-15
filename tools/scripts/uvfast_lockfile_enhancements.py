"""
Enhanced Lockfile Management for uvfast.py

This module adds advanced lockfile generation capabilities, allowing for:
1. Creation of hardware-specific lockfiles
2. Package-focused lockfiles (e.g., only lock certain packages)
3. Automatic lockfile synchronization during setup
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Constants
LOCKFILE_DIR = "lockfiles"
BASE_REQUIREMENTS = "requirements.txt"
DEV_REQUIREMENTS = "requirements-dev.txt"


class LockfileManager:
    """Manages hardware-aware lockfile generation and synchronization"""

    def __init__(self, project_root: Path, config_file: Path | None = None):
        """
        Initialize the lockfile manager

        Args:
            project_root: Root directory of the project
            config_file: Path to config file (default: project_root/uvfast.json)
        """
        self.project_root = project_root
        self.lockfile_dir = project_root / LOCKFILE_DIR

        # Ensure lockfile directory exists
        self.lockfile_dir.mkdir(exist_ok=True, parents=True)

        # Load config if available
        self.config_file = config_file or project_root / "uvfast.json"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from config file"""
        if self.config_file.exists():
            try:
                with Path(self.config_file).open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {self.config_file}")
                return {}
        return {}

    def _get_hardware_specific_requirements(self, hardware_type: str) -> Path:
        """Get path to hardware-specific requirements file"""
        hw_requirements = self.project_root / f"requirements-{hardware_type}.txt"
        if hw_requirements.exists():
            return hw_requirements
        return self.project_root / BASE_REQUIREMENTS

    def _get_lockfile_path(self, hardware_type: str, include_dev: bool = False) -> Path:
        """Get path to lockfile for specific hardware type"""
        suffix = "-dev" if include_dev else ""
        return self.lockfile_dir / f"requirements-{hardware_type}{suffix}.lock"

    def generate_lockfile(
        self,
        hardware_type: str,
        include_dev: bool = False,
        include_packages: list[str] | None = None,
        upgrade: bool = False,
    ) -> Path:
        """
        Generate a lockfile for the specified hardware type

        Args:
            hardware_type: Hardware type (e.g., 'base', 'acm', 'ovino')
            include_dev: Whether to include development dependencies
            include_packages: List of specific packages to include (if None, include all)
            upgrade: Whether to upgrade packages to their latest versions

        Returns:
            Path to generated lockfile
        """
        requirements_file = self._get_hardware_specific_requirements(hardware_type)
        lockfile_path = self._get_lockfile_path(hardware_type, include_dev)

        print(f"Generating lockfile for hardware type '{hardware_type}'...")

        # Prepare command arguments
        cmd = [sys.executable, "-m", "uv", "pip", "compile"]

        # Add upgrade flag if needed
        if upgrade:
            cmd.append("--upgrade")

        # Add specific packages if provided
        if include_packages:
            for pkg in include_packages:
                cmd.extend(["--constraint", f"{pkg}"])

        # Add requirements files
        cmd.extend(["--output-file", str(lockfile_path), str(requirements_file)])

        # Add dev requirements if requested
        if include_dev and (self.project_root / DEV_REQUIREMENTS).exists():
            cmd.append(str(self.project_root / DEV_REQUIREMENTS))

        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully generated lockfile: {lockfile_path}")
            return lockfile_path
        except subprocess.CalledProcessError as e:
            print(f"Error generating lockfile: {e}")
            raise

    def sync_from_lockfile(self, hardware_type: str, include_dev: bool = False, venv_dir: Path | None = None) -> bool:
        """
        Synchronize environment from lockfile

        Args:
            hardware_type: Hardware type (e.g., 'base', 'acm', 'ovino')
            include_dev: Whether to include development dependencies
            venv_dir: Virtual environment directory

        Returns:
            True if successful, False otherwise
        """
        lockfile_path = self._get_lockfile_path(hardware_type, include_dev)

        if not lockfile_path.exists():
            print(f"Lockfile {lockfile_path} does not exist. Generating...")
            self.generate_lockfile(hardware_type, include_dev)

        print(f"Syncing environment from lockfile: {lockfile_path}")

        # Prepare command arguments
        cmd = [sys.executable, "-m", "uv", "pip", "sync"]
        cmd.append(str(lockfile_path))

        # Add virtual environment path if provided
        if venv_dir:
            cmd.extend(["--python", str(venv_dir / "bin" / "python")])

        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            print("Successfully synchronized environment from lockfile")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error syncing from lockfile: {e}")
            return False

    def generate_all_lockfiles(self, include_dev: bool = False) -> list[Path]:
        """
        Generate lockfiles for all hardware types

        Args:
            include_dev: Whether to include development dependencies

        Returns:
            List of paths to generated lockfiles
        """
        hardware_types = self.config.get("hardware_types", ["base", "acm", "ovino"])

        lockfiles = []
        for hw_type in hardware_types:
            lockfile = self.generate_lockfile(hw_type, include_dev)
            lockfiles.append(lockfile)

        return lockfiles

    def lockfile_exists(self, hardware_type: str, include_dev: bool = False) -> bool:
        """Check if lockfile exists for specified hardware type"""
        lockfile_path = self._get_lockfile_path(hardware_type, include_dev)
        return lockfile_path.exists()


def integrate_with_uvfast(lockfile_manager: LockfileManager) -> None:
    """
    Example of how to integrate with uvfast.py

    This would be incorporated into uvfast.py's CLI command handling
    """
    import argparse

    # Example parser (would be part of uvfast.py's main parser)
    parser = argparse.ArgumentParser(description="Lockfile management commands")
    subparsers = parser.add_subparsers(dest="command")

    # Lock command
    lock_parser = subparsers.add_parser("lock", help="Generate lockfiles")
    lock_parser.add_argument("--hardware", default="base", help="Hardware type")
    lock_parser.add_argument("--dev", action="store_true", help="Include dev dependencies")
    lock_parser.add_argument("--all", action="store_true", help="Generate for all hardware types")
    lock_parser.add_argument("--upgrade", action="store_true", help="Upgrade packages")
    lock_parser.add_argument("--include-package", action="append", help="Include specific package")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync from lockfile")
    sync_parser.add_argument("--hardware", default="base", help="Hardware type")
    sync_parser.add_argument("--dev", action="store_true", help="Include dev dependencies")

    # Parse arguments (for demonstration)
    args = parser.parse_args()

    # Handle commands
    if args.command == "lock":
        if args.all:
            lockfile_manager.generate_all_lockfiles(include_dev=args.dev)
        else:
            lockfile_manager.generate_lockfile(
                hardware_type=args.hardware,
                include_dev=args.dev,
                include_packages=args.include_package,
                upgrade=args.upgrade,
            )
    elif args.command == "sync":
        lockfile_manager.sync_from_lockfile(hardware_type=args.hardware, include_dev=args.dev)


# Example usage
if __name__ == "__main__":
    project_root = Path(__file__).parent
    manager = LockfileManager(project_root)

    # Example: Generate lockfiles
    # manager.generate_lockfile("base", include_dev=True)

    # Example: How this would be integrated with uvfast.py
    # integrate_with_uvfast(manager)