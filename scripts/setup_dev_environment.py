#!/usr/bin/env python
"""
One-click Development Environment Setup

This script automates the setup of a complete development environment for AI Playground.
It handles installing uv, creating a virtual environment, installing dependencies,
setting up pre-commit hooks, and configuring VS Code.

Usage:
    python scripts/setup_dev_environment.py

Requirements:
    Python 3.10+
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def print_step(message: str) -> None:
    """Print a step message with formatting."""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def run_command(
    cmd: List[str], cwd: Optional[str] = None, check: bool = True
) -> Tuple[int, str]:
    """Run a command and return the exit code and output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            text=True,
            capture_output=True,
        )
        return result.returncode, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(e.returncode)
        return e.returncode, e.stdout


def is_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(
            ["uv", "--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False


def install_uv() -> None:
    """Install uv based on the platform."""
    print_step("Installing uv")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Use PowerShell to install uv on Windows
        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    elif system in ("linux", "darwin"):
        # Use curl to install uv on Linux/macOS
        cmd = ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
    else:
        print(f"Unsupported platform: {system}")
        print("Please install uv manually: https://github.com/astral-sh/uv")
        sys.exit(1)
    
    run_command(cmd)
    
    # Verify installation
    if not is_uv_installed():
        print("Failed to install uv. Please install it manually.")
        print("See: https://github.com/astral-sh/uv")
        sys.exit(1)


def setup_virtual_environment() -> None:
    """Set up a virtual environment using uv."""
    print_step("Setting up virtual environment")
    
    # Create virtual environment
    run_command(["uv", "venv"])
    
    # Determine the Python executable in the virtual environment
    if platform.system().lower() == "windows":
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(".venv", "bin", "python")
    
    # Verify the virtual environment
    if not os.path.exists(venv_python):
        print(f"Virtual environment Python not found at: {venv_python}")
        sys.exit(1)


def install_dependencies() -> None:
    """Install dependencies using uv."""
    print_step("Installing dependencies")
    
    # Sync dependencies from lockfiles
    run_command(["uv", "pip", "sync", "requirements.lock", "requirements-dev.lock"])


def setup_pre_commit() -> None:
    """Set up pre-commit hooks."""
    print_step("Setting up pre-commit hooks")
    
    # Install pre-commit hooks
    run_command(["pre-commit", "install"])


def setup_vscode() -> None:
    """Set up VS Code configuration."""
    print_step("Setting up VS Code configuration")
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Check if VS Code settings already exist
    if not (vscode_dir / "settings.json").exists() or not (vscode_dir / "extensions.json").exists():
        print("VS Code configuration files already exist in the repository.")
        print("No changes needed.")
    else:
        print("VS Code configuration is ready.")
    
    # Recommend VS Code extensions
    print("\nRecommended VS Code extensions:")
    print("  - ms-python.python")
    print("  - ms-python.vscode-pylance")
    print("  - charliermarsh.ruff")
    print("  - matangover.mypy")
    print("  - davidanson.vscode-markdownlint")
    print("  - github.vscode-github-actions")
    print("  - tamasfe.even-better-toml")


def main() -> None:
    """Main function to set up the development environment."""
    print_step("Setting up AI Playground development environment")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print(f"Python 3.10+ is required. You have Python {python_version.major}.{python_version.minor}")
        sys.exit(1)
    
    # Install uv if not already installed
    if not is_uv_installed():
        install_uv()
    else:
        print("uv is already installed.")
    
    # Set up virtual environment
    setup_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Set up pre-commit hooks
    setup_pre_commit()
    
    # Set up VS Code configuration
    setup_vscode()
    
    print_step("Setup complete!")
    print("\nTo activate the virtual environment:")
    
    if platform.system().lower() == "windows":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
    
    print("\nTo run tests:")
    if platform.system().lower() == "windows":
        print("  .\\scripts\\run_with_uv.ps1 test")
    else:
        print("  ./scripts/run_with_uv.sh test")
    
    print("\nFor more information, see:")
    print("  - QUICKSTART.md")
    print("  - MIGRATION.md")
    print("  - .github/CONTRIBUTING.md")


if __name__ == "__main__":
    main() 