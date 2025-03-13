#!/usr/bin/env python
"""Test script to verify the virtual environment."""

import platform
import sys


def main():
    """Print information about the Python environment."""
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")

    # Try to import installed packages
    for package in ["mypy", "pytest", "ruff", "pre_commit"]:
        try:
            __import__(package)
            print(f"{package} is installed")
        except ImportError:
            print(f"{package} is not installed")


if __name__ == "__main__":
    main()
