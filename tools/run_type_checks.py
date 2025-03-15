#!/usr/bin/env python3
"""
Type Checking Tool

This script runs mypy type checks on specified files or directories.
It can be used in CI pipelines to enforce type correctness.
"""

import os
import sys
import subprocess
import argparse
from typing import List, Optional


def setup_mypy_if_needed() -> bool:
    """Install mypy if it's not already installed.
    
    Returns:
        bool: True if mypy is available, False otherwise.
    """
    try:
        # Check if mypy is already installed
        subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("mypy not found, attempting to install...")
        try:
            # Try to install mypy
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "mypy"],
                check=True
            )
            return True
        except subprocess.SubprocessError:
            print("Failed to install mypy. Please install it manually.")
            return False


def run_mypy(targets: List[str], config_file: Optional[str] = None) -> int:
    """Run mypy on the specified targets.
    
    Args:
        targets: List of files or directories to check.
        config_file: Path to mypy config file. Defaults to None.
        
    Returns:
        int: Return code from mypy (0 for success).
    """
    cmd = [sys.executable, "-m", "mypy"]
    
    if config_file and os.path.exists(config_file):
        cmd.extend(["--config-file", config_file])
    
    cmd.extend(targets)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    """Run the type checking tool.
    
    Returns:
        int: Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Run mypy type checks")
    parser.add_argument(
        "--targets", 
        nargs="+", 
        default=[
            ".github/workflows/scripts/",
            "tools/linting/",
            "tools/apply_type_annotations.py",
            "tools/fix_docstring_indentation.py"
        ],
        help="Files or directories to type check"
    )
    parser.add_argument(
        "--config", 
        default="mypy.ini", 
        help="Path to mypy config file"
    )
    
    args = parser.parse_args()
    
    if not setup_mypy_if_needed():
        return 1
    
    return run_mypy(args.targets, args.config)


if __name__ == "__main__":
    sys.exit(main()) 