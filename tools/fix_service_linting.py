#!/usr/bin/env python3
"""
Script to automatically fix linting issues in the service directory.

This tool runs Ruff with the fix option to automatically correct common
linting problems in the service directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run Ruff to fix linting issues in the service directory."""
    service_dir = Path("service")

    if not service_dir.exists():
        print(f"Error: {service_dir} directory does not exist.")
        return 1

    # Run Ruff with the same parameters as the CI
    print("Running Ruff with the same parameters as the CI...")
    cmd = [
        "ruff",
        "check",
        str(service_dir),
        "--select=E,F,W,I,N,UP,B,A,COM,C90,RSE,RET,SIM",
        "--fix",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}", file=sys.stderr)

        # Check for remaining issues
        print("\nChecking for remaining issues...")
        check_cmd = [
            "ruff",
            "check",
            str(service_dir),
            "--select=E,F,W,I,N,UP,B,A,COM,C90,RSE,RET,SIM",
        ]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)

        if check_result.returncode == 0:
            print("✅ All linting issues fixed!")
            return 0
        else:
            print("❌ Some linting issues remain:")
            print(check_result.stdout)
            return 1

    except subprocess.SubprocessError as e:
        print(f"Error running Ruff: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix linting issues in service directory")
    args = parser.parse_args()
    sys.exit(main())