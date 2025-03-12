#!/usr/bin/env python3
"""
Script to check if all linting issues have been fixed.
"""

import subprocess
import sys


def check_directory(directory):
    """Run ruff check on the specified directory."""
    print(f"Checking {directory}...")
    result = subprocess.run(["ruff", "check", directory], capture_output=True, text=True, check=False)

    if result.returncode == 0:
        print(f"✅ {directory} passed!")
        return True
    else:
        print(
            f"❌ {directory} failed with {result.stdout.count('F401') + result.stdout.count('F841') + result.stdout.count('E402')} issues:"
        )
        print(result.stdout)
        return False


def main():
    """Check all directories for linting issues."""
    directories = [".github/workflows/scripts/", "tests/", "LlamaCPP/", "OpenVINO/"]

    all_passed = True
    for directory in directories:
        if not check_directory(directory):
            all_passed = False

    if all_passed:
        print("\n✅ All checks passed! Your code should now pass the CI linting checks.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the remaining issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
