from pathlib import Path

#!/usr/bin/env python3
"""
Quick Local Ruff Issues Fixer

This script runs Ruff on the service directory to identify and fix common issues.
It's designed to be run locally before pushing changes to ensure CI will pass.

Usage:
    python fix_ruff_issues_local.py

Requirements:
    - ruff must be installed: pip install ruff
"""

import glob
import os
import subprocess
import sys


def print_header(text):
    """Print a nicely formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f" {text} ".center(width, "="))
    print("=" * width + "\n")


def find_python_files(directory="service"):
    """Find all Python files in the given directory."""
    # Check if the directory exists
    if not os.path.isdir(directory):
        # Try common locations
        possible_locations = [
            "service",
            "../service",
            "WebUI/service",
            os.path.join(os.getcwd(), "service"),
        ]

        for loc in possible_locations:
            if os.path.isdir(loc):
                directory = loc
                print(f"Found service directory at: {os.path.abspath(directory)}")
                break
        else:
            print(
                f"Could not find service directory in any of these locations: {possible_locations}",
            )
            return []

    # Find all Python files
    files = []
    print(f"Searching for Python files in: {os.path.abspath(directory)}")
    for filename in glob.glob(f"{directory}/**/*.py", recursive=True):
        files.append(filename)
    return sorted(files)


def run_command(command, show_output=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if show_output:
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    return result


def fix_issues():
    """Fix common Ruff issues."""
    print_header("Finding Python Files")
    files = find_python_files()
    if not files:
        print("No Python files found in service directory!")
        return False

    print(f"Found {len(files)} Python files")

    # Define common Ruff arguments
    ruff_common_args = [
        "--select=E,F",
        "--ignore=E501",
        "--extend-exclude=.git,.github,.venv,venv,__pycache__,build,dist",
        "--line-length=100",
    ]

    # Check for issues first
    print_header("Checking for Issues")
    check_result = run_command(
        ["ruff", "check"] + ruff_common_args + ["--statistics"] + files,
    )

    if check_result.returncode == 0:
        print("\n✅ No issues found! Ruff is happy with your code.")
        return True

    # Fix issues
    print_header("Fixing Issues")
    run_command(["ruff", "check"] + ruff_common_args + ["--fix"] + files)

    # Check again after fixes
    print_header("Checking Again After Fixes")
    recheck_result = run_command(
        ["ruff", "check"] + ruff_common_args + ["--statistics"] + files,
    )

    if recheck_result.returncode == 0:
        print("\n✅ All issues fixed!")
        return True

    # Try more specific fixes
    print_header("Trying More Specific Fixes")

    # Fix unused imports
    print("\n📌 Fixing unused imports (F401)...")
    run_command(
        ["ruff", "check", "--select=F401"] + ruff_common_args[1:] + ["--fix"] + files,
        show_output=False,
    )

    # Fix other formatting issues
    print("\n📌 Fixing formatting issues (E)...")
    run_command(
        ["ruff", "check", "--select=E"] + ruff_common_args[1:] + ["--fix"] + files,
        show_output=False,
    )

    # Final check
    print_header("Final Check")
    final_result = run_command(
        ["ruff", "check"] + ruff_common_args + ["--statistics"] + files,
    )

    if final_result.returncode == 0:
        print("\n✅ All issues fixed successfully!")
        return True
    print("\n⚠️ Some issues still need manual attention")

    # Show remaining issues in a more readable way
    print_header("Issues Needing Manual Attention")
    run_command(["ruff", "check"] + ruff_common_args + ["--format=text"] + files)

    # Suggest manual fixes
    print_header("Suggestions for Manual Fixes")
    print(
        """
Common issues that need manual attention:

1. Unused imports (F401):
   - Remove unused imports from the top of the file
   - Or add '# noqa: F401' at the end of the import line if needed

2. Undefined names (F821):
   - Make sure variables are defined before use
   - Check for typos in variable names

3. Missing whitespace (E2xx):
   - Add spaces around operators
   - Add spaces after commas in lists/dicts
        """,
    )

    return False


if __name__ == "__main__":
    print_header("Ruff Issue Fixer")
    try:
        # Make sure we're in the right directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Check if we're in .github/workflows/scripts
        if os.path.basename(script_dir) == "scripts":
            workflows_dir = os.path.dirname(script_dir)
            if os.path.basename(workflows_dir) == "workflows":
                github_dir = os.path.dirname(workflows_dir)
                if os.path.basename(github_dir) == ".github":
                    # We're in .github/workflows/scripts, so go up to the repo root
                    repo_root = os.path.dirname(github_dir)
                    os.chdir(repo_root)
                    print(f"Changed to repository root: {os.getcwd()}")

        # Check if service directory exists in current location
        if not os.path.isdir("service"):
            print("Service directory not found in current location!")
            print(f"Current directory: {os.getcwd()}")
            print("Contents:", Path().iterdir())

            # Check if it's in the parent directory
            if os.path.isdir("../service"):
                os.chdir("..")
                print(f"Changed to parent directory: {os.getcwd()}")
            # Check if it's in WebUI directory
            elif os.path.isdir("WebUI/service"):
                os.chdir("WebUI")
                print(f"Changed to WebUI directory: {os.getcwd()}")

        success = fix_issues()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)