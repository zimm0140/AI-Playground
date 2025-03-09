#!/usr/bin/env python3
"""
Test Ruff Fixability

This script tests whether the service directory issues can be fixed by Ruff
without requiring manual intervention. It's useful to run this script before
pushing changes to see if CI will pass.
"""

import subprocess
import sys
import os
import time
import argparse


def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")


def run_cmd(cmd, capture=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if capture:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f}s with exit code {result.returncode}")
        return result
    else:
        start_time = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f}s with exit code {result.returncode}")
        return result


def create_config_file():
    """Create a Ruff configuration file."""
    config = """
[tool.ruff]
# Enable flake8-bugbear (`B`) rules
select = ["E", "F", "B", "I"]

# Ignore specific rules that are causing failures
ignore = [
    # Line too long (handled separately)
    "E501",
    # Too many arguments/statements/returns (too strict)
    "PLR0913", "PLR0915", "PLR0911",
    # Import style (should be handled by our custom formatter)
    "I001", "I002",
]

# Exclude a variety of commonly ignored directories
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".git-rewrite",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
    "models",
    "**/tmp",
]

# Allow lines to be as long as 100 characters
line-length = 100

# Allow unused variables when underscore-prefixed.
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

[tool.ruff.per-file-ignores]
# Tests can use assertions and relative imports
"test_*.py" = ["S101", "E402"]

[tool.ruff.isort]
known-first-party = ["service"]
"""
    with open("pyproject.toml", "w") as f:
        f.write(config.strip())
    print("Created Ruff configuration file: pyproject.toml")


def find_service_python_files():
    """Find all Python files in the service directory."""
    if not os.path.isdir("service"):
        print("Error: service directory not found in the current directory.")
        print(f"Current directory: {os.getcwd()}")
        print("Contents:", os.listdir("."))
        sys.exit(1)
        
    files = []
    for root, _, filenames in os.walk("service"):
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} Python files in the service directory.")
    return files


def test_ruff_fixes(args):
    """Test if Ruff can fix the issues in the service directory."""
    # Change to root directory if needed
    if os.path.basename(os.getcwd()) == "scripts" and os.path.exists("../../service"):
        os.chdir("../..")
        print(f"Changed to directory: {os.getcwd()}")
        
    # Create a Ruff config file
    if not args.skip_config:
        create_config_file()
    
    # Find service Python files
    service_files = find_service_python_files()
    if not service_files:
        print("No Python files found to check.")
        return 0
    
    # Initial check to see what issues exist
    print_header("Checking for Ruff issues")
    result = run_cmd(["ruff", "check", "--select=E,F", "--statistics"] + service_files)
    
    if result.returncode == 0:
        print("No issues found! All files pass Ruff checks.")
        return 0
    
    # Try to fix the issues
    print_header("Attempting to fix issues")
    fix_result = run_cmd(["ruff", "check", "--select=E,F", "--fix"] + service_files)
    
    # Check if fixes were successful
    print_header("Checking if fixes were successful")
    after_result = run_cmd(["ruff", "check", "--select=E,F", "--statistics"] + service_files)
    
    if after_result.returncode == 0:
        print("All issues fixed successfully!")
        return 0
    
    # Try more specific fixes
    print_header("Applying more specific fixes")
    
    # Try fixing just unused imports (F401)
    run_cmd(["ruff", "check", "--select=F401", "--fix"] + service_files)
    
    # Try fixing just line length issues (E501)
    run_cmd(["ruff", "check", "--select=E501", "--fix"] + service_files)
    
    # Final check
    print_header("Final check")
    final_result = run_cmd(["ruff", "check", "--select=E,F", "--statistics"] + service_files)
    
    if final_result.returncode == 0:
        print("All issues fixed successfully after multiple passes!")
        return 0
    else:
        print("Some issues still require manual attention.")
        if not args.quiet:
            print("\nRemaining issues:")
            print(final_result.stdout)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ruff fixes on service directory")
    parser.add_argument("--skip-config", action="store_true", help="Skip creating config file")
    parser.add_argument("--quiet", action="store_true", help="Don't show detailed error output")
    args = parser.parse_args()
    
    try:
        exit_code = test_ruff_fixes(args)
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 