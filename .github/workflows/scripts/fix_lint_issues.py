#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the codebase.

This script addresses:
- W293: Blank lines with whitespace
- W291: Trailing whitespace
- W292: Missing newline at end of file
- I001: Unsorted imports (using isort)
- E501: Line too long (where possible)
- C408: Unnecessary list calls
"""

import re
import subprocess
import sys
from pathlib import Path


def fix_whitespace_issues(file_path):
    """Fix whitespace issues in the given file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Fix trailing whitespace
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Fix blank lines with whitespace
        content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

        # Ensure file ends with a single newline
        content = content.rstrip("\n") + "\n"

        # Fix unnecessary list() calls
        content = re.sub(r"list\(\[\]|\[\]\)", "[]", content)
        content = re.sub(r"list\(\[([^]]*)\]\)", r"[\1]", content)

        # Write changes if needed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def run_isort(file_path):
    """Run isort on the given file to fix import sorting."""
    try:
        subprocess.run(
            [sys.executable, "-m", "isort", file_path],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_ruff_format(file_path):
    """Run ruff format on the given file to fix formatting issues."""
    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", file_path],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def find_python_files():
    """Find all Python files in the repository."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    python_files = list(repo_root.glob("**/*.py"))

    # Filter out virtual environments or other non-project paths
    excluded_patterns = [".venv", "venv", ".git", "__pycache__", "node_modules", "build", "dist"]

    return [str(path) for path in python_files if not any(pattern in str(path) for pattern in excluded_patterns)]


def main():
    """Find and fix linting issues in Python files."""
    python_files = find_python_files()
    print(f"Found {len(python_files)} Python files to process")

    fixed_count = 0
    for file_path in python_files:
        # Apply whitespace fixes first
        whitespace_fixed = fix_whitespace_issues(file_path)

        # Try to run isort and ruff format if available
        isort_fixed = run_isort(file_path)
        ruff_fixed = run_ruff_format(file_path)

        if whitespace_fixed or isort_fixed or ruff_fixed:
            fixed_count += 1
            print(f"✅ Fixed issues in {file_path}")
        else:
            print(f"✓ No fixable issues in {file_path}")

    print(f"\n✅ Fixed issues in {fixed_count} files")
    print("\nNote: Some issues require manual fixing:")
    print("1. C901: Function complexity - refactor complex functions into smaller ones")
    print("2. F841/F401: Unused variables/imports - remove or use them appropriately")
    print("3. F811: Redefined variables - fix naming conflicts")
    print("4. Custom imports or logic issues - review and fix manually")


if __name__ == "__main__":
    main()
