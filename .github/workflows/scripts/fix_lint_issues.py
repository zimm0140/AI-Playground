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

        # Fix trailing whitespace (W291) - more aggressively
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Fix blank lines with whitespace (W293) - more aggressively
        content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

        # Ensure file ends with a single newline (W292)
        content = content.rstrip("\n") + "\n"

        # Fix unnecessary list() calls (C408)
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
            [sys.executable, "-m", "isort", "--profile", "black", file_path],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def run_ruff_format(file_path):
    """Run ruff format on the given file."""
    try:
        # First run ruff check --fix to fix fixable issues
        subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--fix", file_path],
            check=False,
            capture_output=True,
        )

        # Then run ruff format to ensure consistent formatting
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", file_path],
            check=False,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def fix_long_lines(file_path, max_line_length=120):
    """Attempt to fix overly long lines by breaking them at logical points."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        new_lines = []

        for line in lines:
            # Skip comments, docstrings, and already short lines
            if line.lstrip().startswith(("#", '"', "'")) or len(line) <= max_line_length:
                new_lines.append(line)
                continue

            # Try to break long lines at logical points
            if "," in line and not any(quote in line for quote in ["'", '"']):
                # Split at commas for function arguments and lists
                parts = line.split(",")
                indent = len(line) - len(line.lstrip())
                current_line = parts[0]

                for part in parts[1:]:
                    if len(current_line + "," + part) <= max_line_length:
                        current_line += "," + part
                    else:
                        new_lines.append(current_line + ",")
                        current_line = " " * (indent + 4) + part.lstrip()

                new_lines.append(current_line)
            else:
                # Can't safely break the line, leave it for manual fixing
                new_lines.append(line)

        new_content = "\n".join(new_lines)

        # Write changes if needed
        if new_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error trying to fix long lines in {file_path}: {e}")
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
    py_files = find_python_files()
    print(f"Found {len(py_files)} Python files to process")

    # First run the fixes in order of simplicity/safety
    whitespace_count = 0
    isort_count = 0
    ruff_count = 0
    line_count = 0

    for file_path in py_files:
        # Run fixes in order of simplicity/safety
        if run_ruff_format(file_path):
            ruff_count += 1

        if run_isort(file_path):
            isort_count += 1

        if fix_whitespace_issues(file_path):
            whitespace_count += 1

        if fix_long_lines(file_path):
            line_count += 1

    print(f"\n✅ Fixed issues in {len(py_files)} files:")
    print(f"- Ruff fixes: {ruff_count}")
    print(f"- Import sorting: {isort_count}")
    print(f"- Whitespace fixes: {whitespace_count}")
    print(f"- Line length fixes: {line_count}")


if __name__ == "__main__":
    main()
