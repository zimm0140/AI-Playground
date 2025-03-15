#!/usr/bin/env python3
"""
Fix PTH208 issues by replacing os.listdir() with pathlib.Path.iterdir()

This script scans Python files and replaces os.listdir() calls with the
pathlib equivalent to fix PTH208 linting errors.
"""

import os
import re
import sys


def fix_pth208(file_path):
    """Fix PTH208 issues in the given file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try with a different encoding
            with open(file_path, encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False

    # Pattern to match os.listdir() calls
    pattern = r"os\.listdir\(([^)]+)\)"

    def replace_with_pathlib(match):
        path_arg = match.group(1)
        return f"Path({path_arg}).iterdir()"

    new_content = re.sub(pattern, replace_with_pathlib, content)

    # Only write to file if changes were made
    if new_content != content:
        # Add pathlib import if not already present
        if "from pathlib import Path" not in new_content and "import pathlib" not in new_content:
            # Add after other imports or at the top if no imports
            import_pattern = r"((?:^import.*?\n|^from.*?\n)+)"
            match = re.search(import_pattern, new_content)
            if match:
                new_content = re.sub(
                    import_pattern,
                    r"\1\nfrom pathlib import Path\n",
                    new_content,
                    count=1,
                )
            else:
                new_content = f"from pathlib import Path\n\n{new_content}"

        # Fix list comprehensions that use os.listdir
        new_content = fix_list_comprehensions(new_content)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False

    return False


def fix_list_comprehensions(content):
    """Fix list comprehensions that use the result of Path.iterdir()

    Path.iterdir() returns Path objects, not strings like os.listdir()
    """
    # Pattern for list comprehensions using iterdir()
    pattern = r"for\s+(\w+)\s+in\s+Path\(([^)]+)\)\.iterdir\(\)"

    def replace_comprehension(match):
        var_name = match.group(1)
        path_arg = match.group(2)
        return f"for {var_name} in Path({path_arg}).iterdir()"

    return re.sub(pattern, replace_comprehension, content)


def find_python_files(directory):
    """Find all Python files in the given directory recursively, excluding .venv directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip .venv directory
        if ".venv" in dirs:
            dirs.remove(".venv")

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main():
    """Main function to fix PTH208 issues in the codebase."""
    root_dir = "."
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    python_files = find_python_files(root_dir)

    fixed_count = 0
    for file_path in python_files:
        if fix_pth208(file_path):
            print(f"Fixed {file_path}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()