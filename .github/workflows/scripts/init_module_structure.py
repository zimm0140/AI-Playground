#!/usr/bin/env python3
"""
Initialize Python Module Structure

This script ensures that all Python package directories have proper __init__.py files
to make imports work correctly in both development and CI environments.
"""

import os
from pathlib import Path


def ensure_init_files(root_dir="."):
    """
    Recursively create __init__.py files in all directories that contain Python files
    but don't already have an __init__.py file.
    """
    root_path = Path(root_dir)
    count = 0

    # Skip these directories
    skip_dirs = {
        ".git",
        ".github",
        ".venv",
        "__pycache__",
        "venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        "tests",
    }

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        # Check if directory contains Python files
        has_py_files = any(f.suffix == '.py') for f in filenames)

        if has_py_files:
            init_file = os.path.join(dirpath, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""Auto-generated package init file."""\n')
                print(f"Created {init_file}")
                count += 1

    return count


def ensure_tools_structure():
    """Ensure the tools directory is properly structured."""
    tools_dir = Path("tools")
    if not tools_dir.exists():
        return 0

    count = 0

    # Make sure tools has an __init__.py
    init_file = tools_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write('"""Tools package."""\n')
        print(f"Created {init_file}")
        count += 1

    # Check subdirectories
    subdirs = [d for d in tools_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for subdir in subdirs:
        subdir_init = subdir / "__init__.py"
        if not subdir_init.exists():
            with open(subdir_init, "w") as f:
                f.write(f'"""Tools {subdir.name} package."""\n')
            print(f"Created {subdir_init}")
            count += 1

    return count


def main():
    """Main function."""
    print("Ensuring Python package structure...")

    # First ensure the specific tools structure we know we need
    tools_count = ensure_tools_structure()

    # Then check all directories for Python files
    other_count = ensure_init_files()

    print(f"Created {tools_count + other_count} __init__.py files")


if __name__ == "__main__":
    main()