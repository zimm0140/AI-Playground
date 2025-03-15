#!/usr/bin/env python3
"""
Script to fix PTH208 linting issues by replacing os.listdir with Path.iterdir.

This tool scans Python files and converts:
1. os.listdir() calls to Path().iterdir()
2. Adds necessary imports
3. Fixes list comprehensions that might need adjustment
"""

import os
import re
import sys


def fix_file(file_path):
    """
    Fix PTH208 issues in the given file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    # Read file content
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Pattern to match os.listdir() calls
    pattern = r'os\.listdir\(([^)]+)\)'

    def replace_with_pathlib(match):
        path_arg = match.group(1)
        return f'Path({path_arg}).iterdir()'

    new_content = re.sub(pattern, replace_with_pathlib, content)

    # Only write to file if changes were made
    if new_content != content:
        # Add pathlib import if it's not already there
        if 'from pathlib import Path' not in new_content and 'import pathlib' not in new_content:
            # Check if there are other imports to add it after
            import_match = re.search(r'^import\s+.*$', new_content, re.MULTILINE)
            if import_match:
                new_content = re.sub(
                    r'^(import\s+.*?)$',
                    r'\1\nfrom pathlib import Path\n',
                    new_content,
                    count=1,
                )
            else:
                new_content = f'from pathlib import Path\n\n{new_content}'

        # Fix list comprehensions that use os.listdir
        new_content = fix_list_comprehensions(new_content)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False

    return False


def fix_list_comprehensions(content):
    """
    Fix list comprehensions that use the result of Path.iterdir()

    Path.iterdir() returns Path objects, not strings like os.listdir()
    """
    # Pattern for list comprehensions using iterdir()
    pattern = r'for\s+(\w+)\s+in\s+Path\(([^)]+)\)\.iterdir\(\)'

    def replace_comprehension(match):
        var_name = match.group(1)
        path_arg = match.group(2)
        return f'for {var_name} in Path({path_arg}).iterdir()'

    return re.sub(pattern, replace_comprehension, content)


def find_python_files(directory='.'):
    """Find all Python files in the given directory recursively."""
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments
        if '.venv' in dirs:
            dirs.remove('.venv')

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def main():
    """Main function to find and fix Python files."""
    root_dir = '.'

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    python_files = find_python_files(root_dir)

    fixed_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            print(f"Fixed {file_path}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()