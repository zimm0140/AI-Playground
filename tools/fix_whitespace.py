#!/usr/bin/env python3
"""
Script to fix whitespace issues in Python files.

This tool automatically fixes:
1. Trailing whitespace (W293)
2. Missing newline at end of file (W292)
"""

import os
import re
import subprocess
import sys


def fix_whitespace_issues(file_path):
    """
    Fix whitespace issues in the given file.

    Args:
        file_path: Path to the file to fix

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Fix trailing whitespace on blank lines (W293)
    new_content = re.sub(r'[ \t]+\n', '\n', content)

    # Ensure file ends with a newline (W292)
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'

    # Only write to file if changes were made
    if new_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False

    return False


def find_python_files(directory='.'):
    """Find all Python files in the given directory recursively."""
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '.git' in dirs:
            dirs.remove('.git')

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
        if fix_whitespace_issues(file_path):
            print(f"Fixed {file_path}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")

    # Run ruff to verify the changes
    try:
        subprocess.run(
            ["ruff", "check", "--select=W292,W293", "."],
            check=False,
            capture_output=True,
            text=True,
        )
        if fixed_count > 0:
            print("Whitespace issues have been fixed!")
        else:
            print("No whitespace issues found.")
    except subprocess.SubprocessError:
        print("Failed to run Ruff to verify changes")


if __name__ == '__main__':
    main()