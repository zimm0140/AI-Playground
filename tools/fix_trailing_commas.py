#!/usr/bin/env python3
"""
Script to fix missing trailing commas in Python files.

This tool automatically adds trailing commas (COM812) to:
1. Multi-line function calls
2. Multi-line list/dict/set literals
3. Multi-line tuple literals
"""

import os
import subprocess
import sys


def fix_files_with_ruff(file_paths):
    """
    Fix trailing comma issues in the given files using ruff.

    Args:
        file_paths: List of paths to files to fix

    Returns:
        int: Number of fixed files
    """
    # Make sure the list is not empty
    if not file_paths:
        return 0

    fixed_count = 0
    for file_path in file_paths:
        try:
            result = subprocess.run(
                ["ruff", "check", "--select=COM812", "--fix", file_path],
                capture_output=True,
                text=True,
                check=False,
            )

            if "1 fix applied" in result.stdout or "fixed" in result.stdout:
                print(f"Fixed trailing commas in {file_path}")
                fixed_count += 1
        except subprocess.SubprocessError as e:
            print(f"Error fixing trailing commas in {file_path}: {e}")

    return fixed_count


def find_python_files(directory='.'):
    """Find all Python files in the given directory recursively."""
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and node_modules
        for skip_dir in ['.venv', '.git', 'node_modules']:
            if skip_dir in dirs:
                dirs.remove(skip_dir)

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

    # Find files with trailing comma issues
    files_with_issues = []
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=COM812", "--output-format=json", "."],
            capture_output=True,
            text=True,
            check=False,
        )

        # If there are issues, the result will be a JSON array
        if result.stdout and result.stdout.strip().startswith('['):
            import json
            data = json.loads(result.stdout)
            unique_files = set()
            for item in data:
                unique_files.add(item.get('filename'))

            files_with_issues = list(unique_files)
            print(f"Found {len(files_with_issues)} files with trailing comma issues")
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        print(f"Error finding files with issues: {e}")
        files_with_issues = python_files  # Fall back to checking all files

    # Fix the files with issues
    fixed_count = fix_files_with_ruff(files_with_issues)

    print(f"\nFixed trailing commas in {fixed_count} files")

    # Verify the fixes
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=COM812", "."],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print("All trailing comma issues have been fixed!")
        else:
            print("Some trailing comma issues remain. Run this script again or fix them manually.")
    except subprocess.SubprocessError:
        print("Failed to verify fixes")


if __name__ == '__main__':
    main()