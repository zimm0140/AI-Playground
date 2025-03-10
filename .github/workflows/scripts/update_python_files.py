#!/usr/bin/env python3
"""
Python Files Updater

This script identifies and updates Python files added or modified in a pull request
to ensure they follow the project's code style guidelines and pass linting.

Usage:
    python update_python_files.py
"""

import os
import glob
import subprocess
import tempfile


def run_command(command, show_output=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if show_output:
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    return result


def find_python_files(directory):
    """Find all Python files in the given directory recursively."""
    python_files = []
    for filename in glob.glob(f"{directory}/**/*.py", recursive=True):
        python_files.append(filename)
    return sorted(python_files)


def is_new_script(filename):
    """Check if a file is one of our new workflow scripts."""
    new_scripts = [
        "validate_workflow_schema.py",
        "version_workflow.py",
        "generate_workflow_docs.py",
    ]
    return os.path.basename(filename) in new_scripts


def fix_ruff_issues(files):
    """Fix common Ruff issues in the specified files."""
    print(f"Fixing Ruff issues in {len(files)} files...")

    # Common Ruff arguments
    ruff_common_args = [
        "--select=E,F",
        "--ignore=E501",  # Ignore line length errors
        "--line-length=100",
    ]

    # Fix issues
    for filename in files:
        print(f"Processing {filename}...")

        # Fix imports
        run_command(
            ["ruff", "check", "--select=F401,F403,F405"]
            + ruff_common_args[1:]
            + ["--fix", filename],
            show_output=False,
        )

        # Fix formatting
        run_command(
            ["ruff", "check", "--select=E"]
            + ruff_common_args[1:]
            + ["--fix", filename],
            show_output=False,
        )

        # Fix other issues
        run_command(
            ["ruff", "check"] + ruff_common_args + ["--fix", filename], show_output=True
        )

    # Print summary
    print("\nRuff fixing complete.")


def add_guard_clauses(files):
    """Add guard clauses to python scripts if missing."""
    for filename in files:
        with open(filename, "r") as f:
            content = f.read()

        if (
            'if __name__ == "__main__"' not in content
            and "if __name__ == '__main__'" not in content
        ):
            # Check if there's a main function
            has_main = "def main(" in content

            # Create temp file for the modified content
            with tempfile.NamedTemporaryFile("w", delete=False) as temp:
                temp.write(content)

                if has_main:
                    temp.write('\n\nif __name__ == "__main__":\n    main()\n')

            # Replace the original file
            os.replace(temp.name, filename)
            print(f"Added guard clause to {filename}")


def main():
    """Main function."""
    # Directory containing the workflow scripts
    scripts_dir = ".github/workflows/scripts"

    # Find Python files
    python_files = find_python_files(scripts_dir)

    # Filter for our new scripts
    new_scripts = [f for f in python_files if is_new_script(f)]

    if not new_scripts:
        print("No new workflow scripts found.")
        return

    print(f"Found {len(new_scripts)} new workflow scripts:")
    for script in new_scripts:
        print(f"  - {script}")

    # Fix Ruff issues
    fix_ruff_issues(new_scripts)

    # Add guard clauses if missing
    add_guard_clauses(new_scripts)


if __name__ == "__main__":
    main()
