#!/usr/bin/env python3
"""
Script to fix unused variable (F841) warnings in CI scripts.
"""

import os
import re
import sys


def fix_unused_variable(file_path):
    """
    Fix F841 (local variable assigned but never used) warnings in a file.

    Strategy:
    1. Change assignments to use '_' variable for unused values
    2. Comment out the assignment
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find lines with variable assignments that are never used
    # This is a simplistic approach - we're looking for patterns like:
    # variable_name = ...
    fixed_content = content

    # List of files and their unused variables from the error output
    unused_vars = {
        "check_requirements_consistency.py": ["exit_code"],
        "collect_ci_metrics.py": ["package_name"],
        "ensure_unique_artifacts.py": ["workflow_id"],
        "fix_artifact_names.py": ["artifact_indent"],
        "fix_ruff_issues.py": ["fix_result"],
        "fix_service_ruff_issues.py": ["fix_result"],
        "generate_api_summary.py": ["func_pattern"],
        "test_ruff_fix.py": ["fix_result"],
    }

    filename = os.path.basename(file_path)
    if filename in unused_vars:
        for var in unused_vars[filename]:
            # Replace variable assignments with _ to indicate they're intentionally unused
            pattern = rf"(\s+)({var})(\s*=\s*.*)"
            fixed_content = re.sub(pattern, r"\1_\3  # noqa: F841 (was \2)", fixed_content)

    # If we made changes, write them back to the file
    if fixed_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        print(f"✅ Fixed unused variables in {file_path}")
        return True
    else:
        print(f"ℹ️ No changes made to {file_path}")
        return False


def main():
    """Fix unused variables in CI scripts."""
    scripts_dir = ".github/workflows/scripts"
    files_to_fix = [
        os.path.join(scripts_dir, "check_requirements_consistency.py"),
        os.path.join(scripts_dir, "collect_ci_metrics.py"),
        os.path.join(scripts_dir, "ensure_unique_artifacts.py"),
        os.path.join(scripts_dir, "fix_artifact_names.py"),
        os.path.join(scripts_dir, "fix_ruff_issues.py"),
        os.path.join(scripts_dir, "fix_service_ruff_issues.py"),
        os.path.join(scripts_dir, "generate_api_summary.py"),
        os.path.join(scripts_dir, "test_ruff_fix.py"),
    ]

    count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_unused_variable(file_path):
                count += 1
        else:
            print(f"⚠️ File not found: {file_path}")

    print(f"\nFixed unused variables in {count} files.")

    # Also handle the F401 (unused import) in custom_test_runner.py
    custom_test_runner = os.path.join(scripts_dir, "custom_test_runner.py")
    if os.path.exists(custom_test_runner):
        with open(custom_test_runner, "r", encoding="utf-8") as f:
            content = f.read()

        # Comment out or modify the unused import
        modified = re.sub(r"(from tests import test_api)", r"# \1  # noqa: F401", content)

        if modified != content:
            with open(custom_test_runner, "w", encoding="utf-8") as f:
                f.write(modified)
            print(f"✅ Fixed unused import in {custom_test_runner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
