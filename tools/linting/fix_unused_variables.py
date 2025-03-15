#!/usr/bin/env python3
"""
Script to fix unused variable (F841) warnings in CI scripts.
"""

import re
import sys
from pathlib import Path


def fix_unused_variable(file_path):
    """
    Fix F841 (local variable assigned but never used) warnings in a file.

    Strategy:
    1. Change assignments to use '_' variable for unused values
    2. Comment out the assignment
    """
    with Path(file_path).open(encoding="utf-8") as f:
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

    filename = Path(file_path).name
    if filename in unused_vars:
        for var in unused_vars[filename]:
            # Replace variable assignments with _ to indicate they're intentionally unused
            pattern = rf"(\s+)({var})(\s*=\s*.*)"
            fixed_content = re.sub(pattern, r"\1_\3  # noqa: F841 (was \2)", fixed_content)

    # If we made changes, write them back to the file
    if fixed_content != content:
        with Path(file_path).open("w", encoding="utf-8") as f:
            f.write(fixed_content)
        print(f"✅ Fixed unused variables in {file_path}")
        return True
    print(f"ℹ️ No changes made to {file_path}")
    return False


def main():
    """Fix unused variables in CI scripts."""
    scripts_dir = ".github/workflows/scripts"
    files_to_fix = [
        Path(scripts_dir) / "check_requirements_consistency.py",
        Path(scripts_dir) / "collect_ci_metrics.py",
        Path(scripts_dir) / "ensure_unique_artifacts.py",
        Path(scripts_dir) / "fix_artifact_names.py",
        Path(scripts_dir) / "fix_ruff_issues.py",
        Path(scripts_dir) / "fix_service_ruff_issues.py",
        Path(scripts_dir) / "generate_api_summary.py",
        Path(scripts_dir) / "test_ruff_fix.py",
    ]

    count = 0
    for file_path in files_to_fix:
        if Path(file_path).exists():
            if fix_unused_variable(file_path):
                count += 1
        else:
            print(f"⚠️ File not found: {file_path}")

    print(f"\nFixed unused variables in {count} files.")

    # Also handle the F401 (unused import) in custom_test_runner.py
    custom_test_runner = Path(scripts_dir) / "custom_test_runner.py"
    if Path(custom_test_runner).exists():
        with Path(custom_test_runner).open(encoding="utf-8") as f:
            content = f.read()

        # Comment out or modify the unused import
        modified = re.sub(r"(from tests import test_api)", r"# \1  # noqa: F401", content)

        if modified != content:
            with Path(custom_test_runner).open("w", encoding="utf-8") as f:
                f.write(modified)
            print(f"✅ Fixed unused import in {custom_test_runner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())