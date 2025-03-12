#!/usr/bin/env python3
"""
Service-specific Ruff Auto-fix Script

This script automatically fixes Ruff linting issues in the service directory by:
1. Finding all Python files in the service directory
2. Running Ruff with the --fix flag to auto-correct formatting issues
3. Generating a report of what was fixed and what needs manual attention
"""

import glob
import os
import subprocess
import sys


def find_service_python_files():
    """Find all Python files in the service directory."""
    py_files = []
    service_dir = "service"

    # Check if service directory exists
    if not os.path.isdir(service_dir):
        print(f"Error: {service_dir} directory not found.")
        return []

    # Find all Python files in the service directory
    for py_file in glob.glob(f"{service_dir}/**/*.py", recursive=True):
        py_files.append(py_file)

    print(f"Found {len(py_files)} Python files in the service directory.")
    return py_files


def run_ruff_fix(files=None):
    """Run Ruff with --fix flag on the specified files."""
    if files is None:
        files = find_service_python_files()

    if not files:
        print("No Python files found to fix.")
        return True, "No Python files found to fix."

    try:
        # First run Ruff check to see what issues exist
        print(f"Running Ruff check on {len(files)} Python files...")
        result = subprocess.run(
            ["ruff", "check", "--select=E,F,W", "--statistics"] + files,
            capture_output=True,
            text=True, check=False,
        )

        # Print the full output for debugging
        print("Initial Ruff check output:")
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return True, "No issues found. All files already conform to Ruff standards."

        # Capture the issues for reporting
        issues_found = result.stdout

        # Now run with --fix to auto-fix issues
        print("Running Ruff fix to automatically correct issues...")
        _ = subprocess.run(  # noqa: F841 (was fix_result)
            ["ruff", "check", "--select=E,F,W", "--fix"] + files,
            capture_output=True,
            text=True, check=False,
        )

        # Run check again to see what issues remain
        print("Running check again to see what issues remain...")
        after_result = subprocess.run(
            ["ruff", "check", "--select=E,F,W", "--statistics"] + files,
            capture_output=True,
            text=True, check=False,
        )

        print("After fix Ruff check output:")
        print(after_result.stdout)
        print(after_result.stderr)

        # Try to fix remaining issues with specific rules
        if after_result.returncode != 0:
            print("Applying more specific fixes for remaining issues...")
            # Try fixing just unused imports (F401)
            subprocess.run(
                ["ruff", "check", "--select=F401", "--fix"] + files,
                capture_output=True,
                text=True, check=False,
            )

            # Try fixing just line length issues (E501)
            subprocess.run(
                ["ruff", "check", "--select=E501", "--fix"] + files,
                capture_output=True,
                text=True, check=False,
            )

            # Run one final check
            final_result = subprocess.run(
                ["ruff", "check", "--select=E,F,W", "--statistics"] + files,
                capture_output=True,
                text=True, check=False,
            )

            if final_result.returncode == 0:
                return (
                    True,
                    f"All issues fixed successfully after multiple passes!\n\nPrevious issues:\n{issues_found}",
                )
            else:
                remaining_issues = final_result.stdout
                return False, (
                    f"Some issues were fixed, but others require manual attention.\n\n"
                    f"Original issues:\n{issues_found}\n\n"
                    f"Remaining issues:\n{remaining_issues}"
                )
        else:
            return (
                True,
                f"All issues fixed successfully!\n\nPrevious issues:\n{issues_found}",
            )

    except Exception as e:
        return False, f"Error running Ruff: {str(e)}"


def generate_report(success, message, output_file=None):
    """Generate a Markdown report of the Ruff fix results."""
    report = "# Service Directory Ruff Auto-fix Report\n\n"

    if success:
        report += "## ✅ Success\n\n"
    else:
        report += "## ⚠️ Partial Success\n\n"

    report += message.replace("\n", "\n\n")

    report += "\n\n## How to Fix Remaining Issues\n\n"
    report += "If any issues remain, you can fix them manually by:\n\n"
    report += "1. Running `ruff check ./service` locally to see all issues\n"
    report += "2. Running `ruff check --fix ./service` to fix auto-fixable issues\n"
    report += "3. Manually editing files that require attention\n\n"

    report += "## Specific Issues to Fix\n\n"
    report += "Common Ruff issues include:\n\n"
    report += "- Unused imports (F401): Remove imports that aren't used\n"
    report += "- Line too long (E501): Break long lines into multiple lines\n"
    report += "- Missing whitespace (E231): Add space after commas\n"
    report += "- Trailing whitespace (W291): Remove spaces at end of lines\n\n"

    report += "## Setting Up Pre-commit Hooks\n\n"
    report += "To prevent future issues, set up pre-commit hooks:\n\n"
    report += "```bash\n"
    report += "# On Linux/macOS/Git Bash\n"
    report += "./.github/setup-hooks.sh\n\n"
    report += "# On Windows PowerShell\n"
    report += "./.github/setup-hooks.ps1\n"
    report += "```\n"

    print(report)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)

    return report


def main():
    """Main entry point for the script."""
    # Create directories for artifacts if running in CI
    output_file = "ci_artifacts/linting/service_ruff_report.md"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Run Ruff fix on service directory
    success, message = run_ruff_fix()

    # Generate report
    generate_report(success, message, output_file)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
