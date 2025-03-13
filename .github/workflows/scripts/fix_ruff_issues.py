#!/usr/bin/env python3
"""
Ruff Auto-fix Script

This script automatically fixes Ruff linting issues in the codebase by:
1. Running Ruff with the --fix flag to auto-correct formatting issues
2. Generating a report of what was fixed and what needs manual attention
3. Supporting both local and CI environments
"""

import glob
import os
import subprocess
import sys


def find_python_files(directory="."):
    """Find all Python files in the given directory, excluding venvs and hidden dirs."""
    py_files = []
    exclude_patterns = ["**/venv/**", "**/.venv/**", "**/__pycache__/**", r"**/\.*/**"]

    # Find all Python files
    for py_file in glob.glob(f"{directory}/**/*.py", recursive=True):
        # Check if file should be excluded
        exclude = False
        for pattern in exclude_patterns:
            if glob.fnmatch.fnmatch(py_file, pattern):
                exclude = True
                break

        if not exclude:
            py_files.append(py_file)

    return py_files


def run_ruff_fix(files=None, directory="."):
    """Run Ruff with --fix flag to automatically fix issues."""
    if files is None:
        files = find_python_files(directory)

    if not files:
        print("No Python files found to fix.")
        return True, "No Python files found to fix."

    try:
        # First run Ruff check to see what issues exist
        print(f"Running Ruff check on {len(files)} Python files...")
        result = subprocess.run(
            ["ruff", "check"] + files, capture_output=True, text=True, check=False,
        )

        if result.returncode == 0:
            return True, "No issues found. All files already conform to Ruff standards."

        # Now run with --fix to auto-fix issues
        print("Running Ruff fix to automatically correct issues...")
        _ = subprocess.run(  # noqa: F841 (was fix_result)
            ["ruff", "check", "--fix"] + files, capture_output=True, text=True, check=False,
        )

        # Run check again to see what issues remain
        after_result = subprocess.run(
            ["ruff", "check"] + files, capture_output=True, text=True, check=False,
        )

        if after_result.returncode == 0:
            return True, "All issues fixed successfully!"
        return (
            False,
            f"Some issues were fixed, but others require manual attention:\n{after_result.stdout}",
        )

    except Exception as e:
        return False, f"Error running Ruff: {str(e)}"


def generate_report(success, message, output_file=None):
    """Generate a Markdown report of the Ruff fix results."""
    report = "# Ruff Auto-fix Report\n\n"

    if success:
        report += "## ✅ Success\n\n"
    else:
        report += "## ⚠️ Partial Success\n\n"

    report += message.replace("\n", "\n\n")

    report += "\n\n## How to Fix Remaining Issues\n\n"
    report += "If any issues remain, you can fix them manually by:\n\n"
    report += "1. Running `ruff check .` locally to see all issues\n"
    report += "2. Running `ruff check --fix .` to fix auto-fixable issues\n"
    report += "3. Manually editing files that require attention\n\n"

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
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python fix_ruff_issues.py [directory] [output_file]")
        print("If no directory is specified, the current directory is used.")
        print("If output_file is specified, a report will be written to that file.")
        return 0

    directory = "."
    output_file = None

    if len(sys.argv) > 1:
        directory = sys.argv[1]

    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Create directories for artifacts if they don't exist
    if in_ci_environment():
        os.makedirs("ci_artifacts/linting", exist_ok=True)
        if not output_file:
            output_file = "ci_artifacts/linting/ruff_autofix_report.md"

    # Run Ruff fix
    success, message = run_ruff_fix(directory=directory)

    # Generate report
    generate_report(success, message, output_file)

    # Exit with appropriate code
    return 0 if success else 1


def in_ci_environment():
    """Check if the script is running in a CI environment."""
    return (
        os.environ.get("CI", "false").lower() == "true"
        or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
    )


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
