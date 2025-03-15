#!/usr/bin/env python3
"""
Gradual Type Adoption Script

This script helps manage the gradual adoption of type checking across the codebase.
It identifies files that can be added to the type checking process and generates
configuration updates for mypy and pre-commit.
"""

import os
import re
import argparse
import subprocess
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


# Files we've already fixed
PRIORITY_FILES = [
    ".github/workflows/scripts/comment_on_workflow_pr.py",
    ".github/workflows/scripts/generate_workflow_docs.py",
    ".github/workflows/scripts/validate_components.py",
    ".github/workflows/scripts/fix_ci_issues.py"
]


def run_mypy_on_file(file_path: str) -> Tuple[bool, str]:
    """Run mypy on a single file and return the result.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple containing success status and output
    """
    result = subprocess.run(
        ["python", "-m", "mypy", "--config-file", "mypy.ini", file_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout


def count_errors(output: str) -> int:
    """Count the number of errors in mypy output.
    
    Args:
        output: mypy output text
        
    Returns:
        Number of errors found
    """
    # Look for the "Found X errors in Y files" line
    match = re.search(r"Found (\d+) errors? in", output)
    if match:
        return int(match.group(1))
    return 0


def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Find all Python files in a directory, excluding certain patterns.
    
    Args:
        directory: Directory to search
        exclude_patterns: List of patterns to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if the file matches any exclude pattern
                excluded = any(re.search(pattern, file_path) for pattern in exclude_patterns)
                if not excluded:
                    python_files.append(file_path)
    
    return python_files


def normalize_path(path: str) -> str:
    """Normalize file path to use forward slashes.
    
    Args:
        path: File path to normalize
        
    Returns:
        Normalized path
    """
    return path.replace("\\", "/")


def attempt_fix_typing_issues(file_path: str) -> Tuple[bool, int]:
    """Attempt to fix typing issues in a file using fix_typing_issues.py.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success, number of issues fixed)
    """
    try:
        print(f"  Attempting to fix typing issues in {file_path}")
        result = subprocess.run(
            ["python", "-m", "tools.fix_typing_issues", "--path", file_path],
            capture_output=True,
            text=True,
        )
        
        # Extract the number of fixed issues from output
        match = re.search(r"Fixed (\d+) typing issues? in", result.stdout)
        issues_fixed = int(match.group(1)) if match else 0
        
        # Check if the file passes mypy after fixing
        success, _ = run_mypy_on_file(file_path)
        
        if success:
            print(f"  ✓ Successfully fixed all typing issues in {file_path}")
            return True, issues_fixed
        else:
            print(f"  ✗ Could not fix all typing issues in {file_path}, fixed {issues_fixed} issues")
            return False, issues_fixed
            
    except Exception as e:
        print(f"  ✗ Error attempting to fix {file_path}: {str(e)}")
        return False, 0


def analyze_files(directory: str, exclude_patterns: List[str] = None, error_threshold: int = 3, 
                 auto_fix: bool = False) -> Dict[str, List[str]]:
    """Analyze files for type errors.
    
    Args:
        directory: Directory to analyze
        exclude_patterns: List of patterns to exclude
        error_threshold: Maximum number of errors allowed for a file to be considered "close to ready"
        auto_fix: Attempt to automatically fix typing issues in close files
        
    Returns:
        Dictionary with analysis results
    """
    python_files = find_python_files(directory, exclude_patterns)
    
    # Sort files to prioritize already fixed files first
    priority_map = {normalize_path(f): i for i, f in enumerate(PRIORITY_FILES)}
    python_files.sort(key=lambda f: priority_map.get(normalize_path(f), len(PRIORITY_FILES) + 1))
    
    results = {
        "ready": [],
        "close": [],
        "not_ready": [],
        "auto_fixed": [],
    }
    
    total_files = len(python_files)
    print(f"Found {total_files} Python files to analyze...")
    
    for i, file_path in enumerate(python_files):
        normalized_path = normalize_path(file_path)
        print(f"Analyzing {i+1}/{total_files}: {file_path}")
        
        # If it's a priority file we've already fixed, add it to ready without checking
        if normalized_path in priority_map:
            print(f"  Priority file: {normalized_path}")
            results["ready"].append(file_path)
            continue
            
        success, output = run_mypy_on_file(file_path)
        error_count = count_errors(output)
        
        if success:
            results["ready"].append(file_path)
        elif error_count <= error_threshold:
            # If auto-fix is enabled and the file is close to ready, attempt to fix it
            if auto_fix:
                fix_success, fixed_issues = attempt_fix_typing_issues(file_path)
                if fix_success:
                    results["auto_fixed"].append(file_path)
                else:
                    results["close"].append(file_path)
            else:
                results["close"].append(file_path)
        else:
            results["not_ready"].append(file_path)
    
    return results


def generate_pre_commit_regex(files: List[str]) -> str:
    """Generate regular expression for pre-commit configuration.
    
    Args:
        files: List of file paths
        
    Returns:
        Regular expression for pre-commit configuration
    """
    if not files:
        return "^()$"
        
    # Normalize paths to forward slashes and escape special characters
    normalized_files = [re.escape(normalize_path(file)) for file in files]
    return "^(" + "|".join(normalized_files) + ")$"


def generate_github_action_cmd(files: List[str]) -> str:
    """Generate command for GitHub Actions workflow.
    
    Args:
        files: List of file paths
        
    Returns:
        Command for GitHub Actions workflow
    """
    normalized_files = [normalize_path(file) for file in files]
    file_list = " ".join(normalized_files)
    return f"python -m mypy --config-file mypy.ini {file_list}"


def generate_reports(results: Dict[str, List[str]], output_file: Optional[str] = None) -> None:
    """Generate reports based on analysis results.
    
    Args:
        results: Analysis results
        output_file: Optional file to write report to
    """
    ready_files = results["ready"]
    close_files = results["close"]
    not_ready_files = results["not_ready"]
    auto_fixed_files = results.get("auto_fixed", [])
    
    # Include auto-fixed files in ready list for configuration generation
    all_ready_files = ready_files + auto_fixed_files
    
    total_files = len(ready_files) + len(close_files) + len(not_ready_files) + len(auto_fixed_files)
    
    if total_files == 0:
        print("No files found to analyze.")
        return
    
    # Normalize all paths to use forward slashes for consistency
    ready_files_normalized = [normalize_path(f) for f in ready_files]
    auto_fixed_files_normalized = [normalize_path(f) for f in auto_fixed_files]
    close_files_normalized = [normalize_path(f) for f in close_files]
    
    # All files that pass type checking (ready + auto-fixed)
    all_ready_normalized = ready_files_normalized + auto_fixed_files_normalized
    
    report = [
        "# Type Checking Adoption Report",
        "",
        f"Total Python files analyzed: {total_files}",
        f"Files ready for type checking: {len(ready_files)} ({len(ready_files) / total_files * 100:.1f}%)",
        f"Files automatically fixed: {len(auto_fixed_files)} ({len(auto_fixed_files) / total_files * 100:.1f}%)",
        f"Files close to ready: {len(close_files)} ({len(close_files) / total_files * 100:.1f}%)",
        f"Files not ready: {len(not_ready_files)} ({len(not_ready_files) / total_files * 100:.1f}%)",
        "",
        "## Pre-commit Configuration",
        "",
    ]
    
    if all_ready_normalized:
        report.extend([
            "Add these files to the mypy pre-commit hook:",
            "```yaml",
            "-   repo: https://github.com/pre-commit/mirrors-mypy",
            "    rev: v1.9.0",
            "    hooks:",
            "    -   id: mypy",
            "        additional_dependencies: [types-requests]",
            "        args: [--config-file=mypy.ini]",
            "        files: " + generate_pre_commit_regex(all_ready_files),
            "```",
            "",
        ])
    else:
        report.extend([
            "No files are ready for type checking yet.",
            "",
        ])
    
    report.extend([
        "## GitHub Actions Workflow",
        "",
    ])
    
    if all_ready_normalized:
        report.extend([
            "Add these files to the type-check job:",
            "```yaml",
            "- name: Run mypy",
            "  run: |",
            "    " + generate_github_action_cmd(all_ready_files),
            "```",
            "",
        ])
    else:
        report.extend([
            "No files are ready for GitHub Actions workflow yet.",
            "",
        ])
    
    report.extend([
        "## Files Ready for Type Checking",
        "",
    ])
    
    # Add ready files
    if ready_files_normalized:
        for file in ready_files_normalized:
            report.append(f"- {file}")
    else:
        report.append("No files are ready for type checking yet.")
    
    # Add auto-fixed files
    if auto_fixed_files_normalized:
        report.extend([
            "",
            "## Files Automatically Fixed",
            "",
        ])
        for file in auto_fixed_files_normalized:
            report.append(f"- {file}")
    
    # Add close files
    report.extend([
        "",
        "## Files Close to Ready",
        "",
    ])
    
    if close_files_normalized:
        for file in close_files_normalized:
            report.append(f"- {file}")
    else:
        report.append("No files are close to ready for type checking.")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"Report written to {output_file}")
    else:
        print(report_text)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Gradual Type Adoption Tool")
    parser.add_argument(
        "--directory",
        "-d",
        default=".",
        help="Directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        default=[r"node_modules/", r"\.venv/", r"venv/", r"build/", r"dist/"],
        help="Patterns to exclude (can be specified multiple times)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=3,
        help="Maximum number of errors for a file to be considered 'close to ready' (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the report (default: print to console)",
    )
    parser.add_argument(
        "--auto-fix",
        "-a",
        action="store_true",
        help="Attempt to automatically fix typing issues in close files",
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing directory: {args.directory}")
    print(f"Excluding patterns: {args.exclude}")
    print(f"Error threshold: {args.threshold}")
    print(f"Auto-fix enabled: {args.auto_fix}")
    
    results = analyze_files(args.directory, args.exclude, args.threshold, args.auto_fix)
    generate_reports(results, args.output)


if __name__ == "__main__":
    main() 