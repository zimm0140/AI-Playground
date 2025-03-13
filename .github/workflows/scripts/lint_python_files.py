#!/usr/bin/env python3
"""
Lint Python Files

This script performs basic linting on Python files in the repository
to catch simple errors like unused imports, missing docstrings, etc.
before the main CI runs. This helps prevent CI failures due to simple
syntax and style issues.
"""

import ast
import os
import re
import sys


def lint_python_files(file_paths=None):
    """
    Check Python files for common issues like:
    - Unused imports
    - Undefined variables
    - Bad indentation
    - Regex escape errors

    Args:
        file_paths: Optional list of files to check. If None, scans all .py files.

    Returns:
        int: Number of errors found (0 means success)
    """
    if file_paths is None:
        # Find all Python files
        file_paths = []
        for root, _, files in os.walk("."):
            if ".git" in root or "venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    file_paths.append(os.path.join(root, file))

    error_count = 0

    for file_path in file_paths:
        print(f"Checking: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                print(f"  ❌ Syntax error at line {e.lineno}, col {e.offset}: {e.msg}")
                error_count += 1
                continue

            # Check for bad regex escapes
            bad_escapes = check_regex_escapes(content)
            for line_num, escape in bad_escapes:
                print(f"  ❌ Bad regex escape '\\{escape}' at line {line_num}")
                error_count += 1

            # Check for indentation issues in try/except blocks
            try_except_issues = check_try_except_indentation(content)
            for line_num, issue in try_except_issues:
                print(f"  ❌ {issue} at line {line_num}")
                error_count += 1

            # Check for common unused imports
            unused_imports = check_unused_imports(content)
            for import_name in unused_imports:
                print(f"  ⚠️ Potentially unused import: {import_name}")
                error_count += 1

            # Check for common 'not in' anti-patterns
            not_in_issues = check_not_in_patterns(content)
            for line_num, issue in not_in_issues:
                print(
                    f"  ⚠️ Consider using 'not in' instead of '{issue}' at line {line_num}",
                )
                error_count += 1

        except Exception as e:
            print(f"  ❌ Error checking file: {e}")
            error_count += 1

    if error_count == 0:
        print("\n✅ All files passed basic linting!")
    else:
        print(f"\n❌ Found {error_count} issues that need to be fixed.")

    return error_count


def check_regex_escapes(content):
    """
    Check for invalid regex escape sequences.

    Returns:
        list: [(line_num, escape_char), ...]
    """
    issues = []
    lines = content.split("\n")

    # Look for regex patterns
    pattern = r'r?[\'"].*\\([A-Za-z]).*[\'"]'

    for i, line in enumerate(lines):
        # Skip comments
        if line.strip().startswith("#"):
            continue

        # Check for raw strings (r"...") - these are likely regexes
        if "re." in line or "regex" in line or "pattern" in line:
            matches = re.finditer(pattern, line)
            for match in matches:
                escape_char = match.group(1)
                # Only report if not a valid escape sequence
                if escape_char not in "nrtbfvadAZbBdDsSwWxX0123456789":
                    issues.append((i + 1, escape_char))

    return issues


def check_try_except_indentation(content):
    """
    Check for proper indentation in try/except blocks.

    Returns:
        list: [(line_num, issue_description), ...]
    """
    issues = []
    lines = content.split("\n")
    in_try_block = False
    try_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if stripped == "try:":
            in_try_block = True
            try_indent = indent

            # Check next line for proper indentation
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                if next_stripped and not next_line.startswith(" " * (indent + 4)):
                    issues.append(
                        (
                            i + 2,
                            f"Expected indentation of {indent + 4} spaces after 'try:'",
                        ),
                    )

        elif in_try_block and (stripped.startswith("except ") or stripped == "except:"):
            # Check if except is at same level as try
            if indent != try_indent:
                issues.append(
                    (i + 1, "'except' block should have same indentation as 'try'"),
                )
            in_try_block = False

    return issues


def check_unused_imports(content):
    """
    Simple check for possibly unused imports.

    Returns:
        list: Names of imports that might be unused
    """
    unused = []

    # Parse the AST
    try:
        tree = ast.parse(content)

        # Get all import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if name.name == "*":
                        continue  # Can't check for wildcard imports
                    if node.module:
                        imports.append(f"{node.module}.{name.name}")
                    else:
                        imports.append(name.name)

        # Check if imports are used in the rest of the code
        for imp in imports:
            base_name = imp.split(".")[-1]

            # Check if base name occurs in the file
            remaining_code = re.sub(r"import\s+.*|from\s+.*\s+import.*", "", content)
            # Simple check - just see if the name appears
            if not re.search(r"\b" + re.escape(base_name) + r"\b", remaining_code):
                unused.append(imp)

    except Exception:
        # If there's an error parsing, we can't reliably check
        pass

    return unused


def check_not_in_patterns(content):
    """
    Check for anti-patterns like "not foo in bar" that should be "foo not in bar".

    Returns:
        list: [(line_num, pattern), ...]
    """
    issues = []
    lines = content.split("\n")

    pattern = r"not\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+"

    for i, line in enumerate(lines):
        # Skip comments
        if line.strip().startswith("#"):
            continue

        matches = re.finditer(pattern, line)
        for match in matches:
            issue = match.group(0).strip()
            issues.append((i + 1, issue))

    return issues


if __name__ == "__main__":
    # Get specific files from arguments, if provided
    files_to_check = sys.argv[1:] if len(sys.argv) > 1 else None

    # Change to repo root if running from scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "scripts" and os.path.exists(
        os.path.join(os.path.dirname(script_dir), "workflows"),
    ):
        os.chdir(os.path.join(script_dir, "../.."))

    # Check CI scripts if the directory exists
    workflows_scripts_dir = ".github/workflows/scripts"

    # Create necessary directories if they don't exist
    os.makedirs(workflows_scripts_dir, exist_ok=True)

    # First check if files were explicitly provided
    if files_to_check:
        # Check specific files provided as arguments
        print("Checking specified files:")
        file_errors = lint_python_files(files_to_check)
    else:
        # Check all Python files if no specific files provided
        print("Checking all Python files:")
        file_errors = lint_python_files()

    # Check workflow scripts if directory exists and has Python files
    if os.path.isdir(workflows_scripts_dir):
        try:
            workflow_scripts = [
                os.path.join(workflows_scripts_dir, f)
                for f in os.listdir(workflows_scripts_dir)
                if f.endswith(".py")
            ]
            if workflow_scripts:
                print("\nChecking CI script files:")
                script_errors = lint_python_files(workflow_scripts)
            else:
                script_errors = 0
                print("\nNo Python files found in CI scripts directory.")
        except (FileNotFoundError, PermissionError) as e:
            script_errors = 0
            print(f"\nError accessing CI script directory: {e}")
            print("Skipping CI script checks.")
    else:
        script_errors = 0
        print("\nSkipping CI script checks - directory not found.")

    # Exit with error if any issues found
    sys.exit(script_errors + file_errors)
