#!/usr/bin/env python3
"""
Type Issue Prioritization Tool

This script analyzes Python files to identify the most common typing issues
and prioritizes which files to fix next based on complexity, impact,
and estimated effort.
"""

import os
import re
import sys
import ast
import argparse
import subprocess
import json
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional, Any, Counter as CounterType
from pathlib import Path


def run_mypy_on_file(file_path: str) -> Tuple[bool, str]:
    """Run mypy on a single file and return the result.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple containing success status and output
    """
    # Use stricter options to find more typing issues
    result = subprocess.run(
        [
            "python", "-m", "mypy",
            "--config-file", "mypy.ini",
            "--disallow-untyped-defs",
            "--disallow-incomplete-defs",
            "--check-untyped-defs",
            "--disallow-untyped-decorators",
            "--no-implicit-optional",
            "--warn-redundant-casts",
            "--warn-unused-ignores",
            "--warn-return-any",
            file_path
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout


def parse_mypy_errors(output: str) -> List[Dict[str, Any]]:
    """Parse mypy error messages into structured data.
    
    Args:
        output: mypy output text
        
    Returns:
        List of error dictionaries with line, message and error type
    """
    errors = []
    
    # Regular expression to match mypy error lines
    # Format: file:line: error: message [error-code]
    pattern = r'(?P<file>.*?):(?P<line>\d+): (?P<level>\w+): (?P<message>.*?)(?:\s+\[(?P<code>[a-z0-9-]+)\])?$'
    
    for line in output.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            errors.append({
                'file': match.group('file'),
                'line': int(match.group('line')),
                'level': match.group('level'),
                'message': match.group('message'),
                'code': match.group('code')
            })
    
    return errors


class TypeIssueVisitor(ast.NodeVisitor):
    """AST visitor to find potential typing issues in Python code."""
    
    def __init__(self):
        self.issues = []
        self.function_count = 0
        self.typed_function_count = 0
        self.missing_return_type_count = 0
        self.missing_param_type_count = 0
        self.variable_count = 0
        self.typed_variable_count = 0
        self.class_count = 0
        self.typed_class_count = 0
        self.import_count = 0
        self.typing_import_count = 0
        self.current_function = None
        
    def visit_FunctionDef(self, node):
        """Visit function definitions to check for typing issues."""
        self.function_count += 1
        self.current_function = node.name
        
        # Check return type
        has_return_type = node.returns is not None
        if not has_return_type:
            self.missing_return_type_count += 1
            self.issues.append({
                'line': node.lineno,
                'message': f"Function '{node.name}' is missing a return type annotation",
                'category': 'missing_return_type'
            })
        
        # Check parameter types
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != 'self' and arg.arg != 'cls':
                self.missing_param_type_count += 1
                self.issues.append({
                    'line': arg.lineno if hasattr(arg, 'lineno') else node.lineno,
                    'message': f"Parameter '{arg.arg}' in function '{node.name}' is missing a type annotation",
                    'category': 'missing_param_type'
                })
        
        # Count functions with all parameters typed
        if has_return_type and all(arg.annotation is not None or arg.arg in ('self', 'cls') for arg in node.args.args):
            self.typed_function_count += 1
            
        # Visit function body
        self.generic_visit(node)
        self.current_function = None
    
    def visit_ClassDef(self, node):
        """Visit class definitions to check for typing issues."""
        self.class_count += 1
        
        # Check if class inherits from a typing class (Protocol, Generic, etc.)
        has_typing_base = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ('Protocol', 'Generic', 'TypedDict'):
                has_typing_base = True
                break
        
        if has_typing_base:
            self.typed_class_count += 1
            
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Visit annotated assignments to count typed variables."""
        self.variable_count += 1
        self.typed_variable_count += 1
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignments to count untyped variables."""
        # Only count module-level or class-level variables, not function locals
        if self.current_function is None:
            self.variable_count += len(node.targets)
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements."""
        self.import_count += len(node.names)
        for name in node.names:
            if name.name == 'typing' or name.name.startswith('typing.'):
                self.typing_import_count += 1
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        self.import_count += len(node.names)
        if node.module == 'typing' or (node.module and node.module.startswith('typing.')):
            self.typing_import_count += len(node.names)
        self.generic_visit(node)


def analyze_file_ast(file_path: str) -> Dict[str, Any]:
    """Analyze typing issues in a file using AST.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the code into an AST
        tree = ast.parse(content, filename=file_path)
        
        # Visit the AST to find typing issues
        visitor = TypeIssueVisitor()
        visitor.visit(tree)
        
        # Also count lines of code
        loc = len(content.splitlines())
        
        # Calculate basic metrics
        typed_function_ratio = visitor.typed_function_count / visitor.function_count if visitor.function_count > 0 else 1.0
        typed_variable_ratio = visitor.typed_variable_count / visitor.variable_count if visitor.variable_count > 0 else 1.0
        
        # Count total issues
        total_issues = len(visitor.issues)
        
        # Extract issues by category
        issue_categories = Counter(issue['category'] for issue in visitor.issues)
        
        # Calculate "typing completeness" score (0-100)
        functions_score = typed_function_ratio * 100
        variables_score = typed_variable_ratio * 100
        typing_score = (functions_score * 0.7 + variables_score * 0.3)
        
        # Calculate effort required to fix issues (higher = more effort)
        effort = (
            visitor.missing_return_type_count * 1.0 +
            visitor.missing_param_type_count * 1.5 +
            (visitor.variable_count - visitor.typed_variable_count) * 1.2
        )
        
        return {
            "file": file_path,
            "total_issues": total_issues,
            "loc": loc,
            "function_count": visitor.function_count,
            "typed_function_count": visitor.typed_function_count,
            "function_type_coverage": typed_function_ratio * 100,
            "variable_count": visitor.variable_count,
            "typed_variable_count": visitor.typed_variable_count,
            "variable_type_coverage": typed_variable_ratio * 100,
            "class_count": visitor.class_count,
            "typed_class_count": visitor.typed_class_count,
            "import_count": visitor.import_count,
            "typing_import_count": visitor.typing_import_count,
            "missing_return_types": visitor.missing_return_type_count,
            "missing_param_types": visitor.missing_param_type_count,
            "typing_score": typing_score,
            "issues": visitor.issues,
            "issue_categories": dict(issue_categories),
            "effort": effort
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {
            "file": file_path,
            "error": str(e),
            "total_issues": 0,
            "loc": 0,
            "function_count": 0,
            "typed_function_count": 0,
            "function_type_coverage": 100,
            "variable_count": 0,
            "typed_variable_count": 0,
            "variable_type_coverage": 100,
            "typing_score": 100,
            "issues": [],
            "issue_categories": {},
            "effort": 0
        }


def categorize_error(error: Dict[str, Any]) -> str:
    """Categorize an error by its type.
    
    Args:
        error: Dictionary containing error information
        
    Returns:
        Error category
    """
    message = error['message'].lower()
    
    # Common error categories
    if "missing a return type annotation" in message:
        return "missing_return_type"
    elif "missing a type annotation" in message:
        return "missing_function_annotation"
    elif "need type annotation for" in message:
        return "missing_variable_type"
    elif 'item "none" of "optional' in message and 'has no attribute' in message:
        return "none_attribute_access"
    elif "incompatible types in assignment" in message:
        return "incompatible_types"
    elif "returning any from function declared to return" in message:
        return "returning_any"
    elif "argument" in message and "has incompatible type" in message:
        return "incompatible_argument"
    elif "has no attribute" in message:
        return "missing_attribute"
    elif "undefined name" in message:
        return "undefined_name"
    elif "module has no attribute" in message:
        return "missing_module_attribute"
    elif "is not a known member of" in message:
        return "unknown_member"
    else:
        return "other"


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


def calculate_impact_score(file_data: Dict[str, Any]) -> float:
    """Calculate the impact score of fixing a file.
    
    Args:
        file_data: File analysis data
        
    Returns:
        Impact score (higher means higher impact)
    """
    # Files with more functions and LOC likely have higher impact when fixed
    function_weight = 2.0
    loc_weight = 0.01
    missing_types_weight = 0.5
    
    # Calculate impact based on functions, LOC and number of issues to fix
    impact = (
        file_data["function_count"] * function_weight +
        file_data["loc"] * loc_weight +
        (file_data["missing_return_types"] + file_data["missing_param_types"]) * missing_types_weight
    )
    
    return impact


def calculate_priority_score(file_data: Dict[str, Any]) -> float:
    """Calculate the priority score for fixing a file.
    
    Args:
        file_data: File analysis data
        
    Returns:
        Priority score (higher means higher priority)
    """
    # If typing score is 100, file is already fully typed
    if file_data["typing_score"] >= 99.9:
        return 0.0
    
    # Calculate impact of fixing the file
    impact = calculate_impact_score(file_data)
    
    # Calculate typing gap (100 - current typing score)
    typing_gap = 100 - file_data["typing_score"]
    
    # Calculate efficiency (impact / effort)
    # Higher efficiency means more impact for less effort
    if file_data["effort"] > 0:
        efficiency = impact / file_data["effort"]
    else:
        efficiency = impact
    
    # Prioritize files with better efficiency and moderate typing gap
    # Files with very low typing score might be too challenging to start with
    typing_gap_factor = typing_gap / 100 * (1 - (typing_gap / 200))  # Peaks around 50% typing
    
    priority = efficiency * typing_gap_factor * (file_data["function_count"] + 1)
    
    return priority


def analyze_directory(directory: str, exclude_patterns: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Analyze all Python files in a directory and prioritize which to fix.
    
    Args:
        directory: Directory to analyze
        exclude_patterns: List of patterns to exclude
        limit: Maximum number of files to include in the report
        
    Returns:
        List of file analysis data sorted by priority
    """
    python_files = find_python_files(directory, exclude_patterns)
    
    # Get the already passing priority files (defined in .pre-commit-config.yaml)
    priority_files = [
        ".github/workflows/scripts/comment_on_workflow_pr.py",
        ".github/workflows/scripts/generate_workflow_docs.py",
        ".github/workflows/scripts/validate_components.py",
        ".github/workflows/scripts/fix_ci_issues.py"
    ]
    
    # Check each file first, only excluded those that are actually passing
    already_passing = []
    for file_path in priority_files:
        success, _ = run_mypy_on_file(file_path)
        if success:
            already_passing.append(file_path)
            print(f"Skipping already type-checked file: {file_path}")
    
    # Filter out only the passing priority files
    python_files = [f for f in python_files if f not in already_passing]
    
    print(f"Analyzing {len(python_files)} Python files...")
    
    # Analyze each file
    file_data = []
    for i, file_path in enumerate(python_files):
        print(f"Analyzing {i+1}/{len(python_files)}: {file_path}")
        data = analyze_file_ast(file_path)
        
        # Calculate priority score
        data["priority"] = calculate_priority_score(data)
        
        file_data.append(data)
    
    # Sort by priority score in descending order
    file_data.sort(key=lambda x: x["priority"], reverse=True)
    
    # Return the top N files by priority
    return file_data[:limit]


def generate_report(file_data: List[Dict[str, Any]], output_file: Optional[str] = None) -> None:
    """Generate a prioritization report.
    
    Args:
        file_data: List of file analysis data
        output_file: Optional file to write report to
    """
    if not file_data:
        print("No files analyzed.")
        return
    
    # Count total issues by category across all files
    all_issues = Counter()
    for data in file_data:
        for category, count in data.get("issue_categories", {}).items():
            all_issues[category] += count
    
    # Generate report
    report = [
        "# Type Fixing Prioritization Report",
        "",
        "## Most Common Issue Types",
        "",
        "| Issue Type | Count | Description |",
        "|------------|-------|-------------|",
    ]
    
    # Issue type descriptions
    issue_descriptions = {
        "missing_return_type": "Function is missing a return type annotation",
        "missing_param_type": "Function parameter is missing a type annotation",
        "missing_function_annotation": "Function is missing parameter type annotations",
        "missing_variable_type": "Variable needs type annotation",
        "none_attribute_access": "Accessing attribute on a potentially None value",
        "incompatible_types": "Incompatible types in assignment",
        "returning_any": "Returning Any from function with explicit return type",
        "incompatible_argument": "Function argument has incompatible type",
        "missing_attribute": "Attribute doesn't exist on the object",
        "undefined_name": "Variable name is not defined in the scope",
        "missing_module_attribute": "Module does not have the referenced attribute",
        "unknown_member": "Member is not a known part of the module or class",
        "other": "Other miscellaneous typing issues"
    }
    
    # Add issue categories to report
    for category, count in all_issues.most_common():
        description = issue_descriptions.get(category, "Unknown issue type")
        report.append(f"| {category} | {count} | {description} |")
    
    report.extend([
        "",
        "## Prioritized Files to Fix",
        "",
        "| File | Functions | Type Coverage (%) | Missing Types | LOC | Priority | Description |",
        "|------|-----------|-------------------|---------------|-----|----------|-------------|",
    ])
    
    # Add prioritized files to report
    for data in file_data:
        # Skip files with perfect typing
        if data["typing_score"] >= 99.9:
            continue
            
        file_name = data["file"]
        function_count = data["function_count"]
        typing_score = data["typing_score"]
        missing_types = data["missing_return_types"] + data["missing_param_types"]
        loc = data["loc"]
        priority = data["priority"]
        
        # Generate a description of the file's typing status
        if typing_score < 30:
            description = "Major typing issues, needs complete overhaul"
        elif typing_score < 60:
            description = "Moderate typing coverage, many functions need annotations"
        elif typing_score < 85:
            description = "Good typing foundation, needs completion"
        else:
            description = "Nearly complete typing, minor fixes needed"
        
        report.append(f"| {file_name} | {function_count} | {typing_score:.1f} | {missing_types} | {loc} | {priority:.2f} | {description} |")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"Report written to {output_file}")
    else:
        print(report_text)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Prioritize typing fixes based on impact and effort")
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
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of files to include in the report (default: 20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the report (default: print to console)",
    )
    parser.add_argument(
        "--json",
        "-j",
        help="Output file for JSON results",
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing directory: {args.directory}")
    print(f"Excluding patterns: {args.exclude}")
    
    file_data = analyze_directory(args.directory, args.exclude, args.limit)
    
    # Save JSON results if requested
    if args.json:
        # Make the data JSON serializable
        serializable_data = []
        for item in file_data:
            item_copy = item.copy()
            item_copy.pop('issues', None)  # Remove the detailed issues list
            serializable_data.append(item_copy)
            
        with open(args.json, "w") as f:
            json.dump(serializable_data, f, indent=2)
        print(f"JSON data written to {args.json}")
    
    generate_report(file_data, args.output)


if __name__ == "__main__":
    main() 