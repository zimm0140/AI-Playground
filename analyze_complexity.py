#!/usr/bin/env python3
"""
Analyze Function Complexity

This script analyzes the codebase to find the most complex functions based on
cyclomatic complexity. It generates a detailed report of the top N most complex
functions, including their location, complexity score, and a summary of their
implementation.

Usage:
    python analyze_complexity.py [--top N] [--output FILE]
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


def run_complexity_check() -> List[Dict]:
    """Run Ruff to check for complex functions."""
    cmd = ["ruff", "check", "--select", "C901", "--output-format=json", "."]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0 and result.stdout:
        return json.loads(result.stdout)
    return []


def get_function_details(file_path: str, line_number: int) -> Tuple[str, int, str]:
    """
    Extract details about a function including its name, complexity, and a code snippet.
    
    Returns:
        Tuple containing (function_name, complexity, code_snippet)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Find the function name and definition
        function_line = lines[line_number - 1].strip()
        function_name = "unknown"
        
        # Extract function name from the line
        if "def " in function_line:
            function_name = function_line.split("def ")[1].split("(")[0].strip()
        elif "class " in function_line:
            function_name = function_line.split("class ")[1].split("(")[0].split(":")[0].strip()
        
        # Extract complexity from the error message
        complexity = 0
        
        # Find the complexity value in the error message using ruff again
        cmd = ["ruff", "check", "--select", "C901", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        for line in result.stdout.splitlines():
            if f":{line_number}:" in line and "C901" in line:
                complexity_str = line.split("too complex (")[1].split(" >")[0]
                try:
                    complexity = int(complexity_str)
                except ValueError:
                    pass
                break
        
        # Extract a code snippet (up to 10 lines)
        start_line = max(0, line_number - 1)
        end_line = min(len(lines), line_number + 9)
        code_snippet = "".join(lines[start_line:end_line])
        
        return function_name, complexity, code_snippet
    
    except Exception as e:
        print(f"Error getting function details for {file_path}:{line_number}: {e}")
        return "unknown", 0, ""


def get_most_complex_functions(top_n: int = 5) -> List[Dict]:
    """Find the top N most complex functions in the codebase."""
    violations = run_complexity_check()
    
    # Extract relevant information
    functions = []
    for violation in violations:
        file_path = violation["filename"]
        line_number = violation["location"]["row"]
        
        name, complexity, snippet = get_function_details(file_path, line_number)
        
        functions.append({
            "name": name,
            "file": file_path,
            "line": line_number,
            "complexity": complexity,
            "snippet": snippet
        })
    
    # Sort by complexity (highest first)
    functions.sort(key=lambda x: x["complexity"], reverse=True)
    
    # Return top N
    return functions[:top_n]


def generate_report(functions: List[Dict], output_file: str = None) -> str:
    """Generate a detailed report of the complex functions."""
    report = "# Most Complex Functions Report\n\n"
    report += "This report identifies the most complex functions in the codebase based on cyclomatic complexity.\n\n"
    
    for i, func in enumerate(functions, 1):
        report += f"## {i}. {func['name']} (Complexity: {func['complexity']})\n\n"
        report += f"**File:** {func['file']}\n\n"
        report += f"**Line:** {func['line']}\n\n"
        report += "**Code Snippet:**\n\n"
        report += f"```python\n{func['snippet']}```\n\n"
        
        report += "**Refactoring Suggestions:**\n\n"
        report += "1. Extract helper methods for cohesive operations\n"
        report += "2. Reduce nesting through early returns\n"
        report += "3. Simplify conditional logic\n"
        report += "4. Consider using a design pattern to reduce complexity\n\n"
        
        report += "---\n\n"
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {output_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze function complexity in the codebase")
    parser.add_argument("--top", type=int, default=5, help="Number of most complex functions to report")
    parser.add_argument("--output", help="Output file for the report (defaults to console output)")
    
    args = parser.parse_args()
    
    print(f"Finding the top {args.top} most complex functions...")
    functions = get_most_complex_functions(args.top)
    
    if not functions:
        print("No complex functions found.")
        return 1
    
    report = generate_report(functions, args.output)
    
    if not args.output:
        print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 