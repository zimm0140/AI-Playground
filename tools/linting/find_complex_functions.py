#!/usr/bin/env python3
"""
Find Complex Functions

This script identifies complex functions in the codebase based on cyclomatic complexity.
It lists the functions sorted by complexity to help prioritize refactoring efforts.

Usage:
    python -m tools.linting.find_complex_functions [--limit N]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_complexity_check() -> List[Dict]:
    """Run ruff to check for complex functions and parse the output."""
    complex_functions = []

    try:
        # Run ruff to check for complex functions - don't specify output format
        cmd = [sys.executable, "-m", "ruff", "check", ".", "--select", "C901"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0 and not result.stdout:
            logger.error("Failed to check for complex functions")
            return complex_functions

        # Parse the output line by line
        for line in result.stdout.strip().split("\n"):
            if "C901" in line:
                try:
                    # Parse the line to extract information
                    parts = line.split(":")
                    if len(parts) < 3:
                        continue

                    file_path = parts[0]

                    # Extract line number
                    try:
                        line_num = int(parts[1])
                    except ValueError:
                        # If we can't parse the line number, skip this line
                        continue

                    # Extract function name and complexity
                    message = ":".join(parts[2:])
                    if "C901" in message:
                        message = message.split("C901")[1].strip()

                    function_name = "unknown"
                    if "'" in message:
                        try:
                            function_name = message.split("'")[1]
                        except IndexError:
                            pass

                    # Extract complexity value
                    complexity = 0
                    if "is too complex" in message:
                        try:
                            complexity_str = message.split("(")[-1].split(")")[0]
                            complexity = int(complexity_str)
                        except (ValueError, IndexError):
                            # Use a default complexity if we can't parse it
                            complexity = 10

                    complex_functions.append(
                        {
                            "file": file_path,
                            "line": line_num,
                            "function": function_name,
                            "complexity": complexity,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error parsing line: {line} - {e}")

    except Exception as e:
        logger.error(f"Error running complexity check: {e}")

    # Sort by complexity (descending) and then by file path
    complex_functions.sort(key=lambda x: (-x["complexity"], x["file"]))

    return complex_functions


def display_functions(functions: List[Dict], limit: int = None) -> None:
    """Display complex functions in a formatted table."""
    if not functions:
        logger.info("No complex functions found")
        return

    # Display limited number if specified
    display_functions = functions[:limit] if limit else functions

    # Print header
    print("\nComplex Functions by Cyclomatic Complexity")
    print("-" * 80)
    print(f"{'Function Name':<30} {'Complexity':<10} {'File':<30} {'Line'}")
    print("-" * 80)

    # Print function details
    for func in display_functions:
        # Shorten file path if needed
        file_path = func["file"]
        if len(file_path) > 30:
            # Try to show the most relevant part of the path
            parts = file_path.split(os.sep)
            if len(parts) > 2:
                file_path = f"...{os.sep}{os.sep.join(parts[-2:])}"

        # Truncate function name if too long
        function_name = func["function"]
        if len(function_name) > 28:
            function_name = function_name[:25] + "..."

        print(f"{function_name:<30} {func['complexity']:<10} {file_path:<30} {func['line']}")

    # Print summary
    print("-" * 80)
    print(f"Total complex functions: {len(functions)}")
    if limit and limit < len(functions):
        print(f"Showing top {limit} of {len(functions)} functions")
    print()


def generate_refactoring_suggestions(functions: List[Dict], limit: int = 5) -> None:
    """Generate refactoring suggestions for the most complex functions."""
    if not functions:
        return

    top_functions = functions[:limit]

    print("\nRefactoring Suggestions for Top Complex Functions")
    print("=" * 80)

    for idx, func in enumerate(top_functions, 1):
        print(f"\n{idx}. Function: {func['function']} (Complexity: {func['complexity']})")
        print(f"   File: {func['file']}:{func['line']}")
        print("   Suggestions:")

        # General suggestions based on complexity level
        if func["complexity"] > 30:
            print(
                "   - CRITICAL: This function is extremely complex and should be split into multiple smaller functions",
            )
            print("   - Consider creating a separate class to encapsulate this functionality")
            print("   - Identify independent logical sections that can be extracted into helper functions")
        elif func["complexity"] > 20:
            print("   - HIGH: Break this function into 3-5 smaller functions with clear responsibilities")
            print("   - Look for logical groupings of operations that can be extracted")
            print("   - Consider replacing complex conditional chains with strategy or state patterns")
        elif func["complexity"] > 10:
            print("   - MEDIUM: Identify 1-2 helper functions that can be extracted")
            print("   - Simplify conditional logic by using early returns or guard clauses")
            print("   - Consider if lookup tables or dictionaries can replace switch/if-else chains")
        else:
            print("   - LOW: Review for simpler conditional expressions or loop simplifications")
            print("   - Consider using list comprehensions or built-in functions where appropriate")

    print("\nGeneral Refactoring Techniques:")
    print("1. Extract Method: Move portions of code into separate, well-named methods")
    print("2. Replace Conditionals: Use polymorphism or dictionaries instead of large if/else chains")
    print("3. Simplify Logic: Combine or eliminate redundant conditions")
    print("4. Reduce Nesting: Extract deeply nested code into separate functions")
    print("5. Use Guard Clauses: Return early for edge cases rather than nesting conditions")
    print("6. Add Documentation: Clearly document complex parts that cannot be simplified further")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find complex functions in the codebase")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of functions to display",
    )
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Generate refactoring suggestions for top complex functions",
    )
    args = parser.parse_args()

    logger.info("Finding complex functions...")
    complex_functions = run_complexity_check()

    display_functions(complex_functions, args.limit)

    if args.suggestions:
        generate_refactoring_suggestions(complex_functions, 5)

    logger.info("Done")


if __name__ == "__main__":
    main()
