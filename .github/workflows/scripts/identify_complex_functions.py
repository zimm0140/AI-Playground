#!/usr/bin/env python3
"""
Script to identify complex functions in the codebase.

This script:
1. Uses ruff to find functions with high complexity (C901)
2. Extracts detailed information about each function
3. Generates a prioritized refactoring report
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List


class ComplexityAnalyzer:
    def __init__(self):
        self.report_file = Path("complex_functions_report.md")

    def run_complexity_check(self) -> List[Dict]:
        """Run ruff to find complex functions."""
        try:
            result = subprocess.run(
                ["ruff", "check", ".", "--select=C901", "--output-format=json"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                return []

            violations = json.loads(result.stdout) if result.stdout else []
            return violations
        except Exception as e:
            print(f"Error running complexity check: {e}")
            return []

    def extract_function_details(self, violations: List[Dict]) -> List[Dict]:
        """Extract details about complex functions."""
        function_details = []

        for violation in violations:
            filename = violation.get("filename", "")
            location = violation.get("location", {})
            line = location.get("row", 0)
            message = violation.get("message", "")

            # Extract complexity from message (e.g., "Function is too complex (15 > 10)")
            complexity = 0
            try:
                complexity_part = message.split("(")[1].split(">")[0].strip()
                complexity = int(complexity_part)
            except (IndexError, ValueError):
                pass

            # Get function name from message
            function_name = ""
            if "Function " in message and " is too complex" in message:
                function_name = message.split("Function ")[1].split(" is too complex")[0].strip()

            function_details.append(
                {
                    "filename": filename,
                    "line": line,
                    "function_name": function_name,
                    "complexity": complexity,
                },
            )

        return function_details

    def prioritize_functions(self, function_details: List[Dict]) -> List[Dict]:
        """Prioritize functions for refactoring."""
        # Sort by complexity (descending)
        return sorted(function_details, key=lambda x: x.get("complexity", 0), reverse=True)

    def generate_report(self, functions: List[Dict]) -> str:
        """Generate a report of complex functions."""
        report = [
            "# Complex Functions Refactoring Report",
            "\nThis report identifies functions with high cyclomatic complexity.",
            "These functions are primary candidates for refactoring to improve maintainability.",
            "\n## Priority Refactoring List",
        ]

        for i, func in enumerate(functions[:20], 1):  # Top 20 most complex functions
            filename = func.get("filename", "")
            line = func.get("line", 0)
            function_name = func.get("function_name", "Unknown")
            complexity = func.get("complexity", 0)

            report.append(f"\n### {i}. {function_name} (Complexity: {complexity})")
            report.append(f"- File: `{filename}`")
            report.append(f"- Line: {line}")
            report.append("- Refactoring suggestions:")
            report.append("  - Break into smaller functions")
            report.append("  - Simplify conditional logic")
            report.append("  - Use helper functions for repeated code")

        report.extend(
            [
                "\n## Next Steps",
                "1. Start with the top 5 most complex functions",
                "2. Create unit tests before refactoring",
                "3. Refactor one function at a time",
                "4. Re-run complexity analysis after each refactoring",
            ],
        )

        return "\n".join(report)

    def run(self) -> None:
        """Run the complexity analysis and generate a report."""
        print("🔍 Analyzing function complexity...")

        violations = self.run_complexity_check()
        if not violations:
            print("✅ No complex functions found.")
            return

        print(f"Found {len(violations)} complex functions.")

        function_details = self.extract_function_details(violations)
        prioritized_functions = self.prioritize_functions(function_details)

        report = self.generate_report(prioritized_functions)
        self.report_file.write_text(report)

        print(f"\n✅ Report saved to {self.report_file}")
        print("\nTop 5 most complex functions:")
        for i, func in enumerate(prioritized_functions[:5], 1):
            print(f"{i}. {func.get('function_name', 'Unknown')} (Complexity: {func.get('complexity', 0)})")


def main():
    """Run the complexity analyzer."""
    analyzer = ComplexityAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()