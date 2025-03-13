#!/usr/bin/env python3
"""
Script to track technical debt progress over time.

This script:
1. Analyzes the codebase for linting issues
2. Tracks progress over time
3. Generates reports and visualizations
4. Provides recommendations for next steps
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class TechnicalDebtTracker:
    def __init__(self):
        self.history_file = Path("technical_debt_history.json")
        self.categories = {
            "high_priority": ["F401", "W291", "F821", "N801", "N802", "N803"],
            "medium_priority": ["C901", "E501", "E402", "RET503", "RET504", "RET505"],
            "low_priority": ["S*", "PTH*", "SIM*"],
        }

    def load_history(self) -> List[Dict]:
        """Load historical technical debt data."""
        if self.history_file.exists():
            return json.loads(self.history_file.read_text())
        return []

    def save_history(self, data: List[Dict]) -> None:
        """Save technical debt history."""
        self.history_file.write_text(json.dumps(data, indent=2))

    def analyze_codebase(self) -> Dict:
        """Analyze current state of technical debt."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "issues": {},
            "total_files": 0,
            "affected_files": 0,
        }

        # Run ruff to get current issues
        for category, rules in self.categories.items():
            category_issues = 0
            affected_files = set()

            for rule in rules:
                cmd = ["ruff", "check", ".", f"--select={rule}", "--format=json"]
                try:
                    output = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if output.returncode == 0:
                        continue

                    violations = json.loads(output.stdout) if output.stdout else []
                    category_issues += len(violations)
                    affected_files.update(v["filename"] for v in violations)
                except Exception as e:
                    print(f"Error analyzing rule {rule}: {e}")

            results["issues"][category] = category_issues
            results["affected_files"] = len(affected_files)

        # Count total Python files
        results["total_files"] = len(list(Path().rglob("*.py")))

        return results

    def generate_report(self, current: Dict, history: List[Dict]) -> str:
        """Generate a technical debt report."""
        report = [
            "# Technical Debt Report",
            f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Current Status",
            f"- Total Python files: {current['total_files']}",
            f"- Files with issues: {current['affected_files']}",
            f"- Clean files: {current['total_files'] - current['affected_files']}",
            "\n## Issues by Priority",
        ]

        for category, count in current["issues"].items():
            report.append(f"- {category.replace('_', ' ').title()}: {count} issues")

        if len(history) > 1:
            report.extend(
                [
                    "\n## Trend Analysis",
                    "Issue count over time:",
                ]
            )
            for record in history[-5:]:  # Show last 5 records
                date = datetime.fromisoformat(record["timestamp"]).strftime("%Y-%m-%d")
                total_issues = sum(record["issues"].values())
                report.append(f"- {date}: {total_issues} total issues")

        report.extend(
            [
                "\n## Recommendations",
                "1. High Priority Fixes:",
                "   - Focus on unused imports (F401)",
                "   - Clean up trailing whitespace (W291)",
                "   - Fix undefined names (F821)",
                "2. Medium Priority Improvements:",
                "   - Reduce code complexity (C901)",
                "   - Address line length issues (E501)",
                "   - Fix import placements (E402)",
                "3. Long-term Goals:",
                "   - Review and fix security issues (S*)",
                "   - Improve path handling (PTH*)",
                "   - Enhance code structure (SIM*)",
            ]
        )

        return "\n".join(report)

    def run(self) -> None:
        """Run the technical debt analysis and generate reports."""
        print("🔍 Analyzing technical debt...")

        # Load history and analyze current state
        history = self.load_history()
        current = self.analyze_codebase()

        # Update history
        history.append(current)
        self.save_history(history)

        # Generate and save report
        report = self.generate_report(current, history)
        report_path = Path("technical_debt_report.md")
        report_path.write_text(report)

        print(f"\n✅ Report saved to {report_path}")
        print("\nSummary:")
        print(f"- Total files: {current['total_files']}")
        print(f"- Files with issues: {current['affected_files']}")
        for category, count in current["issues"].items():
            print(f"- {category.replace('_', ' ').title()}: {count} issues")


def main():
    """Run the technical debt tracker."""
    tracker = TechnicalDebtTracker()
    tracker.run()


if __name__ == "__main__":
    main()
