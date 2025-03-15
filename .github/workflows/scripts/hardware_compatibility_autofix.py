#!/usr/bin/env python
"""
Hardware Compatibility Auto-fix Tool

This script automatically applies recommendations from the Hardware Compatibility
Advisor to standardize package versions across hardware platforms.

Features:
- Automatic application of high-priority recommendations
- Generation of fixed requirements files
- Creation of a summary report of applied changes
- Integration with the existing hardware compatibility system
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Any


class HardwareCompatibilityAutofix:
    def __init__(
        self,
        input_dir: str = "ci_artifacts/hardware_compatibility",
        recommendations_dir: str = "ci_artifacts/hardware_compatibility/recommendations",
        output_dir: str = "ci_artifacts/hardware_compatibility/autofix",
        resolution_plan: str = "resolution_plan.md",
        data_file: str = "hardware_compatibility_data.json",
        backup_suffix: str = ".bak",
        high_priority_only: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize the hardware compatibility auto-fix tool.

        Args:
            input_dir: Directory containing compatibility test results
            recommendations_dir: Directory containing advisor recommendations
            output_dir: Directory to store auto-fix results
            resolution_plan: Markdown file with resolution plan
            data_file: JSON file with compatibility test data
            backup_suffix: Suffix for backup files
            high_priority_only: Whether to only apply high priority fixes
            dry_run: Whether to simulate changes without applying them
        """
        self.input_dir = input_dir
        self.recommendations_dir = recommendations_dir
        self.output_dir = output_dir
        self.resolution_plan_file = os.path.join(recommendations_dir, resolution_plan)
        self.data_file = os.path.join(input_dir, data_file)
        self.backup_suffix = backup_suffix
        self.high_priority_only = high_priority_only
        self.dry_run = dry_run

        self.compatibility_data = {}
        self.resolution_plan = {}
        self.applied_changes = defaultdict(list)
        self.skipped_changes = defaultdict(list)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_compatibility_data(self) -> bool:
        """
        Load compatibility test data from JSON file.

        Returns:
            Boolean indicating whether data was successfully loaded
        """
        if not os.path.exists(self.data_file):
            print(f"Error: Compatibility data file not found: {self.data_file}")
            print("Please run hardware_compatibility_tester.py first.")
            return False

        try:
            with open(self.data_file, encoding="utf-8") as f:
                self.compatibility_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading compatibility data: {e}")
            return False

    def load_recommendations(self) -> bool:
        """
        Load advisor recommendations from resolution plan file.

        Returns:
            Boolean indicating whether recommendations were successfully loaded
        """
        if not os.path.exists(self.resolution_plan_file):
            print(f"Error: Resolution plan file not found: {self.resolution_plan_file}")
            print("Please run hardware_compatibility_advisor.py first.")
            return False

        # Since the resolution plan is in markdown format, we need to extract the data sections
        try:
            # Load the raw resolution plan file content
            with open(self.resolution_plan_file, encoding="utf-8") as f:
                content = f.read()

            # Extract high priority recommendations section
            high_priority_match = re.search(
                r"## High Priority Recommendations\s*\n\n(.*?)(?=\n\n##|\Z)",
                content,
                re.DOTALL,
            )

            if high_priority_match:
                high_priority_text = high_priority_match.group(1)
                # Parse the recommendations
                self.resolution_plan["high_priority"] = self._parse_recommendations(
                    high_priority_text,
                )
            else:
                self.resolution_plan["high_priority"] = []

            # If we want medium priority recommendations too
            if not self.high_priority_only:
                medium_priority_match = re.search(
                    r"## Medium Priority Recommendations\s*\n\n(.*?)(?=\n\n##|\Z)",
                    content,
                    re.DOTALL,
                )

                if medium_priority_match:
                    medium_priority_text = medium_priority_match.group(1)
                    # Parse the recommendations
                    self.resolution_plan[
                        "medium_priority"
                    ] = self._parse_recommendations(medium_priority_text)
                else:
                    self.resolution_plan["medium_priority"] = []

            return True

        except Exception as e:
            print(f"Error loading recommendations: {e}")
            return False

    def _parse_recommendations(self, text: str) -> List[Dict[str, str]]:
        """
        Parse recommendations from markdown text.

        Args:
            text: Markdown text containing recommendations

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Split by package sections (### PackageName)
        package_sections = re.split(r"### ([\w-]+)\s*\n", text)[
            1:
        ]  # Skip the first empty element

        # Process each package section
        for i in range(0, len(package_sections), 2):
            if i + 1 < len(package_sections):
                package_name = package_sections[i]
                package_content = package_sections[i + 1]

                # Extract the suggested version
                suggested_version_match = re.search(
                    r"\*\*Suggested Version\*\*: `(.*?)`", package_content,
                )
                if suggested_version_match:
                    suggested_version = suggested_version_match.group(1)

                    # Extract affected platforms
                    affected_platforms_match = re.search(
                        r"\*\*Affected Platforms\*\*: (.*?)\s*$",
                        package_content,
                        re.MULTILINE,
                    )
                    affected_platforms = []
                    if affected_platforms_match:
                        affected_platforms = [
                            p.strip()
                            for p in affected_platforms_match.group(1).split(",")
                        ]

                    recommendations.append(
                        {
                            "package": package_name,
                            "suggested_version": suggested_version,
                            "affected_platforms": affected_platforms,
                        },
                    )

        return recommendations

    def apply_fixes(self) -> Dict[str, List[dict[str, Any]]]:
        """
        Apply recommendations to fix compatibility issues.

        Returns:
            Dictionary with summary of applied changes
        """
        # Get the hardware requirements data
        hw_requirements = self.compatibility_data.get("hardware_requirements", {})

        # Determine which recommendations to apply
        recommendations = self.resolution_plan.get("high_priority", [])
        if not self.high_priority_only:
            recommendations.extend(self.resolution_plan.get("medium_priority", []))

        # Apply each recommendation
        for recommendation in recommendations:
            package = recommendation["package"]
            suggested_version = recommendation["suggested_version"]
            affected_platforms = recommendation["affected_platforms"]

            for hw_name in affected_platforms:
                for file_name, file_data in hw_requirements.get(hw_name, {}).items():
                    if package in file_data:
                        # Get the original requirements file path
                        original_file = self._find_original_file_path(
                            hw_name, file_name,
                        )
                        if not original_file:
                            self.skipped_changes[hw_name].append(
                                {
                                    "package": package,
                                    "file": file_name,
                                    "reason": "Original file not found",
                                },
                            )
                            continue

                        # Find the line with the package declaration
                        try:
                            self._update_package_version(
                                original_file, package, suggested_version,
                            )

                            # Record the change
                            self.applied_changes[hw_name].append(
                                {
                                    "package": package,
                                    "file": original_file,
                                    "original_version": file_data[package],
                                    "new_version": suggested_version,
                                },
                            )
                        except Exception as e:
                            self.skipped_changes[hw_name].append(
                                {
                                    "package": package,
                                    "file": original_file,
                                    "reason": f"Failed to update: {str(e)}",
                                },
                            )

        return {
            "applied_changes": self.applied_changes,
            "skipped_changes": self.skipped_changes,
        }

    def _find_original_file_path(self, hw_name: str, file_name: str) -> str:
        """
        Find the original path to a requirements file.

        Args:
            hw_name: Hardware platform name
            file_name: Requirements file name

        Returns:
            Full path to the original requirements file, or empty string if not found
        """
        for file_path in (
            self.compatibility_data.get("hardware_requirements", {})
            .get(hw_name, {})

        ):
            if os.path.basename(file_path) == file_name:
                return file_path

        return ""

    def _update_package_version(
        self, file_path: str, package: str, version: str,
    ) -> bool:
        """
        Update package version in a requirements file.

        Args:
            file_path: Path to requirements file
            package: Package name to update
            version: New version to set

        Returns:
            Boolean indicating success
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Requirements file not found: {file_path}")

        # Read the file content
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Create a backup if not in dry run mode
        if not self.dry_run:
            backup_file = f"{file_path}{self.backup_suffix}"
            shutil.copy2(file_path, backup_file)
            print(f"Created backup: {backup_file}")

        # Initialize variables for finding the package
        package_found = False
        updated_lines = []

        # Regular expression to match the package name at the start of the line
        # followed by any version specifier
        package_re = re.compile(rf"^{re.escape(package)}\s*(?:[<>=~!]=?.*)?$")

        # Process each line
        for line in lines:
            # Skip comments and empty lines
            if line.strip().startswith("#") or not line.strip():
                updated_lines.append(line)
                continue

            # Check if this line declares the package
            if package_re.match(line.strip()):
                package_found = True

                # Format the new line with the updated version
                new_line = f"{package}\n" if version == "latest" else f"{package}{version}\n"

                updated_lines.append(new_line)
                print(f"Updating {file_path}: {line.strip()} -> {new_line.strip()}")
            else:
                updated_lines.append(line)

        # If the package wasn't found but should be added
        if not package_found:
            raise ValueError(f"Package {package} not found in {file_path}")

        # Write the updated content if not in dry run mode
        if not self.dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)

        return True

    def generate_report(self) -> str:
        """
        Generate a report of applied and skipped changes.

        Returns:
            Path to the generated report
        """
        report_path = os.path.join(self.output_dir, "autofix_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Hardware Compatibility Auto-fix Report\n\n")

            f.write("## Overview\n\n")

            total_applied = sum(
                len(changes) for changes in self.applied_changes.values()
            )
            total_skipped = sum(
                len(changes) for changes in self.skipped_changes.values()
            )

            f.write(
                f"- **Mode**: {'High Priority Only' if self.high_priority_only else 'All Recommendations'}\n",
            )
            f.write(
                f"- **Dry Run**: {'Yes (no changes applied)' if self.dry_run else 'No (changes applied)'}\n",
            )
            f.write(f"- **Total Changes Applied**: {total_applied}\n")
            f.write(f"- **Total Changes Skipped**: {total_skipped}\n\n")

            # Applied changes section
            f.write("## Applied Changes\n\n")

            if total_applied == 0:
                f.write("No changes were applied.\n\n")
            else:
                # List all changes by platform
                for hw_name, changes in sorted(self.applied_changes.items()):
                    f.write(f"### {hw_name}\n\n")

                    f.write("| Package | File | Original Version | New Version |\n")
                    f.write("|---------|------|------------------|-------------|\n")

                    for change in changes:
                        package = change["package"]
                        file_name = os.path.basename(change["file"])
                        original_version = change["original_version"]
                        new_version = change["new_version"]

                        f.write(
                            f"| {package} | {file_name} | `{original_version}` | `{new_version}` |\n",
                        )

                    f.write("\n")

            # Skipped changes section
            f.write("## Skipped Changes\n\n")

            if total_skipped == 0:
                f.write("No changes were skipped.\n\n")
            else:
                # List all skipped changes by platform
                for hw_name, changes in sorted(self.skipped_changes.items()):
                    f.write(f"### {hw_name}\n\n")

                    f.write("| Package | File | Reason |\n")
                    f.write("|---------|------|--------|\n")

                    for change in changes:
                        package = change["package"]
                        file_name = (
                            os.path.basename(change["file"])
                            if isinstance(change["file"], str)
                            else change["file"]
                        )
                        reason = change["reason"]

                        f.write(f"| {package} | {file_name} | {reason} |\n")

                    f.write("\n")

            # Additional information
            f.write("## Next Steps\n\n")

            if self.dry_run:
                f.write(
                    "This was a dry run, so no changes were actually applied. To apply the changes, run with `--no-dry-run`.\n\n",
                )
            else:
                f.write(
                    "1. Review the applied changes to ensure they meet your requirements.\n",
                )
                f.write(
                    "2. If needed, restore from backups (files with `.bak` extension).\n",
                )
                f.write(
                    "3. Run the hardware compatibility tests again to verify that conflicts have been resolved.\n\n",
                )

            f.write("## Backup Information\n\n")

            if self.dry_run:
                f.write("No backups were created because this was a dry run.\n")
            else:
                f.write(
                    "Backups were created with the suffix `{self.backup_suffix}` for each modified file.\n",
                )
                f.write("To restore from backup, use:\n\n")
                f.write("```bash\n")
                f.write("# Replace with the specific file path\n")
                f.write(f"mv /path/to/file{self.backup_suffix} /path/to/file\n")
                f.write("```\n")

        return report_path

    def write_github_summary(self) -> str:
        """
        Write a summary for GitHub Actions step summary.

        Returns:
            Path to the generated summary
        """
        summary_path = os.path.join(self.output_dir, "github_summary.md")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("## Hardware Compatibility Auto-fix Summary\n\n")

            total_applied = sum(
                len(changes) for changes in self.applied_changes.values()
            )
            total_skipped = sum(
                len(changes) for changes in self.skipped_changes.values()
            )

            f.write(
                f"**Mode**: {'🔴 High Priority Only' if self.high_priority_only else '🟠 All Recommendations'} | ",
            )
            f.write(
                f"**Run Type**: {'🔍 Dry Run (no changes)' if self.dry_run else '🛠️ Live Run (applied changes)'}\n\n",
            )

            # Summary counts
            f.write("| Result | Count | Details |\n")
            f.write("|--------|-------|--------|\n")
            f.write(
                f"| ✅ Applied | {total_applied} | Changes successfully applied |\n",
            )
            f.write(
                f"| ⚠️ Skipped | {total_skipped} | Changes that could not be applied |\n\n",
            )

            # Show sample of applied changes
            if total_applied > 0:
                f.write("### Sample Applied Changes\n\n")

                # Get a few sample changes
                samples = []
                for hw_name, changes in self.applied_changes.items():
                    for change in changes[:2]:  # Take up to 2 from each platform
                        samples.append(
                            {
                                "hardware": hw_name,
                                "package": change["package"],
                                "from": change["original_version"],
                                "to": change["new_version"],
                            },
                        )
                    if len(samples) >= 5:  # Show at most 5 samples
                        break

                for sample in samples[:5]:
                    f.write(
                        f"- {sample['hardware']}: `{sample['package']}` from `{sample['from']}` to `{sample['to']}`\n",
                    )

                if total_applied > 5:
                    f.write(f"\n... and {total_applied - 5} more changes\n")

            # Reference to full report
            f.write(
                "\n[See detailed report](autofix_report.md) for complete information about applied and skipped changes.\n",
            )

        return summary_path

    def run(self) -> int:
        """
        Run the hardware compatibility auto-fix tool.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        print("Starting Hardware Compatibility Auto-fix...")

        # Load compatibility data
        print("Loading compatibility data...")
        if not self.load_compatibility_data():
            return 1

        # Load recommendations
        print("Loading recommendations...")
        if not self.load_recommendations():
            return 1

        # Apply fixes
        print("Applying fixes...")
        self.apply_fixes()

        # Generate report
        print("Generating report...")
        report_path = self.generate_report()

        # Write GitHub summary
        print("Writing GitHub summary...")
        self.write_github_summary()

        print("Hardware Compatibility Auto-fix complete!")
        print(f"Report: {report_path}")

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fix hardware compatibility issues",
    )
    parser.add_argument(
        "--input-dir",
        default="ci_artifacts/hardware_compatibility",
        help="Directory containing hardware compatibility test results",
    )
    parser.add_argument(
        "--recommendations-dir",
        default="ci_artifacts/hardware_compatibility/recommendations",
        help="Directory containing advisor recommendations",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/hardware_compatibility/autofix",
        help="Directory to store auto-fix results",
    )
    parser.add_argument(
        "--resolution-plan",
        default="resolution_plan.md",
        help="Markdown file with resolution plan",
    )
    parser.add_argument(
        "--data-file",
        default="hardware_compatibility_data.json",
        help="JSON file with compatibility test data",
    )
    parser.add_argument(
        "--backup-suffix", default=".bak", help="Suffix for backup files",
    )
    parser.add_argument(
        "--high-priority-only",
        action="store_true",
        default=True,
        help="Only apply high priority fixes",
    )
    parser.add_argument(
        "--all-priorities",
        action="store_true",
        help="Apply both high and medium priority fixes",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate changes without applying them",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Generate GitHub Actions compatible summary",
    )

    args = parser.parse_args()

    # If all priorities is specified, override high-priority-only
    high_priority_only = (
        not args.all_priorities if args.all_priorities else args.high_priority_only
    )

    autofix = HardwareCompatibilityAutofix(
        input_dir=args.input_dir,
        recommendations_dir=args.recommendations_dir,
        output_dir=args.output_dir,
        resolution_plan=args.resolution_plan,
        data_file=args.data_file,
        backup_suffix=args.backup_suffix,
        high_priority_only=high_priority_only,
        dry_run=args.dry_run,
    )

    exit_code = autofix.run()

    if args.github_summary:
        # Get the GITHUB_STEP_SUMMARY environment variable
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            # Copy the GitHub summary to the step summary file
            summary_path = os.path.join(args.output_dir, "github_summary.md")
            if os.path.exists(summary_path):
                with open(summary_path, encoding="utf-8") as src:
                    with open(step_summary, "a", encoding="utf-8") as dest:
                        dest.write(src.read())
                print("Added summary to GitHub Actions output")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()