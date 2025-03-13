#!/usr/bin/env python
"""
Hardware Compatibility Advisor

This script analyzes hardware compatibility testing results and provides
specific recommendations for resolving version conflicts across hardware platforms.
It generates actionable suggestions to standardize package versions and improve
cross-platform compatibility.

Features:
- Automated version conflict resolution suggestions
- Optimized version selection based on compatibility analysis
- Generation of standardized requirements files
- Integration with existing hardware compatibility test results
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any


class HardwareCompatibilityAdvisor:
    def __init__(
        self,
        input_dir: str = "ci_artifacts/hardware_compatibility",
        output_dir: str = "ci_artifacts/hardware_compatibility/recommendations",
        data_file: str = "hardware_compatibility_data.json",
    ):
        """
        Initialize the hardware compatibility advisor.

        Args:
            input_dir: Directory containing hardware compatibility test results
            output_dir: Directory to store advisor recommendations
            data_file: JSON file with compatibility test data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data_file = os.path.join(input_dir, data_file)
        self.compatibility_data = {}
        self.recommendations = {}
        self.resolution_plans = {}
        self.optimized_requirements = {}

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_compatibility_data(self) -> bool:
        """
        Load compatibility test results from JSON file.

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

    def analyze_conflicts(self) -> dict[str, Any]:
        """
        Analyze conflicts and categorize them by severity and type.

        Returns:
            Dictionary with conflict analysis results
        """
        if not self.compatibility_data or "conflicts" not in self.compatibility_data:
            return {}

        conflicts = self.compatibility_data.get("conflicts", [])

        # Group conflicts by package
        conflicts_by_package = {}
        for conflict in conflicts:
            package = conflict["package"]
            conflicts_by_package[package] = conflict

        # Categorize conflicts
        pinned_version_conflicts = []
        constraint_conflicts = []
        mixed_conflicts = []

        for package, conflict in conflicts_by_package.items():
            versions = conflict["all_versions"]

            # Check if all versions are pinned
            all_pinned = all("==" in v for v in versions)
            # Check if all versions are constraints
            all_constraints = all(
                any(op in v for op in [">=", "<=", ">", "<", "~="]) for v in versions
            )

            if all_pinned:
                pinned_version_conflicts.append(conflict)
            elif all_constraints:
                constraint_conflicts.append(conflict)
            else:
                mixed_conflicts.append(conflict)

        return {
            "total_conflicts": len(conflicts),
            "pinned_version_conflicts": pinned_version_conflicts,
            "constraint_conflicts": constraint_conflicts,
            "mixed_conflicts": mixed_conflicts,
        }

    def generate_version_suggestions(self, conflict: dict[str, Any]) -> dict[str, Any]:
        """
        Generate version standardization suggestions for a single conflict.

        Args:
            conflict: Conflict data dictionary

        Returns:
            Dictionary with version suggestions
        """
        package = conflict["package"]
        versions = conflict["all_versions"]
        all_versions = conflict["all_versions"]

        # Extract version numbers from pinned versions
        pinned_versions = []
        for version in all_versions:
            if "==" in version:
                version_num = version.split("==")[1].strip()
                pinned_versions.append(version_num)

        # For version constraints, we need special handling
        constraints = []
        for version in all_versions:
            if any(op in version for op in [">=", "<=", ">", "<", "~="]):
                constraints.append(version)

        # Count hardware platforms using each version
        version_usage = defaultdict(list)
        for hw, hw_versions in versions.items():
            for file, version in hw_versions.items():
                version_usage[version].append((hw, file))

        # Determine most common version
        most_common_version = None
        most_common_count = 0
        for version, usages in version_usage.items():
            if len(usages) > most_common_count:
                most_common_version = version
                most_common_count = len(usages)

        # Find the most common version
        if len(versions) == 1:
            # Only one version, use it
            suggested_version = list(versions.keys())[0]
            rationale = "Only one version found across platforms"
        else:
            # Multiple versions, find the most common
            most_common_version = max(versions.items(), key=lambda x: x[1])
            suggested_version = most_common_version[0]
            rationale = "Most common version across platforms"

            # Check if there's a newer version that's also common
            for version_str, count in versions.items():
                if count == most_common_version[1] and version_str > suggested_version:
                    suggested_version = version_str
                    rationale = "Newest among most common versions"

            # Special case for version constraints
            if ">" in suggested_version or "<" in suggested_version:
                # For constraints, prefer the most permissive
                for version_str in versions:
                    if "==" in version_str:
                        suggested_version = version_str
                        rationale = "Exact version preferred over constraints"
                        break

        # Calculate impact and confidence
        if max(versions.values()) == len(versions.keys()):
            # All platforms use the same version
            impact = "Low"
            confidence = "High"
            rationale = "All platforms already use this version"
        elif max(versions.values()) >= len(versions.keys()) / 2:
            # Majority of platforms use this version
            impact = "Medium"
            confidence = "High"
        else:
            # No clear majority
            impact = "High"
            confidence = "Medium"

        return {
            "package": package,
            "type": "mixed",
            "current_versions": all_versions,
            "suggested_version": suggested_version,
            "rationale": rationale,
            "affected_platforms": list(versions.keys()),
            "impact": impact,
            "confidence": confidence,
        }

    def generate_all_recommendations(self) -> dict[str, list[dict[str, Any]]]:
        """
        Generate recommendations for all conflicts.

        Returns:
            Dictionary of recommendations by category
        """
        if not self.compatibility_data:
            return {}

        conflicts = self.compatibility_data.get("conflicts", [])

        # Generate suggestions for each conflict
        high_priority_recommendations = []
        medium_priority_recommendations = []
        low_priority_recommendations = []

        for conflict in conflicts:
            suggestion = self.generate_version_suggestions(conflict)

            # Categorize by severity
            if conflict["severity"] == "high":
                high_priority_recommendations.append(suggestion)
            elif conflict["severity"] == "medium":
                medium_priority_recommendations.append(suggestion)
            else:
                low_priority_recommendations.append(suggestion)

        self.recommendations = {
            "high_priority": high_priority_recommendations,
            "medium_priority": medium_priority_recommendations,
            "low_priority": low_priority_recommendations,
        }

        return self.recommendations

    def generate_resolution_plan(self) -> dict[str, dict[str, str]]:
        """
        Generate a concrete resolution plan for each hardware platform.

        Returns:
            Dictionary with resolution plans by platform
        """
        if not self.compatibility_data or not self.recommendations:
            return {}

        # Get hardware requirements data
        hw_requirements = self.compatibility_data.get("hardware_requirements", {})

        # Start with current requirements for each platform
        resolution_plans = {}
        for hw_name, hw_data in hw_requirements.items():
            resolution_plans[hw_name] = {}
            for file_name, file_data in hw_data.items():
                resolution_plans[hw_name][file_name] = file_data.copy()

        # Apply high priority recommendations
        for recommendation in self.recommendations.get("high_priority", []):
            package = recommendation["package"]
            suggested_version = recommendation["suggested_version"]

            for hw_name in recommendation["affected_platforms"]:
                for file_name in hw_requirements.get(hw_name, {}):
                    if (
                        hw_name in resolution_plans
                        and file_name in resolution_plans[hw_name]
                    ) and package in resolution_plans[hw_name][file_name]:
                        # Update the version
                        resolution_plans[hw_name][file_name][
                            package
                        ] = suggested_version

        # Apply medium priority recommendations only if they don't conflict with high priority
        for recommendation in self.recommendations.get("medium_priority", []):
            package = recommendation["package"]
            suggested_version = recommendation["suggested_version"]

            # Check if this package was already handled by high priority
            high_priority_packages = [
                r["package"] for r in self.recommendations.get("high_priority", [])
            ]
            if package in high_priority_packages:
                continue

            for hw_name in recommendation["affected_platforms"]:
                for file_name in hw_requirements.get(hw_name, {}):
                    if (
                        hw_name in resolution_plans
                        and file_name in resolution_plans[hw_name]
                    ) and package in resolution_plans[hw_name][file_name]:
                        # Update the version
                        resolution_plans[hw_name][file_name][
                            package
                        ] = suggested_version

        self.resolution_plans = resolution_plans
        return resolution_plans

    def generate_optimized_requirements(self) -> dict[str, dict[str, str]]:
        """
        Generate optimized requirements files based on resolution plan.

        Returns:
            Dictionary with optimized requirements by platform
        """
        if not self.resolution_plans:
            return {}

        optimized_requirements = {}

        for hw_name, hw_files in self.resolution_plans.items():
            optimized_requirements[hw_name] = {}

            for file_name, packages in hw_files.items():
                # Sort packages alphabetically
                sorted_packages = dict(sorted(packages.items()))
                optimized_requirements[hw_name][file_name] = sorted_packages

        self.optimized_requirements = optimized_requirements
        return optimized_requirements

    def write_resolution_plan(self) -> str:
        """
        Write resolution plan to a markdown file.

        Returns:
            Path to the generated report
        """
        if not self.resolution_plans:
            return ""

        report_path = os.path.join(self.output_dir, "resolution_plan.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Hardware Compatibility Resolution Plan\n\n")

            f.write("## Overview\n\n")
            f.write(
                "This plan provides specific recommendations for resolving package compatibility issues across hardware platforms.\n\n"
            )

            # Write summary of recommendations
            f.write("## Recommendations Summary\n\n")

            high_priority_count = len(self.recommendations.get("high_priority", []))
            medium_priority_count = len(self.recommendations.get("medium_priority", []))
            low_priority_count = len(self.recommendations.get("low_priority", []))

            f.write(f"- **High Priority Recommendations**: {high_priority_count}\n")
            f.write(f"- **Medium Priority Recommendations**: {medium_priority_count}\n")
            f.write(f"- **Low Priority Recommendations**: {low_priority_count}\n\n")

            # Write high priority recommendations
            f.write("## High Priority Recommendations\n\n")

            if high_priority_count == 0:
                f.write("No high priority recommendations.\n\n")
            else:
                for recommendation in self.recommendations.get("high_priority", []):
                    package = recommendation["package"]
                    current_versions = ", ".join(
                        f"`{v}`" for v in recommendation["current_versions"]
                    )
                    suggested_version = recommendation["suggested_version"]
                    rationale = recommendation["rationale"]
                    affected_platforms = ", ".join(recommendation["affected_platforms"])

                    f.write(f"### {package}\n\n")
                    f.write(f"- **Current Versions**: {current_versions}\n")
                    f.write(f"- **Suggested Version**: `{suggested_version}`\n")
                    f.write(f"- **Rationale**: {rationale}\n")
                    f.write(f"- **Affected Platforms**: {affected_platforms}\n\n")

            # Write medium priority recommendations
            f.write("## Medium Priority Recommendations\n\n")

            if medium_priority_count == 0:
                f.write("No medium priority recommendations.\n\n")
            else:
                for recommendation in self.recommendations.get("medium_priority", []):
                    package = recommendation["package"]
                    current_versions = ", ".join(
                        f"`{v}`" for v in recommendation["current_versions"]
                    )
                    suggested_version = recommendation["suggested_version"]
                    rationale = recommendation["rationale"]
                    affected_platforms = ", ".join(recommendation["affected_platforms"])

                    f.write(f"### {package}\n\n")
                    f.write(f"- **Current Versions**: {current_versions}\n")
                    f.write(f"- **Suggested Version**: `{suggested_version}`\n")
                    f.write(f"- **Rationale**: {rationale}\n")
                    f.write(f"- **Affected Platforms**: {affected_platforms}\n\n")

            # Write standardized requirements section
            f.write("## Standardized Requirements\n\n")
            f.write(
                "The following sections provide standardized requirements files for each platform with resolved conflicts.\n\n"
            )

            for hw_name, hw_files in self.optimized_requirements.items():
                f.write(f"### {hw_name}\n\n")

                for file_name, packages in hw_files.items():
                    f.write(f"#### {file_name}\n\n")
                    f.write("```\n")

                    for package, version in packages.items():
                        if version == "latest":
                            f.write(f"{package}\n")
                        else:
                            f.write(f"{package}{version}\n")

                    f.write("```\n\n")

        return report_path

    def generate_patch_files(self) -> list[str]:
        """
        Generate patch files that can be applied to resolve conflicts.

        Returns:
            List of generated patch file paths
        """
        if not self.optimized_requirements:
            return []

        patch_files = []

        for hw_name, hw_files in self.optimized_requirements.items():
            for file_name, packages in hw_files.items():
                # Get the original requirements file path from compatibility data
                original_files = []
                for file_path in (
                    self.compatibility_data.get("hardware_requirements", {})
                    .get(hw_name, {})

                ):
                    if os.path.basename(file_path) == file_name:
                        original_files.append(file_path)

                if not original_files:
                    continue

                original_file = original_files[0]  # Take the first matching file

                # Create the patch file
                patch_file = os.path.join(
                    self.output_dir, f"{hw_name}_{file_name}.patch"
                )

                with open(patch_file, "w", encoding="utf-8") as f:
                    f.write(f"--- {original_file}\n")
                    f.write(f"+++ {original_file}\n")

                    # Write the optimized requirements
                    for package, version in packages.items():
                        if version == "latest":
                            f.write(f"+{package}\n")
                        else:
                            f.write(f"+{package}{version}\n")

                patch_files.append(patch_file)

        return patch_files

    def write_github_summary(self) -> str:
        """
        Write a summary for GitHub Actions step summary.

        Returns:
            Path to the generated summary
        """
        summary_path = os.path.join(self.output_dir, "github_summary.md")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("## Hardware Compatibility Advisor Summary\n\n")

            # Count of recommendations
            high_priority_count = len(self.recommendations.get("high_priority", []))
            medium_priority_count = len(self.recommendations.get("medium_priority", []))
            low_priority_count = len(self.recommendations.get("low_priority", []))

            total_recommendations = (
                high_priority_count + medium_priority_count + low_priority_count
            )

            f.write(
                f"Generated **{total_recommendations}** recommendations for resolving hardware compatibility issues:\n\n"
            )

            f.write("| Priority | Count | Action Required |\n")
            f.write("|----------|-------|----------------|\n")
            f.write(
                f"| 🔴 High | {high_priority_count} | Immediate attention required |\n"
            )
            f.write(
                f"| 🟡 Medium | {medium_priority_count} | Review in next development cycle |\n"
            )
            f.write(f"| 🟢 Low | {low_priority_count} | Consider when convenient |\n\n")

            # Show top recommendations if any
            if high_priority_count > 0:
                f.write("### Top Recommendations\n\n")

                for recommendation in self.recommendations.get("high_priority", [])[
                    :3
                ]:  # Show up to 3
                    package = recommendation["package"]
                    suggested_version = recommendation["suggested_version"]
                    affected_platforms = ", ".join(recommendation["affected_platforms"])

                    f.write(
                        f"- **{package}**: Standardize on `{suggested_version}` across {affected_platforms}\n"
                    )

                if high_priority_count > 3:
                    f.write(
                        f"\n... and {high_priority_count - 3} more high priority recommendations\n"
                    )

                f.write("\n")

            # Reference to full report
            f.write(
                "See [detailed resolution plan](resolution_plan.md) for complete recommendations.\n"
            )

        # Add summary to GitHub step summary if running in GitHub Actions
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ.get("GITHUB_STEP_SUMMARY"), "a") as f:
                f.write("## Hardware Compatibility Advisor Results\n\n")

                # Add overview
                f.write("### Overview\n\n")
                f.write(
                    "The Hardware Compatibility Advisor analyzed compatibility data and generated recommendations.\n\n"
                )

                # Add statistics
                f.write("### Statistics\n\n")
                f.write(
                    "- **Total Conflicts:** {}\n".format(
                        len(self.compatibility_data.get("conflicts", []))
                    )
                )
                f.write(
                    "- **High Priority Recommendations:** {}\n".format(
                        len(self.recommendations.get("high_priority", []))
                    )
                )
                f.write(
                    "- **Medium Priority Recommendations:** {}\n".format(
                        len(self.recommendations.get("medium_priority", []))
                    )
                )
                f.write(
                    "- **Low Priority Recommendations:** {}\n".format(
                        len(self.recommendations.get("low_priority", []))
                    )
                )

                # Add recommendation summary
                f.write("\n### Recommendation Summary\n\n")
                if self.recommendations.get("high_priority", []):
                    f.write("#### High Priority\n\n")
                    for rec in self.recommendations["high_priority"][:5]:  # Show top 5
                        f.write(
                            "- `{}`: Standardize to `{}`\n".format(
                                rec["package"], rec["suggested_version"]
                            )
                        )
                    if len(self.recommendations["high_priority"]) > 5:
                        f.write(
                            "- ... and {} more\n".format(
                                len(self.recommendations["high_priority"]) - 5
                            )
                        )
                    f.write("\n")

                # Add next steps
                f.write("\n### Next Steps\n\n")
                f.write(
                    "1. Review the [Resolution Plan]({})\n".format(
                        os.path.join(self.output_dir, "resolution_plan.md")
                    )
                )
                f.write(
                    "2. Apply recommended changes to standardize package versions\n"
                )
                f.write(
                    "3. Re-run the hardware compatibility tests to verify improvements\n"
                )

            print("Added summary to GitHub Actions output")

        return "GitHub step summary not available"

    def run(self) -> int:
        """
        Run the hardware compatibility advisor.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        print("Starting Hardware Compatibility Advisor...")

        # Load compatibility test results
        print("Loading compatibility data...")
        if not self.load_compatibility_data():
            return 1

        # Analyze conflicts
        print("Analyzing conflicts...")
        self.analyze_conflicts()

        # Generate recommendations
        print("Generating recommendations...")
        self.generate_all_recommendations()

        # Generate resolution plan
        print("Generating resolution plan...")
        self.generate_resolution_plan()

        # Generate optimized requirements
        print("Generating optimized requirements...")
        self.generate_optimized_requirements()

        # Write resolution plan
        print("Writing resolution plan...")
        report_path = self.write_resolution_plan()

        # Generate patch files
        print("Generating patch files...")
        self.generate_patch_files()

        # Write GitHub summary
        print("Writing GitHub summary...")
        self.write_github_summary()

        print("Hardware Compatibility Advisor complete!")
        print(f"Resolution plan: {report_path}")

        # Return success
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hardware compatibility and provide recommendations"
    )
    parser.add_argument(
        "--input-dir",
        default="ci_artifacts/hardware_compatibility",
        help="Directory containing hardware compatibility test results",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/hardware_compatibility/recommendations",
        help="Directory to store advisor recommendations",
    )
    parser.add_argument(
        "--data-file",
        default="hardware_compatibility_data.json",
        help="JSON file with compatibility test data",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Generate GitHub Actions compatible summary",
    )

    args = parser.parse_args()

    advisor = HardwareCompatibilityAdvisor(
        input_dir=args.input_dir, output_dir=args.output_dir, data_file=args.data_file
    )

    exit_code = advisor.run()

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

    # Print final message
    if exit_code == 0:
        print("Hardware Compatibility Advisor completed successfully.")
    else:
        print("Hardware Compatibility Advisor completed with errors.")

    return exit_code


if __name__ == "__main__":
    main()
