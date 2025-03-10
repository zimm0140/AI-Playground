#!/usr/bin/env python
"""
Hardware Compatibility Tester

This script analyzes hardware-specific requirements files and validates compatibility
across different Intel architectures. It detects conflicts, incompatible versions,
and provides detailed reports to ensure correct operation on all supported hardware.

Features:
- Cross-validation of package versions between hardware platforms
- Detection of compatibility issues and version conflicts
- Generation of compatibility matrices for visualization
- Performance impact analysis of package versions
- Hardware-specific test environment setup recommendations
"""

import os
import re
import sys
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any
import itertools
import platform


class HardwareCompatibilityTester:
    def __init__(
        self,
        output_dir: str = "ci_artifacts/hardware_compatibility",
        req_files_pattern: str = "**/*requirements*.txt",
    ):
        """
        Initialize the hardware compatibility tester.

        Args:
            output_dir: Directory to store test results and reports
            req_files_pattern: Glob pattern to identify requirements files
        """
        self.output_dir = output_dir
        self.req_files_pattern = req_files_pattern
        self.hardware_patterns = {
            "mtl": r".*mtl.*",  # Meteor Lake
            "lnl": r".*lnl.*",  # Lunar Lake
            "bmg": r".*bmg.*",  # Battlemage
            "arl_h": r".*arl_h.*",  # Arc Alchemist
            "acm": r".*acm.*",  # Advanced Compute Models
            "level_zero": r".*level_zero.*",  # Level Zero interface
            "llamacpp": r".*/LlamaCPP/.*",  # LlamaCPP requirements
        }
        self.hardware_requirements = {}
        self.hardware_compatibility_matrix = {}
        self.conflict_data = []

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def find_hardware_requirements(self) -> Dict[str, List[str]]:
        """Find hardware-specific requirements files."""
        hw_req_files = defaultdict(list)

        for req_file in glob.glob(self.req_files_pattern, recursive=True):
            for hw_name, pattern in self.hardware_patterns.items():
                if re.match(pattern, req_file, re.IGNORECASE):
                    hw_req_files[hw_name].append(req_file)

        # Add default requirements
        default_req = [
            f
            for f in glob.glob("**/requirements.txt", recursive=True)
            if not any(
                re.match(pattern, f, re.IGNORECASE)
                for pattern in self.hardware_patterns.values()
            )
        ]

        if default_req:
            hw_req_files["default"] = default_req

        return dict(hw_req_files)

    def parse_requirements_file(self, file_path: str) -> Dict[str, str]:
        """
        Parse a requirements file and extract package names and versions.

        Args:
            file_path: Path to the requirements file

        Returns:
            Dictionary of package names to version specifications
        """
        requirements = {}

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return requirements

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#") or line.startswith("--"):
                    continue

                # Extract package name and version
                if "==" in line:
                    parts = line.split("==", 1)
                    package = parts[0].strip()
                    version = "==" + parts[1].strip()
                elif ">=" in line:
                    parts = line.split(">=", 1)
                    package = parts[0].strip()
                    version = ">=" + parts[1].strip()
                elif "<=" in line:
                    parts = line.split("<=", 1)
                    package = parts[0].strip()
                    version = "<=" + parts[1].strip()
                elif ">" in line:
                    parts = line.split(">", 1)
                    package = parts[0].strip()
                    version = ">" + parts[1].strip()
                elif "<" in line:
                    parts = line.split("<", 1)
                    package = parts[0].strip()
                    version = "<" + parts[1].strip()
                elif "~=" in line:
                    parts = line.split("~=", 1)
                    package = parts[0].strip()
                    version = "~=" + parts[1].strip()
                else:
                    # No version specification
                    package = line
                    version = "latest"

                requirements[package] = version

        return requirements

    def analyze_hardware_requirements(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Analyze hardware-specific requirements and extract package versions.

        Returns:
            Nested dictionary of hardware platforms, files, and package requirements
        """
        hw_req_files = self.find_hardware_requirements()
        hardware_requirements = {}

        for hw_name, files in hw_req_files.items():
            hardware_requirements[hw_name] = {}

            for file_path in files:
                file_name = os.path.basename(file_path)
                requirements = self.parse_requirements_file(file_path)
                hardware_requirements[hw_name][file_name] = requirements

        self.hardware_requirements = hardware_requirements
        return hardware_requirements

    def identify_conflicts(self) -> List[Dict[str, Any]]:
        """
        Identify conflicts between hardware-specific requirements.

        Returns:
            List of conflict data with details
        """
        conflicts = []

        # Get all packages across all hardware platforms
        all_packages = set()
        for hw_data in self.hardware_requirements.values():
            for file_data in hw_data.values():
                all_packages.update(file_data.keys())

        # Check for conflicts in each package
        for package in all_packages:
            package_versions = {}

            # Collect versions for this package across hardware platforms
            for hw_name, hw_data in self.hardware_requirements.items():
                for file_name, file_data in hw_data.items():
                    if package in file_data:
                        if hw_name not in package_versions:
                            package_versions[hw_name] = {}
                        package_versions[hw_name][file_name] = file_data[package]

            # Check if there are different versions across platforms
            all_versions = set()
            for hw_versions in package_versions.values():
                all_versions.update(hw_versions.values())

            if len(all_versions) > 1:
                # We have a potential conflict
                conflicts.append(
                    {
                        "package": package,
                        "versions": package_versions,
                        "all_versions": list(all_versions),
                        "severity": "high"
                        if "==" in "".join(all_versions)
                        else "medium",
                    }
                )

        self.conflict_data = conflicts
        return conflicts

    def generate_compatibility_matrix(self) -> Dict[str, Dict[str, str]]:
        """
        Generate a compatibility matrix between hardware platforms.

        Returns:
            Dictionary representing the compatibility matrix
        """
        hw_platforms = list(self.hardware_requirements.keys())
        compatibility_matrix = {}

        for hw1, hw2 in itertools.combinations(hw_platforms, 2):
            # Compare packages between two hardware platforms
            conflicts = []

            # Get all packages for hw1
            hw1_packages = {}
            for file_data in self.hardware_requirements[hw1].values():
                hw1_packages.update(file_data)

            # Get all packages for hw2
            hw2_packages = {}
            for file_data in self.hardware_requirements[hw2].values():
                hw2_packages.update(file_data)

            # Find common packages
            common_packages = set(hw1_packages.keys()) & set(hw2_packages.keys())

            # Check for version conflicts
            for package in common_packages:
                if hw1_packages[package] != hw2_packages[package]:
                    conflicts.append(package)

            # Calculate compatibility score
            if not common_packages:
                score = "N/A"  # No common packages
            else:
                conflict_pct = len(conflicts) / len(common_packages) * 100

                if conflict_pct == 0:
                    score = "100% Compatible"
                elif conflict_pct < 10:
                    score = "High"
                elif conflict_pct < 30:
                    score = "Medium"
                else:
                    score = "Low"

            # Store in matrix
            if hw1 not in compatibility_matrix:
                compatibility_matrix[hw1] = {}
            compatibility_matrix[hw1][hw2] = {
                "score": score,
                "common_packages": len(common_packages),
                "conflicts": conflicts,
            }

            if hw2 not in compatibility_matrix:
                compatibility_matrix[hw2] = {}
            compatibility_matrix[hw2][hw1] = {
                "score": score,
                "common_packages": len(common_packages),
                "conflicts": conflicts,
            }

        self.hardware_compatibility_matrix = compatibility_matrix
        return compatibility_matrix

    def generate_markdown_report(self) -> str:
        """
        Generate a detailed markdown report of the analysis.

        Returns:
            Path to the generated report
        """
        report_path = os.path.join(self.output_dir, "hardware_compatibility_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Hardware Compatibility Report\n\n")

            f.write("## Overview\n\n")
            f.write(
                "This report analyzes package compatibility across different hardware platforms.\n\n"
            )

            f.write(f"- **Generated on**: {platform.node()}\n")
            f.write(f"- **System**: {platform.system()} {platform.release()}\n")
            f.write(f"- **Python**: {platform.python_version()}\n\n")

            f.write("## Hardware Platforms\n\n")
            f.write("| Platform | Requirements Files | Packages |\n")
            f.write("|----------|-------------------|----------|\n")

            for hw_name, hw_data in self.hardware_requirements.items():
                files = list(hw_data.keys())
                package_count = sum(len(data) for data in hw_data.values())
                f.write(f"| {hw_name} | {len(files)} | {package_count} |\n")

            f.write("\n## Compatibility Matrix\n\n")

            # Write compatibility matrix
            hw_platforms = sorted(self.hardware_compatibility_matrix.keys())

            # Header row
            f.write("| Platform |")
            for hw in hw_platforms:
                f.write(f" {hw} |")
            f.write("\n")

            # Separator row
            f.write("|----------|")
            for _ in hw_platforms:
                f.write("----------|")
            f.write("\n")

            # Data rows
            for hw1 in hw_platforms:
                f.write(f"| {hw1} |")
                for hw2 in hw_platforms:
                    if hw1 == hw2:
                        f.write(" — |")
                    else:
                        if hw2 in self.hardware_compatibility_matrix[hw1]:
                            f.write(
                                f" {self.hardware_compatibility_matrix[hw1][hw2]['score']} |"
                            )
                        else:
                            f.write(" N/A |")
                f.write("\n")

            # Write conflict details
            f.write("\n## Conflicts\n\n")

            if not self.conflict_data:
                f.write("No conflicts detected.\n")
            else:
                f.write(
                    f"Found {len(self.conflict_data)} packages with compatibility issues.\n\n"
                )

                # Sort conflicts by severity
                sorted_conflicts = sorted(
                    self.conflict_data,
                    key=lambda x: (0 if x["severity"] == "high" else 1, x["package"]),
                )

                for conflict in sorted_conflicts:
                    package = conflict["package"]
                    severity = conflict["severity"]

                    f.write(f"### {package} (Severity: {severity})\n\n")
                    f.write("| Hardware | File | Version |\n")
                    f.write("|----------|------|--------|\n")

                    # Sort by hardware platform for consistent output
                    for hw_name in sorted(conflict["versions"].keys()):
                        hw_versions = conflict["versions"][hw_name]

                        # For each file containing this package
                        for file_name, version in sorted(hw_versions.items()):
                            f.write(f"| {hw_name} | {file_name} | `{version}` |\n")

                    f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if not self.conflict_data:
                f.write(
                    "All packages are compatible across hardware platforms. No action required.\n"
                )
            else:
                f.write("### High Priority Fixes\n\n")

                high_priority = [
                    c for c in self.conflict_data if c["severity"] == "high"
                ]
                if high_priority:
                    for conflict in high_priority:
                        package = conflict["package"]
                        versions = ", ".join(
                            f"`{v}`" for v in sorted(conflict["all_versions"])
                        )
                        f.write(
                            f"- **{package}**: Standardize version across platforms. Current versions: {versions}\n"
                        )
                else:
                    f.write("No high priority fixes required.\n")

                f.write("\n### Medium Priority Fixes\n\n")

                medium_priority = [
                    c for c in self.conflict_data if c["severity"] == "medium"
                ]
                if medium_priority:
                    for conflict in medium_priority:
                        package = conflict["package"]
                        versions = ", ".join(
                            f"`{v}`" for v in sorted(conflict["all_versions"])
                        )
                        f.write(
                            f"- **{package}**: Consider standardizing version requirements. Current specs: {versions}\n"
                        )
                else:
                    f.write("No medium priority fixes required.\n")

        # Also generate JSON data
        json_path = os.path.join(self.output_dir, "hardware_compatibility_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hardware_requirements": self.hardware_requirements,
                    "compatibility_matrix": self.hardware_compatibility_matrix,
                    "conflicts": self.conflict_data,
                },
                f,
                indent=2,
            )

        return report_path

    def generate_github_summary(self) -> None:
        """Generate a summary for GitHub Actions step summary."""
        # Create a simplified version for GitHub Actions summary
        summary_path = os.path.join(self.output_dir, "github_summary.md")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("## Hardware Compatibility Summary\n\n")

            # Count of platforms
            platform_count = len(self.hardware_requirements)
            f.write(f"Analyzed {platform_count} hardware platforms:\n\n")

            platforms_list = ", ".join(
                f"`{hw}`" for hw in sorted(self.hardware_requirements.keys())
            )
            f.write(f"{platforms_list}\n\n")

            # Show conflict summary
            conflict_count = len(self.conflict_data)
            high_priority = len(
                [c for c in self.conflict_data if c["severity"] == "high"]
            )

            if conflict_count > 0:
                f.write(f"Found **{conflict_count}** package conflicts")
                if high_priority > 0:
                    f.write(f" ({high_priority} high priority)")
                f.write(".\n\n")

                if high_priority > 0:
                    f.write("### High Priority Conflicts\n\n")
                    for conflict in [
                        c for c in self.conflict_data if c["severity"] == "high"
                    ]:
                        package = conflict["package"]
                        platforms = ", ".join(sorted(conflict["versions"].keys()))
                        f.write(f"- **{package}**: Different versions on {platforms}\n")
                    f.write("\n")
            else:
                f.write("✅ **No compatibility issues detected!**\n\n")

            # Show compatibility overview
            f.write("### Compatibility Overview\n\n")
            total_pairs = sum(
                1
                for hw1 in self.hardware_compatibility_matrix
                for hw2 in self.hardware_compatibility_matrix[hw1]
                if hw1 != hw2
            )

            fully_compatible = sum(
                1
                for hw1 in self.hardware_compatibility_matrix
                for hw2 in self.hardware_compatibility_matrix[hw1]
                if hw1 != hw2
                and self.hardware_compatibility_matrix[hw1][hw2]["score"]
                == "100% Compatible"
            )

            compatibility_rate = (
                fully_compatible / total_pairs * 100 if total_pairs > 0 else 0
            )

            f.write(f"- **Overall compatibility rate**: {compatibility_rate:.1f}%\n")
            f.write(
                f"- **Fully compatible hardware pairs**: {fully_compatible} out of {total_pairs}\n"
            )

            # Reference to full report
            f.write(
                "\nSee [detailed report](hardware_compatibility_report.md) for more information.\n"
            )

    def run(self) -> int:
        """
        Run the hardware compatibility analysis.

        Returns:
            Exit code (0 for success, 1 for issues found)
        """
        try:
            print("Finding hardware-specific requirements files...")
            hw_req_files = self.find_hardware_requirements()

            # Print found hardware requirements files
            for hw, files in hw_req_files.items():
                print(f"Found {len(files)} requirements files for {hw}:")
                for file in files:
                    print(f"  - {file}")

            print("\nAnalyzing hardware requirements...")
            self.analyze_hardware_requirements()

            print("Identifying conflicts...")
            conflicts = self.identify_conflicts()

            print("Generating compatibility matrix...")
            self.generate_compatibility_matrix()

            print("Generating reports...")
            report_path = self.generate_markdown_report()
            self.generate_github_summary()

            # Summary output
            print("\nAnalysis complete!")
            print(f"- Analyzed {len(self.hardware_requirements)} hardware platforms")
            print(f"- Found {len(conflicts)} package conflicts")

            high_priority = len([c for c in conflicts if c["severity"] == "high"])
            if high_priority > 0:
                print(f"- {high_priority} high priority conflicts need attention")

            print(f"\nReport generated: {report_path}")

            # Return code based on conflicts
            return 1 if high_priority > 0 else 0

        except Exception as e:
            print(f"Error during hardware compatibility analysis: {e}")
            import traceback

            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hardware-specific package compatibility"
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/hardware_compatibility",
        help="Directory to store output reports",
    )
    parser.add_argument(
        "--req-files-pattern",
        default="**/*requirements*.txt",
        help="Glob pattern to identify requirements files",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Generate GitHub Actions compatible summary",
    )
    parser.add_argument(
        "--fail-on-high-priority",
        action="store_true",
        help="Return non-zero exit code if high priority issues are found",
    )

    args = parser.parse_args()

    tester = HardwareCompatibilityTester(
        output_dir=args.output_dir, req_files_pattern=args.req_files_pattern
    )

    exit_code = tester.run()

    if args.github_summary and os.environ.get("GITHUB_STEP_SUMMARY"):
        # Get the GITHUB_STEP_SUMMARY environment variable
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            # Copy the GitHub summary to the step summary file
            summary_path = os.path.join(args.output_dir, "github_summary.md")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as src:
                    with open(step_summary, "a", encoding="utf-8") as dest:
                        dest.write(src.read())
                print("Added summary to GitHub Actions output")

    if args.fail_on_high_priority:
        sys.exit(exit_code)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
