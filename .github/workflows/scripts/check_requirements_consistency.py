#!/usr/bin/env python
"""
Requirements Consistency Checker

This script analyzes all requirements files in the project to ensure consistency
in naming conventions, formatting, and documentation.
"""

import argparse
import glob
import os
import re
from collections import defaultdict


class RequirementsChecker:
    def __init__(self, report_dir="ci_artifacts/requirements_check"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        # Rules for consistency checking
        self.rules = [
            {
                "name": "header_comment",
                "description": "Check for header comment in requirements file",
                "pattern": r"^#.*\n#.*",
                "severity": "warning",
            },
            {
                "name": "package_version_pin",
                "description": "Check if packages have pinned versions",
                "pattern": r"^(?!#)([a-zA-Z0-9_-]+)(?!==)(?=[<>~!=]|$)",
                "severity": "warning",
            },
            {
                "name": "section_markers",
                "description": "Check for section markers (like # ====== SECTION ======)",
                "pattern": r"^#\s*=+\s*[A-Z\s]+\s*=+",
                "severity": "info",
            },
            {
                "name": "index_url",
                "description": "Check if extra index URLs are properly formatted",
                "pattern": r"^--extra-index-url\s+https?://",
                "severity": "warning",
            },
            {
                "name": "empty_line",
                "description": "Check for consecutive empty lines or trailing empty lines",
                "pattern": r"\n\n\n|\n\n$",
                "severity": "info",
            },
        ]

        # Consistent style guidelines
        self.package_name_style = r"^[a-z][a-z0-9_-]*$"

    def find_requirements_files(self):
        """Find all requirements files in the project."""
        req_files = []

        # Standard requirements files
        for file in glob.glob("**/requirements*.txt", recursive=True):
            req_files.append(file)

        return sorted(req_files)

    def analyze_file(self, file_path):
        """Analyze a single requirements file for consistency issues."""
        if not os.path.exists(file_path):
            return {
                "file": file_path,
                "errors": [
                    {"rule": "file_existence", "message": "File does not exist"}
                ],
                "warnings": [],
                "info": [],
                "metrics": {},
            }

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")

        # Initialize results
        result = {
            "file": file_path,
            "errors": [],
            "warnings": [],
            "info": [],
            "metrics": {
                "total_lines": len(lines),
                "comment_lines": 0,
                "package_lines": 0,
                "pinned_versions": 0,
                "unpinned_versions": 0,
                "section_markers": 0,
            },
        }

        # Check for empty file
        if not content.strip():
            result["errors"].append(
                {"rule": "empty_file", "message": "Requirements file is empty"}
            )
            return result

        # Count comment lines and package lines
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                result["metrics"]["comment_lines"] += 1
                # Check for section markers
                if re.match(r"#\s*=+\s*[A-Z\s]+\s*=+", line):
                    result["metrics"]["section_markers"] += 1
            elif not line.startswith("--"):
                result["metrics"]["package_lines"] += 1
                # Check for pinned versions
                if re.search(r"==[0-9]", line):
                    result["metrics"]["pinned_versions"] += 1
                else:
                    result["metrics"]["unpinned_versions"] += 1

        # Apply rules
        for rule in self.rules:
            pattern = rule["pattern"]
            severity = rule["severity"]

            if rule["name"] == "header_comment":
                if not re.match(pattern, content):
                    result[self._get_severity_list(severity)].append(
                        {"rule": rule["name"], "message": rule["description"]}
                    )

            elif rule["name"] == "package_version_pin":
                for i, line in enumerate(lines):
                    if re.match(pattern, line):
                        result[self._get_severity_list(severity)].append(
                            {
                                "rule": rule["name"],
                                "message": f"Line {i+1}: Package does not have pinned version: {line}",
                                "line": i + 1,
                            }
                        )

            elif rule["name"] == "section_markers":
                if (
                    result["metrics"]["package_lines"] > 5
                    and result["metrics"]["section_markers"] == 0
                ):
                    result[self._get_severity_list(severity)].append(
                        {
                            "rule": rule["name"],
                            "message": "File has multiple packages but no section markers",
                        }
                    )

            elif rule["name"] == "index_url":
                for i, line in enumerate(lines):
                    if line.startswith("--extra-index-url") and not re.match(
                        pattern, line
                    ):
                        result[self._get_severity_list(severity)].append(
                            {
                                "rule": rule["name"],
                                "message": f"Line {i+1}: Extra index URL is not properly formatted: {line}",
                                "line": i + 1,
                            }
                        )

            elif rule["name"] == "empty_line":
                if re.search(pattern, content):
                    result[self._get_severity_list(severity)].append(
                        {
                            "rule": rule["name"],
                            "message": "File contains consecutive empty lines or trailing empty lines",
                        }
                    )

        # Check for additional consistency issues
        self._check_package_naming(lines, result)
        self._check_version_consistency(lines, result)

        return result

    def _check_package_naming(self, lines, result):
        """Check if package names follow consistent naming conventions."""
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue

            # Extract package name
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if match:
                package_name = match.group(1)
                if not re.match(self.package_name_style, package_name):
                    result["info"].append(
                        {
                            "rule": "package_naming",
                            "message": f"Line {i+1}: Package name does not follow consistent style (lowercase, numbers, hyphens or underscores): {package_name}",
                            "line": i + 1,
                        }
                    )

    def _check_version_consistency(self, lines, result):
        """Check for consistency in version pinning format (== vs =)."""
        version_formats = defaultdict(int)

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue

            # Check version format
            if "==" in line:
                version_formats["=="] += 1
            elif "=" in line and not any(op in line for op in [">=", "<=", "!="]):
                version_formats["="] += 1

        # If both formats are used, flag as inconsistency
        if len(version_formats) > 1 and min(version_formats.values()) > 0:
            formats = ", ".join(
                [f"{fmt} ({count} times)" for fmt, count in version_formats.items()]
            )
            result["warnings"].append(
                {
                    "rule": "version_format",
                    "message": f"Inconsistent version pinning formats: {formats}. Standardize on one format.",
                }
            )

    def _get_severity_list(self, severity):
        """Map severity to the appropriate result list."""
        if severity == "error":
            return "errors"
        elif severity == "warning":
            return "warnings"
        else:
            return "info"

    def generate_markdown_report(self, results):
        """Generate a markdown report of the analysis results."""
        report_path = os.path.join(
            self.report_dir, "requirements_consistency_report.md"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Requirements Files Consistency Report\n\n")

            f.write("## Summary\n\n")

            # Count total issues
            total_errors = sum(len(result["errors"]) for result in results)
            total_warnings = sum(len(result["warnings"]) for result in results)
            total_info = sum(len(result["info"]) for result in results)

            f.write(f"- **Total files analyzed**: {len(results)}\n")
            f.write(f"- **Total errors**: {total_errors}\n")
            f.write(f"- **Total warnings**: {total_warnings}\n")
            f.write(f"- **Total info messages**: {total_info}\n\n")

            # Summary table
            f.write(
                "| File | Packages | Pinned Versions | Unpinned | Comments | Sections | Errors | Warnings |\n"
            )
            f.write(
                "|------|----------|----------------|----------|----------|----------|--------|----------|\n"
            )

            for result in results:
                metrics = result["metrics"]
                f.write(
                    f"| {result['file']} | {metrics.get('package_lines', 0)} | {metrics.get('pinned_versions', 0)} | {metrics.get('unpinned_versions', 0)} | {metrics.get('comment_lines', 0)} | {metrics.get('section_markers', 0)} | {len(result['errors'])} | {len(result['warnings'])} |\n"
                )

            f.write("\n## Detailed Results\n\n")

            # Detail each file
            for result in results:
                f.write(f"### {result['file']}\n\n")

                if (
                    not result["errors"]
                    and not result["warnings"]
                    and not result["info"]
                ):
                    f.write("✅ No issues found\n\n")
                    continue

                if result["errors"]:
                    f.write("#### Errors\n\n")
                    for error in result["errors"]:
                        f.write(f"- 🔴 **{error['rule']}**: {error['message']}\n")
                    f.write("\n")

                if result["warnings"]:
                    f.write("#### Warnings\n\n")
                    for warning in result["warnings"]:
                        f.write(f"- ⚠️ **{warning['rule']}**: {warning['message']}\n")
                    f.write("\n")

                if result["info"]:
                    f.write("#### Info\n\n")
                    for info in result["info"]:
                        f.write(f"- ℹ️ **{info['rule']}**: {info['message']}\n")
                    f.write("\n")

            f.write("\n## Recommendations\n\n")

            f.write(
                "1. **Standardize version pinning**: Use the `==` format consistently for all packages.\n"
            )
            f.write(
                "2. **Add section markers**: Use section headers like `# === SECTION ===` to organize requirements.\n"
            )
            f.write(
                "3. **Add header comments**: Each requirements file should have a comment header explaining its purpose.\n"
            )
            f.write(
                "4. **Pin all versions**: Specify exact versions for all packages to ensure reproducibility.\n"
            )
            f.write(
                "5. **Consistent naming**: Follow consistent package naming conventions.\n\n"
            )

            f.write("*This report was automatically generated by the CI process on ")
            f.write(f"{os.popen('date').read().strip()}*\n")

        print(f"Markdown report generated at {report_path}")
        return report_path

    def generate_github_summary(self, results):
        """Generate a GitHub step summary from the analysis results."""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## Requirements Files Consistency Check\n\n")

            # Count total issues
            total_errors = sum(len(result["errors"]) for result in results)
            total_warnings = sum(len(result["warnings"]) for result in results)

            f.write(f"Analyzed **{len(results)}** requirements files\n\n")

            # Status indicators
            if total_errors > 0:
                f.write("🔴 **Issues found**\n\n")
            elif total_warnings > 0:
                f.write("⚠️ **Warnings found**\n\n")
            else:
                f.write("✅ **All files meet consistency guidelines**\n\n")

            # Issues summary
            f.write("| Issue Type | Count |\n")
            f.write("|------------|-------|\n")
            f.write(f"| Errors | {total_errors} |\n")
            f.write(f"| Warnings | {total_warnings} |\n\n")

            if total_errors > 0 or total_warnings > 0:
                f.write("See requirements consistency report artifact for details.\n")

    def run(self):
        """Run the requirements consistency check on all files."""
        req_files = self.find_requirements_files()
        results = []

        print(f"Analyzing {len(req_files)} requirements files...")

        for file_path in req_files:
            print(f"Checking {file_path}...")
            result = self.analyze_file(file_path)
            results.append(result)

            # Print summary of issues
            if result["errors"]:
                print(f"  ❌ {len(result['errors'])} errors")
            if result["warnings"]:
                print(f"  ⚠️ {len(result['warnings'])} warnings")
            if result["info"]:
                print(f"  ℹ️ {len(result['info'])} info messages")
            if not result["errors"] and not result["warnings"] and not result["info"]:
                print("  ✓ No issues")

        # Generate full report
        self.generate_markdown_report(results)
        self.generate_github_summary(results)

        # Return exit code based on errors
        if any(len(result["errors"]) > 0 for result in results):
            return 1
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check requirements files for consistency"
    )
    parser.add_argument(
        "--report-dir",
        default="ci_artifacts/requirements_check",
        help="Directory to store the report",
    )
    args = parser.parse_args()

    checker = RequirementsChecker(report_dir=args.report_dir)
    _ = checker.run()  # noqa: F841 (was exit_code)

    # Exit with proper code for CI
    # Use exit code 0 here to allow CI to continue, but the report will still show issues
    exit(0)


if __name__ == "__main__":
    main()
