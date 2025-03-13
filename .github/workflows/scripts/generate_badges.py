#!/usr/bin/env python
"""
CI Badges Generator

This script generates badges for different aspects of the project such as:
- Test coverage
- Build status
- Security status
- Platform compatibility
- Documentation status

These badges can be included in the README file to provide quick insights into the project's health.
"""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

import requests


class BadgesGenerator:
    def __init__(self, artifacts_dir="ci_artifacts", output_dir="ci_artifacts/badges"):
        self.artifacts_dir = artifacts_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define badge templates with shields.io format
        self.badge_base_url = "https://img.shields.io/badge"

        # Badge colors
        self.colors = {
            "success": "brightgreen",
            "warning": "yellow",
            "error": "red",
            "info": "blue",
            "inactive": "lightgrey",
        }

        # Badge details
        self.badges = {
            "build": {
                "label": "build",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
            "coverage": {
                "label": "coverage",
                "message": "0%",
                "color": self.colors["inactive"],
            },
            "tests": {
                "label": "tests",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
            "security": {
                "label": "security",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
            "platforms": {
                "label": "platforms",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
            "docs": {
                "label": "docs",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
            "dependencies": {
                "label": "dependencies",
                "message": "unknown",
                "color": self.colors["inactive"],
            },
        }

        # Links to GitHub Actions workflows
        self.github_actions_url = os.environ.get(
            "GITHUB_SERVER_URL", "https://github.com"
        )
        self.github_repo = os.environ.get("GITHUB_REPOSITORY", "")
        self.actions_base_url = f"{self.github_actions_url}/{self.github_repo}/actions"

    def generate_build_badge(self):
        """Generate build status badge based on CI results."""
        # Check for environment variables indicating build status
        build_result = os.environ.get("GITHUB_JOB_STATUS", "")

        if build_result == "success":
            self.badges["build"]["message"] = "passing"
            self.badges["build"]["color"] = self.colors["success"]
        elif build_result == "failure":
            self.badges["build"]["message"] = "failing"
            self.badges["build"]["color"] = self.colors["error"]
        else:
            # Look for artifacts that might indicate build status
            perf_files = glob.glob(
                f"{self.artifacts_dir}/performance*/performance_data.csv"
            )
            if perf_files:
                self.badges["build"]["message"] = "passing"
                self.badges["build"]["color"] = self.colors["success"]

    def generate_coverage_badge(self):
        """Generate code coverage badge based on coverage reports."""
        coverage_files = glob.glob(f"{self.artifacts_dir}/coverage*/*.xml") + glob.glob(
            f"{self.artifacts_dir}/*/coverage.xml"
        )

        linux_coverage = None

        for coverage_file in coverage_files:
            try:
                tree = ET.parse(coverage_file)
                root = tree.getroot()

                # Extract platform from filename or path
                platform = "unknown"
                if "linux" in coverage_file.lower():
                    platform = "linux"

                # Use Linux coverage as the primary metric if available
                if platform == "linux":
                    line_rate = float(root.attrib.get("line-rate", 0))
                    coverage_pct = round(line_rate * 100, 1)
                    linux_coverage = coverage_pct
            except Exception as e:
                print(f"Error processing coverage file {coverage_file}: {str(e)}")

        # If we found Linux coverage, use it for the badge
        if linux_coverage is not None:
            self.badges["coverage"]["message"] = f"{linux_coverage}%"

            # Set color based on coverage percentage
            if linux_coverage >= 80:
                self.badges["coverage"]["color"] = self.colors["success"]
            elif linux_coverage >= 60:
                self.badges["coverage"]["color"] = self.colors["warning"]
            else:
                self.badges["coverage"]["color"] = self.colors["error"]

    def generate_tests_badge(self):
        """Generate tests status badge based on test results."""
        metrics_file = os.path.join(self.artifacts_dir, "metrics/ci_metrics.json")

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)

                test_metrics = metrics.get("test_metrics", {})
                total = test_metrics.get("total_tests", 0)
                passed = test_metrics.get("passed_tests", 0)

                if total > 0:
                    pass_rate = round((passed / total) * 100, 1)
                    self.badges["tests"]["message"] = f"{passed}/{total} ({pass_rate}%)"

                    # Set color based on pass rate
                    if pass_rate >= 90:
                        self.badges["tests"]["color"] = self.colors["success"]
                    elif pass_rate >= 75:
                        self.badges["tests"]["color"] = self.colors["warning"]
                    else:
                        self.badges["tests"]["color"] = self.colors["error"]
            except Exception as e:
                print(f"Error reading metrics file: {str(e)}")

    def generate_security_badge(self):
        """Generate security status badge based on security scan results."""
        metrics_file = os.path.join(self.artifacts_dir, "metrics/ci_metrics.json")

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)

                security_metrics = metrics.get("security_metrics", {})
                total_vulns = security_metrics.get("total_vulnerabilities", 0)
                high_vulns = security_metrics.get("high_vulnerabilities", 0)

                if total_vulns == 0:
                    self.badges["security"]["message"] = "no vulnerabilities"
                    self.badges["security"]["color"] = self.colors["success"]
                else:
                    self.badges["security"][
                        "message"
                    ] = f"{total_vulns} vulnerabilities"

                    # Set color based on high severity vulnerabilities
                    if high_vulns > 0:
                        self.badges["security"]["color"] = self.colors["error"]
                    else:
                        self.badges["security"]["color"] = self.colors["warning"]
            except Exception as e:
                print(f"Error reading metrics file: {str(e)}")

    def generate_platforms_badge(self):
        """Generate platforms compatibility badge."""
        # Check for existence of platform compatibility reports
        linux_report = glob.glob(f"{self.artifacts_dir}/compatibility-report-linux*")
        windows_report = glob.glob(
            f"{self.artifacts_dir}/compatibility-report-windows*"
        )
        macos_report = glob.glob(f"{self.artifacts_dir}/compatibility-report-macos*")

        supported_platforms = []
        if linux_report:
            supported_platforms.append("Linux")
        if windows_report:
            supported_platforms.append("Windows")
        if macos_report:
            supported_platforms.append("macOS")

        if supported_platforms:
            self.badges["platforms"]["message"] = ", ".join(supported_platforms)

            # Set color based on the number of supported platforms
            if len(supported_platforms) >= 3:
                self.badges["platforms"]["color"] = self.colors["success"]
            elif len(supported_platforms) >= 2:
                self.badges["platforms"]["color"] = self.colors["warning"]
            else:
                self.badges["platforms"]["color"] = self.colors["info"]

    def generate_docs_badge(self):
        """Generate documentation status badge."""
        # Check for existence of API documentation
        api_docs = glob.glob(f"{self.artifacts_dir}/api-documentation*") + glob.glob(
            f"{self.artifacts_dir}/api-endpoints-summary*"
        )

        if api_docs:
            self.badges["docs"]["message"] = "available"
            self.badges["docs"]["color"] = self.colors["success"]
        else:
            self.badges["docs"]["message"] = "not available"
            self.badges["docs"]["color"] = self.colors["warning"]

    def generate_dependencies_badge(self):
        """Generate dependencies status badge."""
        # Check for dependency updates
        dependency_report = os.path.join(
            self.artifacts_dir, "dependencies/main-diff.txt"
        )

        if os.path.exists(dependency_report):
            try:
                with open(dependency_report) as f:
                    content = f.read()

                # Count the number of outdated packages (lines starting with "+")
                outdated_count = content.count("\n+")

                if outdated_count == 0:
                    self.badges["dependencies"]["message"] = "up to date"
                    self.badges["dependencies"]["color"] = self.colors["success"]
                else:
                    self.badges["dependencies"][
                        "message"
                    ] = f"{outdated_count} outdated"
                    self.badges["dependencies"]["color"] = self.colors["warning"]
            except Exception as e:
                print(f"Error reading dependency report: {str(e)}")

    def download_badge(self, badge_type):
        """Download a badge from shields.io."""
        badge_config = self.badges[badge_type]

        # Construct the badge URL
        badge_url = f"{self.badge_base_url}/{badge_config['label']}-{badge_config['message']}-{badge_config['color']}"

        # Replace spaces with underscores in the URL
        badge_url = badge_url.replace(" ", "_")

        try:
            response = requests.get(badge_url)
            if response.status_code == 200:
                output_file = os.path.join(self.output_dir, f"{badge_type}.svg")
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"Badge downloaded: {output_file}")
                return True
            else:
                print(f"Failed to download badge: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading badge: {str(e)}")
            return False

    def generate_badge_urls(self):
        """Generate URLs for all badges to be used in README."""
        badge_urls = {}

        for badge_type, badge_config in self.badges.items():
            # Construct the badge URL
            badge_url = f"{self.badge_base_url}/{badge_config['label']}-{badge_config['message']}-{badge_config['color']}"

            # Replace spaces with underscores in the URL
            badge_url = badge_url.replace(" ", "_")

            # Link URL (typically to CI actions)
            link_url = f"{self.actions_base_url}"

            badge_urls[badge_type] = {"image_url": badge_url, "link_url": link_url}

        return badge_urls

    def generate_badge_markdown(self):
        """Generate markdown for badges to be inserted into README."""
        badge_urls = self.generate_badge_urls()

        markdown = "<!-- CI Badges - Auto-generated, do not edit -->\n"
        markdown += "# Project Status\n\n"

        for badge_type, urls in badge_urls.items():
            markdown += f"[![{self.badges[badge_type]['label']}]({urls['image_url']})]({urls['link_url']})\n"

        markdown += "\n<!-- End of CI Badges -->\n"

        # Write to a file
        output_file = os.path.join(self.output_dir, "badges.md")
        with open(output_file, "w") as f:
            f.write(markdown)

        print(f"Badge markdown generated: {output_file}")
        return markdown

    def generate_github_summary(self):
        """Generate GitHub step summary with badge information."""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## CI Badges\n\n")

            f.write("| Badge | Status | Color |\n")
            f.write("|-------|--------|-------|\n")

            for _badge_type, badge_config in self.badges.items():
                f.write(
                    f"| {badge_config['label']} | {badge_config['message']} | {badge_config['color']} |\n"
                )

            f.write("\nBadges have been generated and can be added to your README.md\n")

    def run(self):
        """Run the badges generation process."""
        print("Generating badges...")

        # Generate all badges
        self.generate_build_badge()
        self.generate_coverage_badge()
        self.generate_tests_badge()
        self.generate_security_badge()
        self.generate_platforms_badge()
        self.generate_docs_badge()
        self.generate_dependencies_badge()

        # Download badges from shields.io
        for badge_type in self.badges:
            self.download_badge(badge_type)

        # Generate markdown for README
        self.generate_badge_markdown()

        # Generate GitHub step summary
        self.generate_github_summary()

        print("Badges generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate badges for CI status")
    parser.add_argument(
        "--artifacts-dir",
        default="ci_artifacts",
        help="Directory containing CI artifacts",
    )
    parser.add_argument(
        "--output-dir", default="ci_artifacts/badges", help="Directory to store badges"
    )
    args = parser.parse_args()

    generator = BadgesGenerator(
        artifacts_dir=args.artifacts_dir, output_dir=args.output_dir
    )
    generator.run()


if __name__ == "__main__":
    main()
