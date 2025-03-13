#!/usr/bin/env python
"""
CI Metrics Collector

This script collects key performance indicators from the CI process and generates
a metrics dashboard with insights on test performance, coverage trends, and build times.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


class CIMetricsCollector:
    def __init__(self, artifacts_dir="ci_artifacts", output_dir="ci_artifacts/metrics"):
        self.artifacts_dir = artifacts_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Time series data storage
        self.metrics_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER", "0"),
            "test_metrics": {},
            "coverage_metrics": {},
            "performance_metrics": {},
            "security_metrics": {},
            "platform_metrics": {},
        }

    def collect_test_metrics(self):
        """Collect metrics about test execution."""
        # Initialize test metrics
        self.metrics_data["test_metrics"] = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_duration": 0,
            "platform_results": {},
        }

        # Look for test result files
        test_result_files = glob.glob(
            f"{self.artifacts_dir}/*/test_results.json"
        ) + glob.glob(f"{self.artifacts_dir}/*/test_results_*.json")

        for result_file in test_result_files:
            try:
                with open(result_file) as f:
                    results = json.load(f)

                # Extract platform from filename or content
                platform = "unknown"
                if "linux" in result_file.lower():
                    platform = "linux"
                elif "windows" in result_file.lower():
                    platform = "windows"
                elif "macos" in result_file.lower():
                    platform = "macos"

                # Initialize platform metrics if not present
                if (
                    platform
                    not in self.metrics_data["test_metrics"]["platform_results"]
                ):
                    self.metrics_data["test_metrics"]["platform_results"][platform] = {
                        "total_tests": 0,
                        "passed_tests": 0,
                        "failed_tests": 0,
                        "skipped_tests": 0,
                        "test_duration": 0,
                    }

                # Aggregate test counts
                platform_metrics = self.metrics_data["test_metrics"][
                    "platform_results"
                ][platform]

                # Extract relevant metrics from the results file
                platform_metrics["total_tests"] += results.get("total", 0)
                platform_metrics["passed_tests"] += results.get("passed", 0)
                platform_metrics["failed_tests"] += results.get("failed", 0)
                platform_metrics["skipped_tests"] += results.get("skipped", 0)
                platform_metrics["test_duration"] += results.get("duration", 0)

                # Update total metrics
                self.metrics_data["test_metrics"]["total_tests"] += results.get(
                    "total", 0
                )
                self.metrics_data["test_metrics"]["passed_tests"] += results.get(
                    "passed", 0
                )
                self.metrics_data["test_metrics"]["failed_tests"] += results.get(
                    "failed", 0
                )
                self.metrics_data["test_metrics"]["skipped_tests"] += results.get(
                    "skipped", 0
                )
                self.metrics_data["test_metrics"]["test_duration"] += results.get(
                    "duration", 0
                )

            except Exception as e:
                print(f"Error processing test results file {result_file}: {str(e)}")

        # Calculate pass rate
        total = self.metrics_data["test_metrics"]["total_tests"]
        if total > 0:
            self.metrics_data["test_metrics"]["pass_rate"] = round(
                (self.metrics_data["test_metrics"]["passed_tests"] / total) * 100, 2
            )
        else:
            self.metrics_data["test_metrics"]["pass_rate"] = 0

        print(f"Collected test metrics: {self.metrics_data['test_metrics']}")

    def collect_coverage_metrics(self):
        """Collect code coverage metrics."""
        # Initialize coverage metrics
        self.metrics_data["coverage_metrics"] = {
            "overall_coverage": 0,
            "platform_coverage": {},
            "file_coverage": {},
        }

        # Look for coverage XML files
        coverage_files = glob.glob(f"{self.artifacts_dir}/coverage*/*.xml") + glob.glob(
            f"{self.artifacts_dir}/*/coverage.xml"
        )

        for coverage_file in coverage_files:
            try:
                tree = ET.parse(coverage_file)
                root = tree.getroot()

                # Extract platform from filename or path
                platform = "unknown"
                if "linux" in coverage_file.lower():
                    platform = "linux"
                elif "windows" in coverage_file.lower():
                    platform = "windows"
                elif "macos" in coverage_file.lower():
                    platform = "macos"

                # Extract overall coverage
                line_rate = float(root.attrib.get("line-rate", 0))
                coverage_pct = round(line_rate * 100, 2)

                # Store platform-specific coverage
                self.metrics_data["coverage_metrics"]["platform_coverage"][
                    platform
                ] = coverage_pct

                # Extract file-specific coverage if this is the main coverage file
                if platform == "linux":
                    for package in root.findall(".//package"):
                        _ = package.attrib.get("name", "")  # noqa: F841 (was package_name)
                        for cls in package.findall(".//class"):
                            filename = cls.attrib.get("filename", "")
                            if filename:
                                file_line_rate = float(cls.attrib.get("line-rate", 0))
                                file_coverage = round(file_line_rate * 100, 2)
                                self.metrics_data["coverage_metrics"]["file_coverage"][
                                    filename
                                ] = file_coverage

                # Use Linux coverage as the overall coverage if available
                if (
                    platform == "linux"
                    and "overall_coverage" in self.metrics_data["coverage_metrics"]
                ):
                    self.metrics_data["coverage_metrics"][
                        "overall_coverage"
                    ] = coverage_pct

            except Exception as e:
                print(f"Error processing coverage file {coverage_file}: {str(e)}")

        # Calculate average coverage if no Linux coverage file was found
        if (
            self.metrics_data["coverage_metrics"]["overall_coverage"] == 0
            and self.metrics_data["coverage_metrics"]["platform_coverage"]
        ):
            platform_coverages = list(
                self.metrics_data["coverage_metrics"]["platform_coverage"].values()
            )
            self.metrics_data["coverage_metrics"]["overall_coverage"] = round(
                sum(platform_coverages) / len(platform_coverages), 2
            )

        print(f"Collected coverage metrics: {self.metrics_data['coverage_metrics']}")

    def collect_performance_metrics(self):
        """Collect CI performance metrics."""
        # Initialize performance metrics
        self.metrics_data["performance_metrics"] = {
            "total_duration": 0,
            "step_durations": {},
            "slowest_steps": [],
        }

        # Look for performance data files
        performance_files = glob.glob(
            f"{self.artifacts_dir}/performance*/performance_data.csv"
        )

        for perf_file in performance_files:
            try:
                with open(perf_file) as f:
                    reader = csv.DictReader(f)
                    step_durations = {}
                    for row in reader:
                        step_name = row.get("Step", "")
                        duration = float(row.get("Duration", 0))
                        exit_code = int(row.get("ExitCode", 0))

                        # Store step duration
                        step_durations[step_name] = {
                            "duration": duration,
                            "exit_code": exit_code,
                        }

                        # Update total duration
                        self.metrics_data["performance_metrics"][
                            "total_duration"
                        ] += duration

                    # Store step durations
                    self.metrics_data["performance_metrics"][
                        "step_durations"
                    ] = step_durations

                    # Find slowest steps
                    sorted_steps = sorted(
                        step_durations.items(),
                        key=lambda x: x[1]["duration"],
                        reverse=True,
                    )
                    self.metrics_data["performance_metrics"]["slowest_steps"] = [
                        {"name": step[0], "duration": step[1]["duration"]}
                        for step in sorted_steps[:5]  # Top 5 slowest
                    ]

            except Exception as e:
                print(f"Error processing performance file {perf_file}: {str(e)}")

        print(
            f"Collected performance metrics: {self.metrics_data['performance_metrics']}"
        )

    def collect_security_metrics(self):
        """Collect security metrics."""
        # Initialize security metrics
        self.metrics_data["security_metrics"] = {
            "total_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "vulnerable_packages": [],
        }

        # Look for security scan files
        security_files = glob.glob(f"{self.artifacts_dir}/security*/*_scan.txt")

        for security_file in security_files:
            try:
                with open(security_file) as f:
                    content = f.read()

                # Parse vulnerability counts
                high_matches = content.count("high severity")
                medium_matches = content.count("medium severity")
                low_matches = content.count("low severity")

                # Update metrics
                self.metrics_data["security_metrics"][
                    "high_vulnerabilities"
                ] += high_matches
                self.metrics_data["security_metrics"][
                    "medium_vulnerabilities"
                ] += medium_matches
                self.metrics_data["security_metrics"][
                    "low_vulnerabilities"
                ] += low_matches
                self.metrics_data["security_metrics"]["total_vulnerabilities"] += (
                    high_matches + medium_matches + low_matches
                )

                # Extract vulnerable package names
                lines = content.split("\n")
                for line in lines:
                    if "vulnerability found in" in line:
                        parts = line.split("vulnerability found in")
                        if len(parts) > 1:
                            package = parts[1].strip()
                            if (
                                package
                                and package
                                not in self.metrics_data["security_metrics"][
                                    "vulnerable_packages"
                                ]
                            ):
                                self.metrics_data["security_metrics"][
                                    "vulnerable_packages"
                                ].append(package)

            except Exception as e:
                print(f"Error processing security file {security_file}: {str(e)}")

        print(f"Collected security metrics: {self.metrics_data['security_metrics']}")

    def collect_platform_metrics(self):
        """Collect platform compatibility metrics."""
        # Initialize platform metrics
        self.metrics_data["platform_metrics"] = {
            "total_issues": 0,
            "platform_issues": {},
            "severity_counts": {"error": 0, "warning": 0, "info": 0},
        }

        # Look for platform compatibility JSON file
        platform_files = glob.glob(
            f"{self.artifacts_dir}/platform*/platform_issues.json"
        )

        for platform_file in platform_files:
            try:
                with open(platform_file) as f:
                    platform_issues = json.load(f)

                # Count total issues
                self.metrics_data["platform_metrics"]["total_issues"] = len(
                    platform_issues
                )

                # Count by severity
                for issue in platform_issues:
                    severity = issue.get("severity", "info")
                    self.metrics_data["platform_metrics"]["severity_counts"][
                        severity
                    ] += 1

                    # Count by platform
                    for platform in issue.get("platforms", []):
                        if (
                            platform
                            not in self.metrics_data["platform_metrics"][
                                "platform_issues"
                            ]
                        ):
                            self.metrics_data["platform_metrics"]["platform_issues"][
                                platform
                            ] = 0
                        self.metrics_data["platform_metrics"]["platform_issues"][
                            platform
                        ] += 1

            except Exception as e:
                print(f"Error processing platform file {platform_file}: {str(e)}")

        print(f"Collected platform metrics: {self.metrics_data['platform_metrics']}")

    def generate_metrics_files(self):
        """Generate metrics files and visualizations."""
        # Save the metrics data to JSON
        metrics_file = os.path.join(self.output_dir, "ci_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_data, f, indent=2)

        # Append to historical data if it exists
        history_file = os.path.join(self.output_dir, "metrics_history.json")
        history_data = []

        if os.path.exists(history_file):
            try:
                with open(history_file) as f:
                    history_data = json.load(f)
            except Exception as e:
                print(f"Error reading history file: {str(e)}")

        # Add current data to history
        history_entry = {
            "timestamp": self.metrics_data["timestamp"],
            "run_id": self.metrics_data["run_id"],
            "run_number": self.metrics_data["run_number"],
            "test_pass_rate": self.metrics_data["test_metrics"].get("pass_rate", 0),
            "overall_coverage": self.metrics_data["coverage_metrics"].get(
                "overall_coverage", 0
            ),
            "total_duration": self.metrics_data["performance_metrics"].get(
                "total_duration", 0
            ),
            "total_vulnerabilities": self.metrics_data["security_metrics"].get(
                "total_vulnerabilities", 0
            ),
            "platform_issues": self.metrics_data["platform_metrics"].get(
                "total_issues", 0
            ),
        }
        history_data.append(history_entry)

        # Keep only the last 30 entries
        if len(history_data) > 30:
            history_data = history_data[-30:]

        # Save updated history data
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

        # Generate markdown report
        self.generate_markdown_report()

        # Generate visualizations if matplotlib is available
        try:
            self.generate_visualizations(history_data)
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

    def generate_markdown_report(self):
        """Generate a markdown report of the metrics."""
        report_file = os.path.join(self.output_dir, "ci_metrics_report.md")

        with open(report_file, "w") as f:
            f.write("# CI Metrics Report\n\n")
            f.write(f"Generated on: {self.metrics_data['timestamp']}\n")
            f.write(f"Run ID: {self.metrics_data['run_id']}\n")
            f.write(f"Run Number: {self.metrics_data['run_number']}\n\n")

            # Test Metrics
            f.write("## Test Metrics\n\n")
            test_metrics = self.metrics_data["test_metrics"]
            f.write(f"- **Total Tests**: {test_metrics.get('total_tests', 0)}\n")
            f.write(f"- **Passed Tests**: {test_metrics.get('passed_tests', 0)}\n")
            f.write(f"- **Failed Tests**: {test_metrics.get('failed_tests', 0)}\n")
            f.write(f"- **Skipped Tests**: {test_metrics.get('skipped_tests', 0)}\n")
            f.write(f"- **Pass Rate**: {test_metrics.get('pass_rate', 0)}%\n")
            f.write(
                f"- **Total Duration**: {test_metrics.get('test_duration', 0):.2f} seconds\n\n"
            )

            # Platform Test Results
            if test_metrics.get("platform_results"):
                f.write("### Platform Test Results\n\n")
                f.write(
                    "| Platform | Tests | Passed | Failed | Skipped | Pass Rate |\n"
                )
                f.write("|----------|-------|--------|--------|---------|----------|\n")

                for platform, metrics in test_metrics.get(
                    "platform_results", {}
                ).items():
                    total = metrics.get("total_tests", 0)
                    pass_rate = (
                        round((metrics.get("passed_tests", 0) / total) * 100, 2)
                        if total > 0
                        else 0
                    )
                    f.write(
                        f"| {platform.capitalize()} | {total} | {metrics.get('passed_tests', 0)} | {metrics.get('failed_tests', 0)} | {metrics.get('skipped_tests', 0)} | {pass_rate}% |\n"
                    )

                f.write("\n")

            # Coverage Metrics
            f.write("## Coverage Metrics\n\n")
            coverage_metrics = self.metrics_data["coverage_metrics"]
            f.write(
                f"- **Overall Coverage**: {coverage_metrics.get('overall_coverage', 0)}%\n\n"
            )

            # Platform Coverage
            if coverage_metrics.get("platform_coverage"):
                f.write("### Platform Coverage\n\n")
                f.write("| Platform | Coverage |\n")
                f.write("|----------|----------|\n")

                for platform, coverage in coverage_metrics.get(
                    "platform_coverage", {}
                ).items():
                    f.write(f"| {platform.capitalize()} | {coverage}% |\n")

                f.write("\n")

            # File Coverage (Top 10 and Bottom 10)
            if coverage_metrics.get("file_coverage"):
                file_coverages = coverage_metrics.get("file_coverage", {})
                sorted_files = sorted(file_coverages.items(), key=lambda x: x[1])

                # Bottom 10 files
                f.write("### Lowest Coverage Files\n\n")
                f.write("| File | Coverage |\n")
                f.write("|------|----------|\n")

                for filename, coverage in sorted_files[:10]:  # Bottom 10
                    f.write(f"| {filename} | {coverage}% |\n")

                f.write("\n")

                # Top 10 files
                f.write("### Highest Coverage Files\n\n")
                f.write("| File | Coverage |\n")
                f.write("|------|----------|\n")

                for filename, coverage in sorted_files[-10:]:  # Top 10, reversed
                    f.write(f"| {filename} | {coverage}% |\n")

                f.write("\n")

            # Performance Metrics
            f.write("## Performance Metrics\n\n")
            performance_metrics = self.metrics_data["performance_metrics"]
            f.write(
                f"- **Total CI Duration**: {performance_metrics.get('total_duration', 0):.2f} seconds\n\n"
            )

            # Slowest Steps
            if performance_metrics.get("slowest_steps"):
                f.write("### Slowest Steps\n\n")
                f.write("| Step | Duration (seconds) |\n")
                f.write("|------|--------------------|\n")

                for step in performance_metrics.get("slowest_steps", []):
                    f.write(f"| {step['name']} | {step['duration']:.2f} |\n")

                f.write("\n")

            # Security Metrics
            f.write("## Security Metrics\n\n")
            security_metrics = self.metrics_data["security_metrics"]
            f.write(
                f"- **Total Vulnerabilities**: {security_metrics.get('total_vulnerabilities', 0)}\n"
            )
            f.write(
                f"- **High Severity**: {security_metrics.get('high_vulnerabilities', 0)}\n"
            )
            f.write(
                f"- **Medium Severity**: {security_metrics.get('medium_vulnerabilities', 0)}\n"
            )
            f.write(
                f"- **Low Severity**: {security_metrics.get('low_vulnerabilities', 0)}\n\n"
            )

            # Vulnerable Packages
            if security_metrics.get("vulnerable_packages"):
                f.write("### Vulnerable Packages\n\n")
                for package in security_metrics.get("vulnerable_packages", []):
                    f.write(f"- {package}\n")

                f.write("\n")

            # Platform Compatibility Metrics
            f.write("## Platform Compatibility Metrics\n\n")
            platform_metrics = self.metrics_data["platform_metrics"]
            f.write(
                f"- **Total Platform Issues**: {platform_metrics.get('total_issues', 0)}\n"
            )
            f.write(
                f"- **Error Level Issues**: {platform_metrics.get('severity_counts', {}).get('error', 0)}\n"
            )
            f.write(
                f"- **Warning Level Issues**: {platform_metrics.get('severity_counts', {}).get('warning', 0)}\n"
            )
            f.write(
                f"- **Info Level Issues**: {platform_metrics.get('severity_counts', {}).get('info', 0)}\n\n"
            )

            # Platform-specific Issues
            if platform_metrics.get("platform_issues"):
                f.write("### Platform-specific Issues\n\n")
                f.write("| Platform | Issues |\n")
                f.write("|----------|--------|\n")

                for platform, count in platform_metrics.get(
                    "platform_issues", {}
                ).items():
                    f.write(f"| {platform.capitalize()} | {count} |\n")

        print(f"Generated metrics report at {report_file}")

    def generate_visualizations(self, history_data):
        """Generate visualizations of the metrics over time."""
        # Extract data for charts
        timestamps = [
            entry.get("timestamp", "").split()[0] for entry in history_data
        ]  # Just date part
        pass_rates = [entry.get("test_pass_rate", 0) for entry in history_data]
        coverages = [entry.get("overall_coverage", 0) for entry in history_data]
        durations = [entry.get("total_duration", 0) for entry in history_data]
        vulnerabilities = [
            entry.get("total_vulnerabilities", 0) for entry in history_data
        ]
        platform_issues = [entry.get("platform_issues", 0) for entry in history_data]

        # Set up the figure for multiple plots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Plot pass rate over time
        axs[0, 0].plot(timestamps, pass_rates, marker="o", linestyle="-")
        axs[0, 0].set_title("Test Pass Rate Over Time")
        axs[0, 0].set_ylabel("Pass Rate (%)")
        axs[0, 0].tick_params(axis="x", rotation=45)
        axs[0, 0].grid(True)

        # Plot coverage over time
        axs[0, 1].plot(timestamps, coverages, marker="o", linestyle="-", color="green")
        axs[0, 1].set_title("Code Coverage Over Time")
        axs[0, 1].set_ylabel("Coverage (%)")
        axs[0, 1].tick_params(axis="x", rotation=45)
        axs[0, 1].grid(True)

        # Plot CI duration over time
        axs[1, 0].plot(timestamps, durations, marker="o", linestyle="-", color="orange")
        axs[1, 0].set_title("CI Duration Over Time")
        axs[1, 0].set_ylabel("Duration (seconds)")
        axs[1, 0].tick_params(axis="x", rotation=45)
        axs[1, 0].grid(True)

        # Plot security vulnerabilities over time
        axs[1, 1].plot(
            timestamps, vulnerabilities, marker="o", linestyle="-", color="red"
        )
        axs[1, 1].set_title("Security Vulnerabilities Over Time")
        axs[1, 1].set_ylabel("Vulnerabilities Count")
        axs[1, 1].tick_params(axis="x", rotation=45)
        axs[1, 1].grid(True)

        # Plot platform issues over time
        axs[2, 0].plot(
            timestamps, platform_issues, marker="o", linestyle="-", color="purple"
        )
        axs[2, 0].set_title("Platform Compatibility Issues Over Time")
        axs[2, 0].set_ylabel("Issues Count")
        axs[2, 0].tick_params(axis="x", rotation=45)
        axs[2, 0].grid(True)

        # Create a pie chart for the current run's test results
        test_metrics = self.metrics_data["test_metrics"]
        passed = test_metrics.get("passed_tests", 0)
        failed = test_metrics.get("failed_tests", 0)
        skipped = test_metrics.get("skipped_tests", 0)

        if passed + failed + skipped > 0:
            test_results = [passed, failed, skipped]
            labels = ["Passed", "Failed", "Skipped"]
            colors = ["green", "red", "gray"]

            axs[2, 1].pie(test_results, labels=labels, colors=colors, autopct="%1.1f%%")
            axs[2, 1].set_title("Current Run Test Results")

        # Adjust layout and save
        plt.tight_layout()
        charts_file = os.path.join(self.output_dir, "metrics_charts.png")
        plt.savefig(charts_file)

        print(f"Generated metrics charts at {charts_file}")

    def generate_github_summary(self):
        """Generate a GitHub step summary."""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## CI Metrics Summary\n\n")

            # Test metrics
            test_metrics = self.metrics_data["test_metrics"]
            f.write("### Test Results\n\n")
            f.write(f"Total Tests: **{test_metrics.get('total_tests', 0)}**\n\n")

            f.write("| Result | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")

            passed = test_metrics.get("passed_tests", 0)
            failed = test_metrics.get("failed_tests", 0)
            skipped = test_metrics.get("skipped_tests", 0)
            total = test_metrics.get("total_tests", 0)

            passed_pct = round((passed / total) * 100, 2) if total > 0 else 0
            failed_pct = round((failed / total) * 100, 2) if total > 0 else 0
            skipped_pct = round((skipped / total) * 100, 2) if total > 0 else 0

            f.write(f"| Passed | {passed} | {passed_pct}% |\n")
            f.write(f"| Failed | {failed} | {failed_pct}% |\n")
            f.write(f"| Skipped | {skipped} | {skipped_pct}% |\n\n")

            # Code coverage
            coverage = self.metrics_data["coverage_metrics"].get("overall_coverage", 0)
            f.write("### Code Coverage\n\n")
            f.write(f"Overall coverage: **{coverage}%**\n\n")

            # Platform coverage
            platform_coverage = self.metrics_data["coverage_metrics"].get(
                "platform_coverage", {}
            )
            if platform_coverage:
                f.write("| Platform | Coverage |\n")
                f.write("|----------|----------|\n")

                for platform, cov in platform_coverage.items():
                    f.write(f"| {platform.capitalize()} | {cov}% |\n")

                f.write("\n")

            # Performance highlights
            f.write("### Performance Highlights\n\n")
            duration = self.metrics_data["performance_metrics"].get("total_duration", 0)
            f.write(f"Total CI duration: **{duration:.2f} seconds**\n\n")

            slowest_steps = self.metrics_data["performance_metrics"].get(
                "slowest_steps", []
            )
            if slowest_steps:
                f.write("Slowest steps:\n\n")
                for step in slowest_steps[:3]:  # Top 3 slowest
                    f.write(f"- {step['name']}: {step['duration']:.2f}s\n")
                f.write("\n")

            # Security summary
            security = self.metrics_data["security_metrics"]
            f.write("### Security Summary\n\n")

            total_vulns = security.get("total_vulnerabilities", 0)
            high_vulns = security.get("high_vulnerabilities", 0)
            medium_vulns = security.get("medium_vulnerabilities", 0)
            low_vulns = security.get("low_vulnerabilities", 0)

            if total_vulns > 0:
                f.write("| Severity | Count |\n")
                f.write("|----------|-------|\n")
                f.write(f"| High | {high_vulns} |\n")
                f.write(f"| Medium | {medium_vulns} |\n")
                f.write(f"| Low | {low_vulns} |\n")
                f.write(f"| **Total** | **{total_vulns}** |\n\n")
            else:
                f.write("✅ No security vulnerabilities found\n\n")

            # Platform compatibility
            platform = self.metrics_data["platform_metrics"]
            f.write("### Platform Compatibility\n\n")

            total_issues = platform.get("total_issues", 0)
            errors = platform.get("severity_counts", {}).get("error", 0)
            warnings = platform.get("severity_counts", {}).get("warning", 0)

            if total_issues > 0:
                f.write(f"Found **{total_issues}** platform compatibility issues\n\n")
                f.write(f"- Errors: {errors}\n")
                f.write(f"- Warnings: {warnings}\n\n")
            else:
                f.write("✅ No platform compatibility issues found\n\n")

            f.write("See metrics artifacts for detailed reports and visualizations.\n")

    def run(self):
        """Run the metrics collection and report generation."""
        # Collect various metrics
        self.collect_test_metrics()
        self.collect_coverage_metrics()
        self.collect_performance_metrics()
        self.collect_security_metrics()
        self.collect_platform_metrics()

        # Generate metrics files and visualizations
        self.generate_metrics_files()

        # Generate GitHub step summary
        self.generate_github_summary()

        print(f"CI metrics collection completed. Reports saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect CI metrics and generate reports"
    )
    parser.add_argument(
        "--artifacts-dir",
        default="ci_artifacts",
        help="Directory containing CI artifacts",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/metrics",
        help="Directory to store metrics reports",
    )
    args = parser.parse_args()

    collector = CIMetricsCollector(
        artifacts_dir=args.artifacts_dir, output_dir=args.output_dir
    )
    collector.run()


if __name__ == "__main__":
    main()
