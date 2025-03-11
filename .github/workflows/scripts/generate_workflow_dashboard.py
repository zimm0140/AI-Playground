#!/usr/bin/env python
"""
ComfyUI Workflow Dashboard Generator

This script generates a comprehensive dashboard for ComfyUI workflows,
combining information from various validation tools:
1. Structural validation results
2. Requirements analysis
3. Execution simulation results
4. Version history
5. Compatibility matrix

The dashboard provides a single view of all workflow information,
making it easier to manage and track the workflows.
"""

import os
import json
import argparse
import datetime
from typing import Dict, Any, Set


class ComfyWorkflowDashboard:
    """Generates a comprehensive dashboard for ComfyUI workflows"""

    def __init__(
        self,
        validation_dir: str = "ci_artifacts/workflow_validation",
        requirements_dir: str = "ci_artifacts/workflow_requirements",
        tests_dir: str = "ci_artifacts/workflow_tests",
        versions_dir: str = "ci_artifacts/workflow_versions",
        output_dir: str = "ci_artifacts/workflow_dashboard",
    ):
        self.validation_dir = validation_dir
        self.requirements_dir = requirements_dir
        self.tests_dir = tests_dir
        self.versions_dir = versions_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Data storage
        self.validation_data = None
        self.requirements_data = None
        self.tests_data = None
        self.versions_data = None
        self.dashboard_data = {}

    def load_validation_data(self) -> bool:
        """Load workflow validation results"""
        try:
            validation_file = os.path.join(
                self.validation_dir, "workflow_validation_results.json"
            )
            if os.path.exists(validation_file):
                with open(validation_file, "r", encoding="utf-8") as f:
                    self.validation_data = json.load(f)
                print(
                    f"Loaded validation data for {len(self.validation_data.get('workflows', []))} workflows"
                )
                return True
            else:
                print(f"Validation data file not found: {validation_file}")
                return False
        except Exception as e:
            print(f"Error loading validation data: {e}")
            return False

    def load_requirements_data(self) -> bool:
        """Load workflow requirements analysis"""
        try:
            requirements_file = os.path.join(
                self.requirements_dir, "workflow_requirements_results.json"
            )
            if os.path.exists(requirements_file):
                with open(requirements_file, "r", encoding="utf-8") as f:
                    self.requirements_data = json.load(f)
                print(
                    f"Loaded requirements data for {len(self.requirements_data.get('workflows', []))} workflows"
                )
                return True
            else:
                print(f"Requirements data file not found: {requirements_file}")
                return False
        except Exception as e:
            print(f"Error loading requirements data: {e}")
            return False

    def load_tests_data(self) -> bool:
        """Load workflow execution test results"""
        try:
            tests_file = os.path.join(self.tests_dir, "workflow_test_results.json")
            if os.path.exists(tests_file):
                with open(tests_file, "r", encoding="utf-8") as f:
                    self.tests_data = json.load(f)
                print(
                    f"Loaded test data for {len(self.tests_data.get('workflow_results', []))} workflows"
                )
                return True
            else:
                print(f"Tests data file not found: {tests_file}")
                return False
        except Exception as e:
            print(f"Error loading tests data: {e}")
            return False

    def load_versions_data(self) -> bool:
        """Load workflow version history"""
        try:
            history_file = os.path.join(self.versions_dir, "workflow_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    self.versions_data = json.load(f)
                print(
                    f"Loaded version history for {len(self.versions_data.get('workflows', []))} workflows"
                )
                return True
            else:
                print(f"Version history file not found: {history_file}")
                return False
        except Exception as e:
            print(f"Error loading version history: {e}")
            return False

    def collect_all_workflows(self) -> Set[str]:
        """Collect names of all workflows from all sources"""
        workflows = set()

        # From validation data
        if self.validation_data:
            for workflow in self.validation_data.get("workflows", []):
                if "filename" in workflow:
                    workflows.add(workflow["filename"])

        # From requirements data
        if self.requirements_data:
            for workflow in self.requirements_data.get("workflows", []):
                if "filename" in workflow:
                    workflows.add(workflow["filename"])

        # From test data
        if self.tests_data:
            for workflow in self.tests_data.get("workflow_results", []):
                if "filename" in workflow:
                    workflows.add(workflow["filename"])

        # From version history
        if self.versions_data:
            for workflow in self.versions_data.get("workflows", []):
                if "filename" in workflow:
                    workflows.add(workflow["filename"])

        return workflows

    def integrate_data(self) -> Dict[str, Dict[str, Any]]:
        """Integrate data from all sources into a single dashboard"""
        # Collect all workflows
        all_workflows = self.collect_all_workflows()
        print(f"Found {len(all_workflows)} unique workflows across all sources")

        # Prepare integrated data
        for workflow_name in all_workflows:
            self.dashboard_data[workflow_name] = {
                "name": workflow_name,
                "validation": {"is_valid": False, "issues": []},
                "requirements": {
                    "models": {},
                    "custom_nodes": [],
                    "memory_required": {"min": 0, "recommended": 0},
                },
                "tests": {"passed": False, "issues": [], "execution_time": 0},
                "versions": {
                    "total_versions": 0,
                    "latest_version": None,
                    "latest_changes": [],
                    "has_breaking_changes": False,
                },
                "status": {
                    "overall": "unknown",
                    "validation": "unknown",
                    "requirements": "unknown",
                    "test": "unknown",
                    "version": "unknown",
                },
            }

        # Integrate validation data
        if self.validation_data:
            for workflow in self.validation_data.get("workflows", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    self.dashboard_data[filename]["validation"][
                        "is_valid"
                    ] = workflow.get("is_valid", False)
                    self.dashboard_data[filename]["validation"][
                        "issues"
                    ] = workflow.get("issues", [])

                    # Set validation status
                    if workflow.get("is_valid", False):
                        self.dashboard_data[filename]["status"]["validation"] = "pass"
                    else:
                        # Check if there are errors or just warnings
                        has_errors = any(
                            issue.get("type", "").endswith("_error")
                            for issue in workflow.get("issues", [])
                        )
                        self.dashboard_data[filename]["status"]["validation"] = (
                            "fail" if has_errors else "warning"
                        )

        # Integrate requirements data
        if self.requirements_data:
            for workflow in self.requirements_data.get("workflows", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    self.dashboard_data[filename]["requirements"][
                        "models"
                    ] = workflow.get("models", {})
                    self.dashboard_data[filename]["requirements"][
                        "custom_nodes"
                    ] = workflow.get("custom_nodes", [])
                    self.dashboard_data[filename]["requirements"][
                        "memory_required"
                    ] = workflow.get("memory_required", {"min": 0, "recommended": 0})

                    # Set requirements status based on analysis success
                    self.dashboard_data[filename]["status"]["requirements"] = (
                        "pass" if workflow.get("is_analyzed", False) else "fail"
                    )

        # Integrate test data
        if self.tests_data:
            for workflow in self.tests_data.get("workflow_results", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    self.dashboard_data[filename]["tests"]["passed"] = workflow.get(
                        "passed", False
                    )
                    self.dashboard_data[filename]["tests"]["issues"] = workflow.get(
                        "issues", []
                    )
                    self.dashboard_data[filename]["tests"][
                        "execution_time"
                    ] = workflow.get("execution_time", 0)

                    # Set test status
                    if workflow.get("passed", False):
                        self.dashboard_data[filename]["status"]["test"] = "pass"
                    else:
                        # Check if there are errors or just warnings
                        has_errors = any(
                            "Error:" in issue for issue in workflow.get("issues", [])
                        )
                        self.dashboard_data[filename]["status"]["test"] = (
                            "fail" if has_errors else "warning"
                        )

        # Integrate version history
        if self.versions_data:
            for workflow in self.versions_data.get("workflows", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    versions = workflow.get("versions", [])
                    self.dashboard_data[filename]["versions"]["total_versions"] = len(
                        versions
                    )

                    if versions:
                        # Find latest version (by timestamp)
                        latest = max(versions, key=lambda v: v.get("timestamp", ""))
                        self.dashboard_data[filename]["versions"][
                            "latest_version"
                        ] = latest.get("hash", "")
                        self.dashboard_data[filename]["versions"][
                            "latest_changes"
                        ] = latest.get("changes", [])

                        # Check for breaking changes
                        has_breaking = any(
                            change.startswith("Removed") or "type changed" in change
                            for change in latest.get("changes", [])
                        )
                        self.dashboard_data[filename]["versions"][
                            "has_breaking_changes"
                        ] = has_breaking

                        # Set version status
                        if has_breaking:
                            self.dashboard_data[filename]["status"][
                                "version"
                            ] = "warning"
                        else:
                            self.dashboard_data[filename]["status"]["version"] = "pass"

        # Determine overall status for each workflow
        for filename, data in self.dashboard_data.items():
            statuses = [
                data["status"]["validation"],
                data["status"]["requirements"],
                data["status"]["test"],
                data["status"]["version"],
            ]

            if "fail" in statuses:
                data["status"]["overall"] = "fail"
            elif "warning" in statuses:
                data["status"]["overall"] = "warning"
            elif all(status == "pass" for status in statuses):
                data["status"]["overall"] = "pass"
            else:
                data["status"]["overall"] = "unknown"

        return self.dashboard_data

    def generate_dashboard_json(self) -> str:
        """Generate a JSON file with the integrated dashboard data"""
        json_path = os.path.join(self.output_dir, "workflow_dashboard.json")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generated_at": datetime.datetime.now().isoformat(),
                        "workflows": list(self.dashboard_data.values()),
                    },
                    f,
                    indent=2,
                )

            print(f"Dashboard JSON generated at {json_path}")
            return json_path
        except Exception as e:
            print(f"Error generating dashboard JSON: {e}")
            return ""

    def generate_dashboard_markdown(self) -> str:
        """Generate a markdown dashboard report"""
        md_path = os.path.join(self.output_dir, "workflow_dashboard.md")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# ComfyUI Workflow Dashboard\n\n")
                f.write(
                    f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                # Summary section
                f.write("## Summary\n\n")
                f.write(f"- Total workflows: {len(self.dashboard_data)}\n")

                # Count workflows by status
                status_counts = {
                    "pass": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "pass"
                    ),
                    "warning": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "warning"
                    ),
                    "fail": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "fail"
                    ),
                    "unknown": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "unknown"
                    ),
                }

                f.write(f"- Passing workflows: {status_counts['pass']}\n")
                f.write(f"- Workflows with warnings: {status_counts['warning']}\n")
                f.write(f"- Failing workflows: {status_counts['fail']}\n")
                f.write(f"- Unknown status: {status_counts['unknown']}\n\n")

                # Dashboard table
                f.write("## Workflow Status Dashboard\n\n")
                f.write(
                    "| Workflow | Validation | Test | Memory Req. | Nodes | Custom Nodes | Versions | Status |\n"
                )
                f.write(
                    "|----------|------------|------|-------------|-------|--------------|----------|--------|\n"
                )

                for filename, data in sorted(self.dashboard_data.items()):
                    # Validation status
                    if data["status"]["validation"] == "pass":
                        validation_status = "✅"
                    elif data["status"]["validation"] == "warning":
                        validation_status = "⚠️"
                    elif data["status"]["validation"] == "fail":
                        validation_status = "❌"
                    else:
                        validation_status = "❓"

                    # Test status
                    if data["status"]["test"] == "pass":
                        test_status = "✅"
                    elif data["status"]["test"] == "warning":
                        test_status = "⚠️"
                    elif data["status"]["test"] == "fail":
                        test_status = "❌"
                    else:
                        test_status = "❓"

                    # Memory requirements
                    mem_req = data["requirements"]["memory_required"]["min"]
                    if mem_req == 0:
                        memory_text = "Unknown"
                    else:
                        memory_text = f"{mem_req}GB"

                        if mem_req > 16:
                            memory_text = f"**{memory_text}**"  # Highlight high memory requirements

                    # Node count from version data
                    versions = data["versions"]
                    node_count = "Unknown"
                    if versions["latest_version"]:
                        # Extract from latest changes if available
                        for workflow in self.versions_data.get("workflows", []):
                            if workflow.get("filename") == filename:
                                latest_version = max(
                                    workflow.get("versions", []),
                                    key=lambda v: v.get("timestamp", ""),
                                )
                                if "node_count" in latest_version:
                                    node_count = str(latest_version["node_count"])

                    # Custom nodes
                    custom_nodes = len(data["requirements"]["custom_nodes"])
                    custom_nodes_text = str(custom_nodes) if custom_nodes > 0 else "-"

                    # Version status
                    version_status = data["versions"]["total_versions"]
                    if data["versions"]["has_breaking_changes"]:
                        version_status = f"{version_status} ⚠️"

                    # Overall status
                    if data["status"]["overall"] == "pass":
                        status_text = "✅ Pass"
                    elif data["status"]["overall"] == "warning":
                        status_text = "⚠️ Warning"
                    elif data["status"]["overall"] == "fail":
                        status_text = "❌ Fail"
                    else:
                        status_text = "❓ Unknown"

                    f.write(
                        f"| {filename} | {validation_status} | {test_status} | {memory_text} | {node_count} | {custom_nodes_text} | {version_status} | {status_text} |\n"
                    )

                f.write("\n")

                # Status legend
                f.write("### Status Legend\n\n")
                f.write("- ✅ **Pass**: No issues detected\n")
                f.write(
                    "- ⚠️ **Warning**: Minor issues that might not affect functionality\n"
                )
                f.write("- ❌ **Fail**: Critical issues that need to be addressed\n")
                f.write(
                    "- ❓ **Unknown**: Not enough information to determine status\n\n"
                )

                # Workflow details
                f.write("## Workflow Details\n\n")

                # Group workflows by status for better organization
                grouped_workflows = {
                    "fail": [],
                    "warning": [],
                    "pass": [],
                    "unknown": [],
                }

                for filename, data in self.dashboard_data.items():
                    grouped_workflows[data["status"]["overall"]].append(
                        (filename, data)
                    )

                # First show failing workflows
                if grouped_workflows["fail"]:
                    f.write("### ❌ Failing Workflows\n\n")
                    for filename, data in sorted(grouped_workflows["fail"]):
                        f.write(f"#### {filename}\n\n")

                        # Validation issues
                        if data["validation"]["issues"]:
                            f.write("**Validation Issues:**\n\n")
                            for issue in data["validation"]["issues"]:
                                issue_type = (
                                    issue.get("type", "").replace("_", " ").title()
                                )
                                f.write(f"- {issue_type}: {issue.get('message', '')}\n")
                            f.write("\n")

                        # Test issues
                        if data["tests"]["issues"]:
                            f.write("**Test Issues:**\n\n")
                            for issue in data["tests"]["issues"]:
                                f.write(f"- {issue}\n")
                            f.write("\n")

                        # Requirements
                        f.write("**Requirements:**\n\n")
                        f.write(
                            f"- Memory: Min {data['requirements']['memory_required']['min']}GB, Recommended {data['requirements']['memory_required']['recommended']}GB\n"
                        )
                        if data["requirements"]["custom_nodes"]:
                            f.write(
                                f"- Custom Nodes: {', '.join(data['requirements']['custom_nodes'])}\n"
                            )
                        f.write("\n")

                        # Version information
                        if data["versions"]["latest_changes"]:
                            f.write("**Recent Changes:**\n\n")
                            for change in data["versions"]["latest_changes"]:
                                f.write(f"- {change}\n")
                            f.write("\n")

                # Then show workflows with warnings
                if grouped_workflows["warning"]:
                    f.write("### ⚠️ Workflows with Warnings\n\n")
                    for filename, data in sorted(grouped_workflows["warning"]):
                        f.write(f"#### {filename}\n\n")

                        # Validation issues (warnings only)
                        warnings = [
                            issue
                            for issue in data["validation"]["issues"]
                            if not issue.get("type", "").endswith("_error")
                        ]
                        if warnings:
                            f.write("**Validation Warnings:**\n\n")
                            for issue in warnings:
                                issue_type = (
                                    issue.get("type", "").replace("_", " ").title()
                                )
                                f.write(f"- {issue_type}: {issue.get('message', '')}\n")
                            f.write("\n")

                        # Test warnings
                        warnings = [
                            issue
                            for issue in data["tests"]["issues"]
                            if not issue.startswith("Error:")
                        ]
                        if warnings:
                            f.write("**Test Warnings:**\n\n")
                            for issue in warnings:
                                f.write(f"- {issue}\n")
                            f.write("\n")

                        # Requirements
                        f.write("**Requirements:**\n\n")
                        f.write(
                            f"- Memory: Min {data['requirements']['memory_required']['min']}GB, Recommended {data['requirements']['memory_required']['recommended']}GB\n"
                        )
                        if data["requirements"]["custom_nodes"]:
                            f.write(
                                f"- Custom Nodes: {', '.join(data['requirements']['custom_nodes'])}\n"
                            )
                        f.write("\n")

                        # Version information for breaking changes
                        if data["versions"]["has_breaking_changes"]:
                            f.write("**Breaking Changes:**\n\n")
                            for change in data["versions"]["latest_changes"]:
                                if (
                                    change.startswith("Removed")
                                    or "type changed" in change
                                ):
                                    f.write(f"- ⚠️ {change}\n")
                                else:
                                    f.write(f"- {change}\n")
                            f.write("\n")

                # Passing workflows (summary only)
                if grouped_workflows["pass"]:
                    f.write("### ✅ Passing Workflows\n\n")
                    f.write("The following workflows passed all validations:\n\n")
                    for filename, _ in sorted(grouped_workflows["pass"]):
                        f.write(f"- {filename}\n")
                    f.write("\n")

                # Recommendations section
                f.write("## Recommendations\n\n")

                if grouped_workflows["fail"]:
                    f.write("### High Priority\n\n")
                    f.write("Fix critical issues in the following workflows:\n\n")
                    for filename, data in sorted(grouped_workflows["fail"]):
                        f.write(f"- **{filename}**: ")

                        issues = []
                        if data["validation"]["issues"]:
                            issues.append("Fix validation errors")
                        if [
                            issue
                            for issue in data["tests"]["issues"]
                            if issue.startswith("Error:")
                        ]:
                            issues.append("Address execution errors")

                        f.write(f"{', '.join(issues)}\n")
                    f.write("\n")

                if grouped_workflows["warning"]:
                    f.write("### Medium Priority\n\n")
                    f.write("Address warnings in the following workflows:\n\n")
                    for filename, data in sorted(grouped_workflows["warning"]):
                        f.write(f"- **{filename}**: ")

                        issues = []
                        if data["versions"]["has_breaking_changes"]:
                            issues.append("Review breaking changes")
                        if data["requirements"]["memory_required"]["min"] > 12:
                            issues.append("Consider memory optimization")
                        if [
                            issue
                            for issue in data["tests"]["issues"]
                            if not issue.startswith("Error:")
                        ]:
                            issues.append("Check test warnings")

                        f.write(f"{', '.join(issues)}\n")
                    f.write("\n")

                # Final notes
                f.write("## Notes\n\n")
                f.write(
                    "1. This dashboard is automatically generated by the CI workflow.\n"
                )
                f.write(
                    "2. Workflows with high memory requirements may not run on all systems.\n"
                )
                f.write(
                    "3. Breaking changes in workflows may affect compatibility with older versions.\n"
                )
                f.write(
                    "4. Custom nodes required by workflows must be installed separately.\n\n"
                )

                f.write("---\n")
                f.write("*Generated by the ComfyUI Workflow Dashboard Generator*\n")

            print(f"Dashboard markdown generated at {md_path}")
            return md_path
        except Exception as e:
            print(f"Error generating dashboard markdown: {e}")
            return ""

    def generate_github_summary(self) -> None:
        """Generate a GitHub step summary with dashboard information"""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        try:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write("## ComfyUI Workflow Dashboard\n\n")

                # Count workflows by status
                status_counts = {
                    "pass": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "pass"
                    ),
                    "warning": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "warning"
                    ),
                    "fail": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "fail"
                    ),
                    "unknown": sum(
                        1
                        for data in self.dashboard_data.values()
                        if data["status"]["overall"] == "unknown"
                    ),
                }

                # Status indicators based on counts
                if status_counts["fail"] > 0:
                    f.write(
                        f"❌ **{status_counts['fail']} workflow(s) have critical issues**\n\n"
                    )
                elif status_counts["warning"] > 0:
                    f.write(
                        f"⚠️ **{status_counts['warning']} workflow(s) have warnings**\n\n"
                    )
                else:
                    f.write("✅ **All workflows are passing**\n\n")

                # Summary table
                f.write("| Status | Count |\n")
                f.write("|--------|-------|\n")
                f.write(f"| ✅ Pass | {status_counts['pass']} |\n")
                f.write(f"| ⚠️ Warning | {status_counts['warning']} |\n")
                f.write(f"| ❌ Fail | {status_counts['fail']} |\n")
                f.write(f"| ❓ Unknown | {status_counts['unknown']} |\n\n")

                # Failing workflows table
                failing_workflows = [
                    (filename, data)
                    for filename, data in self.dashboard_data.items()
                    if data["status"]["overall"] == "fail"
                ]

                if failing_workflows:
                    f.write("### Failing Workflows\n\n")
                    f.write("| Workflow | Validation | Test | Issues |\n")
                    f.write("|----------|------------|------|--------|\n")

                    for filename, data in sorted(failing_workflows):
                        # Validation status
                        if data["status"]["validation"] == "pass":
                            validation_status = "✅"
                        elif data["status"]["validation"] == "warning":
                            validation_status = "⚠️"
                        elif data["status"]["validation"] == "fail":
                            validation_status = "❌"
                        else:
                            validation_status = "❓"

                        # Test status
                        if data["status"]["test"] == "pass":
                            test_status = "✅"
                        elif data["status"]["test"] == "warning":
                            test_status = "⚠️"
                        elif data["status"]["test"] == "fail":
                            test_status = "❌"
                        else:
                            test_status = "❓"

                        # Get most critical issues
                        critical_issues = []

                        # Add validation errors
                        for issue in data["validation"]["issues"]:
                            if issue.get("type", "").endswith("_error"):
                                issue_msg = issue.get("message", "")
                                if len(issue_msg) > 50:
                                    issue_msg = issue_msg[:47] + "..."
                                critical_issues.append(issue_msg)

                        # Add test errors
                        for issue in data["tests"]["issues"]:
                            if issue.startswith("Error:"):
                                issue_msg = issue
                                if len(issue_msg) > 50:
                                    issue_msg = issue_msg[:47] + "..."
                                critical_issues.append(issue_msg)

                        # Format issues for table
                        issues_text = ", ".join(
                            critical_issues[:2]
                        )  # Show only first 2 issues
                        if len(critical_issues) > 2:
                            issues_text += f", +{len(critical_issues) - 2} more"

                        f.write(
                            f"| {filename} | {validation_status} | {test_status} | {issues_text} |\n"
                        )

                    f.write("\n")

                f.write("See workflow dashboard artifact for complete details.\n")
        except Exception as e:
            print(f"Error generating GitHub summary: {e}")

    def run(self) -> None:
        """Run the dashboard generation process"""
        print("Generating ComfyUI workflow dashboard...")

        # Load data from all sources
        self.load_validation_data()
        self.load_requirements_data()
        self.load_tests_data()
        self.load_versions_data()

        # Integrate data into a single dashboard
        self.integrate_data()

        # Generate dashboard outputs
        self.generate_dashboard_json()
        self.generate_dashboard_markdown()
        self.generate_github_summary()

        print("Dashboard generation complete")


def main():
    parser = argparse.ArgumentParser(description="Generate workflow dashboard markdown and JSON")
    parser.add_argument(
        "--validation-dir",
        help="Directory containing workflow validation results",
        default="ci_artifacts/workflow_validation",
    )
    parser.add_argument(
        "--requirements-dir",
        help="Directory containing workflow requirements analysis",
        default="ci_artifacts/workflow_requirements",
    )
    parser.add_argument(
        "--tests-dir",
        help="Directory containing workflow test results",
        default="ci_artifacts/workflow_tests",
    )
    parser.add_argument(
        "--versions-dir",
        help="Directory containing workflow version history",
        default="ci_artifacts/workflow_versions",
    )
    parser.add_argument(
        "--workflows-dir",
        help="Directory containing workflow JSON files",
        default="WebUI/external/workflows",
    )
    parser.add_argument(
        "--components-dir",
        help="Directory containing component JSON files",
        default="WebUI/external/components",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write dashboard files",
        default="ci_artifacts/workflow_dashboard",
    )
    args = parser.parse_args()

    dashboard = ComfyWorkflowDashboard(
        validation_dir=args.validation_dir,
        requirements_dir=args.requirements_dir,
        tests_dir=args.tests_dir,
        versions_dir=args.versions_dir,
        output_dir=args.output_dir,
    )

    dashboard.run()


if __name__ == "__main__":
    main()
