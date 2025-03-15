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

import argparse
import datetime
import json
import os
import traceback
from typing import Any


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
                self.validation_dir,
                "workflow_validation_results.json",
            )
            if os.path.exists(validation_file):
                with open(validation_file, encoding="utf-8") as f:
                    self.validation_data = json.load(f)
                print(
                    f"Loaded validation data for {len(self.validation_data.get('workflows', []))} workflows",
                )
                return True
            print(f"Validation data file not found: {validation_file}")
            return False
        except Exception as e:
            print(f"Error loading validation data: {e}")
            return False

    def load_requirements_data(self) -> bool:
        """Load workflow requirements analysis"""
        try:
            requirements_file = os.path.join(
                self.requirements_dir,
                "workflow_requirements_results.json",
            )
            if os.path.exists(requirements_file):
                with open(requirements_file, encoding="utf-8") as f:
                    self.requirements_data = json.load(f)
                print(
                    f"Loaded requirements data for {len(self.requirements_data.get('workflows', []))} workflows",
                )
                return True
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
                with open(tests_file, encoding="utf-8") as f:
                    self.tests_data = json.load(f)
                print(
                    f"Loaded test data for {len(self.tests_data.get('workflow_results', []))} workflows",
                )
                return True
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
                with open(history_file, encoding="utf-8") as f:
                    self.versions_data = json.load(f)
                print(
                    f"Loaded version history for {len(self.versions_data.get('workflows', []))} workflows",
                )
                return True
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

    def integrate_data(self) -> Dict[str, dict[str, Any]]:
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
                    self.dashboard_data[filename]["validation"]["is_valid"] = workflow.get("is_valid", False)
                    self.dashboard_data[filename]["validation"]["issues"] = workflow.get("issues", [])

                    # Set validation status
                    if workflow.get("is_valid", False):
                        self.dashboard_data[filename]["status"]["validation"] = "pass"
                    else:
                        # Check if there are errors or just warnings
                        has_errors = any(
                            issue.get("type", "").endswith("_error") for issue in workflow.get("issues", [])
                        )
                        self.dashboard_data[filename]["status"]["validation"] = "fail" if has_errors else "warning"

        # Integrate requirements data
        if self.requirements_data:
            for workflow in self.requirements_data.get("workflows", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    self.dashboard_data[filename]["requirements"]["models"] = workflow.get("models", {})
                    self.dashboard_data[filename]["requirements"]["custom_nodes"] = workflow.get("custom_nodes", [])
                    self.dashboard_data[filename]["requirements"]["memory_required"] = workflow.get(
                        "memory_required", {"min": 0, "recommended": 0},
                    )

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
                        "passed",
                        False,
                    )
                    self.dashboard_data[filename]["tests"]["issues"] = workflow.get(
                        "issues",
                        [],
                    )
                    self.dashboard_data[filename]["tests"]["execution_time"] = workflow.get("execution_time", 0)

                    # Set test status
                    if workflow.get("passed", False):
                        self.dashboard_data[filename]["status"]["test"] = "pass"
                    else:
                        # Check if there are errors or just warnings
                        has_errors = any("Error:" in issue for issue in workflow.get("issues", []))
                        self.dashboard_data[filename]["status"]["test"] = "fail" if has_errors else "warning"

        # Integrate version history
        if self.versions_data:
            for workflow in self.versions_data.get("workflows", []):
                if "filename" not in workflow:
                    continue

                filename = workflow["filename"]
                if filename in self.dashboard_data:
                    versions = workflow.get("versions", [])
                    self.dashboard_data[filename]["versions"]["total_versions"] = len(
                        versions,
                    )

                    if versions:
                        # Find latest version (by timestamp)
                        latest = max(versions, key=lambda v: v.get("timestamp", ""))
                        self.dashboard_data[filename]["versions"]["latest_version"] = latest.get("hash", "")
                        self.dashboard_data[filename]["versions"]["latest_changes"] = latest.get("changes", [])

                        # Check for breaking changes
                        has_breaking = any(
                            change.startswith("Removed") or "type changed" in change
                            for change in latest.get("changes", [])
                        )
                        self.dashboard_data[filename]["versions"]["has_breaking_changes"] = has_breaking

                        # Set version status
                        if has_breaking:
                            self.dashboard_data[filename]["status"]["version"] = "warning"
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

    def _write_summary_section(self, file_handle):
        """Write the summary section to the markdown file."""
        file_handle.write("## Summary\n\n")
        file_handle.write(f"- Total workflows: {len(self.dashboard_data)}\n")

        # Count workflows by status
        status_counts = {
            "pass": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "pass"),
            "warning": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "warning"),
            "fail": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "fail"),
            "unknown": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "unknown"),
        }

        file_handle.write(f"- Passing workflows: {status_counts['pass']}\n")
        file_handle.write(f"- Workflows with warnings: {status_counts['warning']}\n")
        file_handle.write(f"- Failing workflows: {status_counts['fail']}\n")
        file_handle.write(f"- Unknown status: {status_counts['unknown']}\n\n")

    def _get_status_emoji(self, status):
        """Convert status string to emoji representation."""
        status_mapping = {
            "pass": "✅",
            "warning": "⚠️",
            "fail": "❌",
            "unknown": "❓",
        }
        return status_mapping.get(status, "❓")

    def _format_memory_text(self, mem_req):
        """Format memory requirement text with highlighting for high values."""
        if mem_req == 0:
            return "Unknown"

        memory_text = f"{mem_req}GB"
        if mem_req > 16:
            memory_text = f"**{memory_text}**"  # Highlight high memory requirements

        return memory_text

    def _get_node_count(self, filename, versions):
        """Extract node count from version data."""
        if not versions["latest_version"]:
            return "Unknown"

        # Extract from latest changes if available
        for workflow in self.versions_data.get("workflows", []):
            if workflow.get("filename") == filename:
                latest_version = max(
                    workflow.get("versions", []),
                    key=lambda v: v.get("timestamp", ""),
                )
                if "node_count" in latest_version:
                    return str(latest_version["node_count"])

        return "Unknown"

    def _format_version_status(self, version_data):
        """Format version status text with warning if breaking changes."""
        version_status = version_data["total_versions"]
        if version_data["has_breaking_changes"]:
            version_status = f"{version_status} ⚠️"
        return version_status

    def _write_table_header(self, file_handle):
        """Write the markdown table header."""
        file_handle.write("## Workflow Status Dashboard\n\n")
        file_handle.write(
            "| Workflow | Validation | Test | Memory Req. | Nodes | Custom Nodes | Versions | Status |\n",
        )
        file_handle.write(
            "|----------|------------|------|-------------|-------|--------------|----------|--------|\n",
        )

    def _write_workflow_row(self, file_handle, filename, data):
        """Write a single workflow row to the markdown table."""
        # Get status indicators
        validation_status = self._get_status_emoji(data["status"]["validation"])
        test_status = self._get_status_emoji(data["status"]["test"])

        # Format memory requirements
        mem_req = data["requirements"]["memory_required"]["min"]
        memory_text = self._format_memory_text(mem_req)

        # Get node counts
        node_count = self._get_node_count(filename, data["versions"])

        # Format custom nodes
        custom_nodes = len(data["requirements"]["custom_nodes"])
        custom_nodes_text = str(custom_nodes) if custom_nodes > 0 else "-"

        # Format version status
        version_status = self._format_version_status(data["versions"])

        # Get overall status
        overall_status = self._get_status_emoji(data["status"]["overall"])

        # Write the table row
        file_handle.write(
            f"| [{filename}]({data['repo_link']}) | {validation_status} | {test_status} | {memory_text} | {node_count} | {custom_nodes_text} | {version_status} | {overall_status} |\n",
        )

    def generate_dashboard_markdown(self) -> str:
        """Generate a markdown dashboard report"""
        md_path = os.path.join(self.output_dir, "workflow_dashboard.md")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                # Write header
                f.write("# ComfyUI Workflow Dashboard\n\n")
                f.write(
                    f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                )

                # Write summary section
                self._write_summary_section(f)

                # Write table header
                self._write_table_header(f)

                # Write workflow rows
                for filename, data in sorted(self.dashboard_data.items()):
                    self._write_workflow_row(f, filename, data)

                # Add legend
                f.write("\n## Status Legend\n\n")
                f.write("- ✅ Pass - All checks passing\n")
                f.write("- ⚠️ Warning - Issues detected but workflow is usable\n")
                f.write("- ❌ Fail - Workflow has critical issues\n")
                f.write("- ❓ Unknown - Status cannot be determined\n")

            print(f"Generated markdown dashboard at {md_path}")
            return md_path
        except Exception as e:
            print(f"Error generating dashboard markdown: {e}")
            traceback.print_exc()
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
                    "pass": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "pass"),
                    "warning": sum(
                        1 for data in self.dashboard_data.values() if data["status"]["overall"] == "warning"
                    ),
                    "fail": sum(1 for data in self.dashboard_data.values() if data["status"]["overall"] == "fail"),
                    "unknown": sum(
                        1 for data in self.dashboard_data.values() if data["status"]["overall"] == "unknown"
                    ),
                }

                # Status indicators based on counts
                if status_counts["fail"] > 0:
                    f.write(
                        f"❌ **{status_counts['fail']} workflow(s) have critical issues**\n\n",
                    )
                elif status_counts["warning"] > 0:
                    f.write(
                        f"⚠️ **{status_counts['warning']} workflow(s) have warnings**\n\n",
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
                            critical_issues[:2],
                        )  # Show only first 2 issues
                        if len(critical_issues) > 2:
                            issues_text += f", +{len(critical_issues) - 2} more"

                        f.write(
                            f"| {filename} | {validation_status} | {test_status} | {issues_text} |\n",
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