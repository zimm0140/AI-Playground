#!/usr/bin/env python
"""
PR Comment Generator for ComfyUI Workflow Changes

This script:
1. Generates a detailed comment for PRs that modify ComfyUI workflow files
2. Summarizes validation results, simulation results, and detected changes
3. Provides actionable feedback to PR authors on workflow quality and compatibility
4. Highlights potential issues and improvement recommendations

The comment helps reviewers understand the impact of workflow changes and
helps PR authors address problems before merging.
"""

import argparse
import json
import os
import sys
from typing import Any


class PRCommentGenerator:
    """Generates PR comments for ComfyUI workflow changes"""

    def __init__(
        self,
        validation_dir: str = "ci_artifacts/workflow_validation",
        requirements_dir: str = "ci_artifacts/workflow_requirements",
        tests_dir: str = "ci_artifacts/workflow_tests",
        simulation_dir: str = "ci_artifacts/workflow_simulation",
        versions_dir: str = "ci_artifacts/workflow_versions",
        dashboard_dir: str = "ci_artifacts/workflow_dashboard",
        changed_files: list[str] = None,
        output_file: str = "workflow_pr_comment.md",
    ):
        self.validation_dir = validation_dir
        self.requirements_dir = requirements_dir
        self.tests_dir = tests_dir
        self.simulation_dir = simulation_dir
        self.versions_dir = versions_dir
        self.dashboard_dir = dashboard_dir
        self.changed_files = changed_files or []
        self.output_file = output_file

        # Only keep ComfyUI workflow files
        self.changed_workflows = [
            f
            for f in self.changed_files
            if f.startswith("WebUI/external/workflows/") and f.endswith(".json")
        ]

        # Data storage
        self.validation_data = None
        self.requirements_data = None
        self.tests_data = None
        self.simulation_data = None
        self.versions_data = None
        self.dashboard_data = None

    def load_validation_data(self) -> bool:
        """Load workflow validation results"""
        try:
            validation_file = os.path.join(
                self.validation_dir, "workflow_validation_results.json"
            )
            if os.path.exists(validation_file):
                with open(validation_file, encoding="utf-8") as f:
                    self.validation_data = json.load(f)
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
                with open(requirements_file, encoding="utf-8") as f:
                    self.requirements_data = json.load(f)
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
                with open(tests_file, encoding="utf-8") as f:
                    self.tests_data = json.load(f)
                return True
            else:
                print(f"Tests data file not found: {tests_file}")
                return False
        except Exception as e:
            print(f"Error loading tests data: {e}")
            return False

    def load_simulation_data(self) -> bool:
        """Load workflow simulation results"""
        try:
            simulation_file = os.path.join(
                self.simulation_dir, "workflow_simulation_results.json"
            )
            if os.path.exists(simulation_file):
                with open(simulation_file, encoding="utf-8") as f:
                    self.simulation_data = json.load(f)
                return True
            else:
                print(f"Simulation data file not found: {simulation_file}")
                return False
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return False

    def load_versions_data(self) -> bool:
        """Load workflow version history"""
        try:
            history_file = os.path.join(self.versions_dir, "workflow_history.json")
            if os.path.exists(history_file):
                with open(history_file, encoding="utf-8") as f:
                    self.versions_data = json.load(f)
                return True
            else:
                print(f"Version history file not found: {history_file}")
                return False
        except Exception as e:
            print(f"Error loading version history: {e}")
            return False

    def get_workflow_filenames(self) -> set[str]:
        """Extract workflow filenames from changed files paths"""
        return {os.path.basename(file) for file in self.changed_workflows}

    def get_workflow_validation_status(self, filename: str) -> dict[str, Any]:
        """Get validation status for a specific workflow"""
        if not self.validation_data:
            return {
                "is_valid": False,
                "status": "unknown",
                "issues": ["No validation data available"],
            }

        for workflow in self.validation_data.get("workflows", []):
            if workflow.get("filename") == filename:
                status = "pass" if workflow.get("is_valid", False) else "fail"
                return {
                    "is_valid": workflow.get("is_valid", False),
                    "status": status,
                    "issues": workflow.get("issues", []),
                }

        return {
            "is_valid": False,
            "status": "unknown",
            "issues": ["Workflow not found in validation results"],
        }

    def get_workflow_requirements(self, filename: str) -> dict[str, Any]:
        """Get requirements analysis for a specific workflow"""
        if not self.requirements_data:
            return {
                "is_analyzed": False,
                "status": "unknown",
                "memory_required": {"min": 0, "recommended": 0},
            }

        for workflow in self.requirements_data.get("workflows", []):
            if workflow.get("filename") == filename:
                return {
                    "is_analyzed": workflow.get("is_analyzed", False),
                    "status": "pass" if workflow.get("is_analyzed", False) else "fail",
                    "memory_required": workflow.get(
                        "memory_required", {"min": 0, "recommended": 0}
                    ),
                    "models": workflow.get("models", {}),
                    "custom_nodes": workflow.get("custom_nodes", []),
                }

        return {
            "is_analyzed": False,
            "status": "unknown",
            "memory_required": {"min": 0, "recommended": 0},
        }

    def get_workflow_test_status(self, filename: str) -> dict[str, Any]:
        """Get test execution status for a specific workflow"""
        if not self.tests_data:
            return {
                "passed": False,
                "status": "unknown",
                "issues": ["No test data available"],
            }

        for workflow in self.tests_data.get("workflow_results", []):
            if workflow.get("filename") == filename:
                return {
                    "passed": workflow.get("passed", False),
                    "status": "pass" if workflow.get("passed", False) else "fail",
                    "issues": workflow.get("issues", []),
                }

        return {
            "passed": False,
            "status": "unknown",
            "issues": ["Workflow not found in test results"],
        }

    def get_workflow_simulation_status(self, filename: str) -> dict[str, Any]:
        """Get simulation status for a specific workflow"""
        if not self.simulation_data:
            return {
                "success": False,
                "status": "unknown",
                "errors": ["No simulation data available"],
            }

        for workflow in self.simulation_data.get("workflows", []):
            if workflow.get("filename") == filename:
                return {
                    "success": workflow.get("success", False),
                    "status": "pass" if workflow.get("success", False) else "fail",
                    "errors": workflow.get("errors", []),
                    "node_results": workflow.get("node_results", {}),
                }

        return {
            "success": False,
            "status": "unknown",
            "errors": ["Workflow not found in simulation results"],
        }

    def get_workflow_changes(self, filename: str) -> dict[str, Any]:
        """Get version changes for a specific workflow"""
        if not self.versions_data:
            return {"has_history": False, "latest_changes": []}

        for workflow in self.versions_data.get("workflows", []):
            if workflow.get("filename") == filename:
                versions = workflow.get("versions", [])
                if not versions:
                    return {"has_history": True, "latest_changes": ["Initial version"]}

                # Find latest version by timestamp
                latest = max(versions, key=lambda v: v.get("timestamp", ""))

                # Check for breaking changes
                has_breaking = any(
                    change.startswith("Removed") or "type changed" in change
                    for change in latest.get("changes", [])
                )

                return {
                    "has_history": True,
                    "latest_changes": latest.get("changes", []),
                    "latest_version": latest.get("hash", ""),
                    "has_breaking_changes": has_breaking,
                    "node_count": latest.get("node_count", 0),
                    "link_count": latest.get("link_count", 0),
                }

        return {"has_history": False, "latest_changes": []}

    def generate_workflow_summary(self, filename: str) -> dict[str, Any]:
        """Generate a complete summary for a workflow"""
        validation = self.get_workflow_validation_status(filename)
        requirements = self.get_workflow_requirements(filename)
        test = self.get_workflow_test_status(filename)
        simulation = self.get_workflow_simulation_status(filename)
        changes = self.get_workflow_changes(filename)

        # Determine overall status
        statuses = [
            validation["status"],
            requirements["status"],
            test["status"],
            simulation["status"],
        ]

        if "fail" in statuses:
            overall_status = "fail"
        elif "warning" in statuses or changes.get("has_breaking_changes", False):
            overall_status = "warning"
        elif all(status == "pass" for status in statuses):
            overall_status = "pass"
        else:
            overall_status = "unknown"

        return {
            "filename": filename,
            "validation": validation,
            "requirements": requirements,
            "test": test,
            "simulation": simulation,
            "changes": changes,
            "status": overall_status,
        }

    def generate_comment(self) -> str:
        """Generate a PR comment for workflow changes"""
        if not self.changed_workflows:
            print("No workflow changes detected")
            return "## ComfyUI Workflow Changes\n\nNo workflow files were modified in this PR."

        # Load all data sources
        self.load_validation_data()
        self.load_requirements_data()
        self.load_tests_data()
        self.load_simulation_data()
        self.load_versions_data()

        # Get workflow filenames from changed files
        filenames = self.get_workflow_filenames()
        print(f"Generating comment for {len(filenames)} changed workflows")

        # Generate summary for each workflow
        workflow_summaries = {}
        for filename in filenames:
            workflow_summaries[filename] = self.generate_workflow_summary(filename)

        # Generate the comment markdown
        comment = "## ComfyUI Workflow Validation Results\n\n"

        # Overall status summary
        status_counts = {
            "pass": sum(
                1 for w in workflow_summaries.values() if w["status"] == "pass"
            ),
            "warning": sum(
                1 for w in workflow_summaries.values() if w["status"] == "warning"
            ),
            "fail": sum(
                1 for w in workflow_summaries.values() if w["status"] == "fail"
            ),
            "unknown": sum(
                1 for w in workflow_summaries.values() if w["status"] == "unknown"
            ),
        }

        # Add appropriate status icon
        if status_counts["fail"] > 0:
            comment += f"❌ **{status_counts['fail']} of {len(filenames)} workflows have issues that need to be fixed**\n\n"
        elif status_counts["warning"] > 0:
            comment += f"⚠️ **{status_counts['warning']} of {len(filenames)} workflows have warnings to review**\n\n"
        else:
            comment += "✅ **All workflow changes look good!**\n\n"

        # Add workflow summary table
        comment += "### Changed Workflows Status\n\n"
        comment += "| Workflow | Validation | Test | Simulation | Memory | Status |\n"
        comment += "|----------|------------|------|------------|--------|--------|\n"

        for filename, summary in sorted(workflow_summaries.items()):
            # Validation status icon
            if summary["validation"]["status"] == "pass":
                validation_icon = "✅"
            elif summary["validation"]["status"] == "warning":
                validation_icon = "⚠️"
            elif summary["validation"]["status"] == "fail":
                validation_icon = "❌"
            else:
                validation_icon = "❓"

            # Test status icon
            if summary["test"]["status"] == "pass":
                test_icon = "✅"
            elif summary["test"]["status"] == "warning":
                test_icon = "⚠️"
            elif summary["test"]["status"] == "fail":
                test_icon = "❌"
            else:
                test_icon = "❓"

            # Simulation status icon
            if summary["simulation"]["status"] == "pass":
                simulation_icon = "✅"
            elif summary["simulation"]["status"] == "warning":
                simulation_icon = "⚠️"
            elif summary["simulation"]["status"] == "fail":
                simulation_icon = "❌"
            else:
                simulation_icon = "❓"

            # Memory requirements
            mem_req = summary["requirements"]["memory_required"]["min"]
            if mem_req == 0:
                memory_text = "Unknown"
            else:
                memory_text = f"{mem_req}GB"

                # Highlight high memory requirements
                if mem_req > 16:
                    memory_text += " ⚠️"

            # Overall status
            if summary["status"] == "pass":
                status_text = "✅ Pass"
            elif summary["status"] == "warning":
                status_text = "⚠️ Warning"
            elif summary["status"] == "fail":
                status_text = "❌ Fix Required"
            else:
                status_text = "❓ Unknown"

            # Add row to table
            comment += f"| {filename} | {validation_icon} | {test_icon} | {simulation_icon} | {memory_text} | {status_text} |\n"

        # Add details for workflows with issues
        failing_workflows = [
            w for w in workflow_summaries.values() if w["status"] in ["fail", "warning"]
        ]
        if failing_workflows:
            comment += "\n### Issues Requiring Attention\n\n"

            for workflow in failing_workflows:
                comment += f"#### {workflow['filename']}\n\n"

                # List validation issues
                if workflow["validation"]["status"] == "fail":
                    comment += "**Validation Issues:**\n\n"
                    for issue in workflow["validation"]["issues"]:
                        issue_type = issue.get("type", "").replace("_", " ").title()
                        comment += f"- {issue_type}: {issue.get('message', '')}\n"
                    comment += "\n"

                # List test issues
                if workflow["test"]["status"] == "fail":
                    comment += "**Test Issues:**\n\n"
                    for issue in workflow["test"]["issues"]:
                        comment += f"- {issue}\n"
                    comment += "\n"

                # List simulation errors
                if workflow["simulation"]["status"] == "fail":
                    comment += "**Simulation Errors:**\n\n"
                    for error in workflow["simulation"]["errors"][
                        :5
                    ]:  # Show at most 5 errors
                        comment += f"- {error}\n"

                    if len(workflow["simulation"]["errors"]) > 5:
                        comment += f"- ... and {len(workflow['simulation']['errors']) - 5} more errors\n"
                    comment += "\n"

                # List breaking changes
                if workflow["changes"].get("has_breaking_changes", False):
                    comment += "**Breaking Changes Detected:**\n\n"
                    for change in workflow["changes"]["latest_changes"]:
                        if change.startswith("Removed") or "type changed" in change:
                            comment += f"- ⚠️ {change}\n"
                        else:
                            comment += f"- {change}\n"
                    comment += "\n"

                # Show requirements
                comment += "**Resource Requirements:**\n\n"
                comment += f"- Minimum memory: {workflow['requirements']['memory_required']['min']}GB\n"
                comment += f"- Recommended memory: {workflow['requirements']['memory_required']['recommended']}GB\n"

                if workflow["requirements"]["custom_nodes"]:
                    comment += f"- Custom nodes: {', '.join(workflow['requirements']['custom_nodes'])}\n"

                # Show node and link counts
                if workflow["changes"].get("node_count", 0) > 0:
                    comment += f"- Total nodes: {workflow['changes']['node_count']}\n"
                    comment += (
                        f"- Total connections: {workflow['changes']['link_count']}\n"
                    )

                comment += "\n"

                # Add recommendations based on issues
                recommendations = []

                if workflow["validation"]["status"] == "fail":
                    recommendations.append("Fix JSON structure and validation issues")

                if workflow["test"]["status"] == "fail":
                    recommendations.append(
                        "Ensure workflow graph is properly connected without circular dependencies"
                    )

                if workflow["simulation"]["status"] == "fail":
                    recommendations.append("Review node implementation compatibility")

                if workflow["changes"].get("has_breaking_changes", False):
                    recommendations.append(
                        "Consider impact of breaking changes on existing users"
                    )

                if workflow["requirements"]["memory_required"]["min"] > 16:
                    recommendations.append(
                        "Optimize for lower memory usage if possible"
                    )

                if recommendations:
                    comment += "**Recommendations:**\n\n"
                    for recommendation in recommendations:
                        comment += f"- {recommendation}\n"
                    comment += "\n"

        # Add notes about workflow validation process
        comment += "\n### About Workflow Validation\n\n"
        comment += "This comment was automatically generated by the CI workflow validation process. It analyzes:\n\n"
        comment += (
            "1. **Structural Validation**: Checks JSON structure and node connections\n"
        )
        comment += "2. **Execution Testing**: Verifies workflow execution logic\n"
        comment += "3. **Simulation**: Tests execution with simulated models (no GPU required)\n"
        comment += (
            "4. **Resource Analysis**: Estimates memory and dependency requirements\n"
        )
        comment += "5. **Version Tracking**: Detects breaking changes from previous versions\n\n"
        comment += "For more details, see the CI artifacts from this PR build.\n"

        # Save comment to file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(comment)

        print(f"Comment saved to {self.output_file}")
        return comment


def main():
    parser = argparse.ArgumentParser(
        description="Generate PR comment for ComfyUI workflow changes"
    )
    parser.add_argument(
        "--validation-dir",
        default="ci_artifacts/workflow_validation",
        help="Directory containing validation results",
    )
    parser.add_argument(
        "--requirements-dir",
        default="ci_artifacts/workflow_requirements",
        help="Directory containing requirements analysis",
    )
    parser.add_argument(
        "--tests-dir",
        default="ci_artifacts/workflow_tests",
        help="Directory containing test results",
    )
    parser.add_argument(
        "--simulation-dir",
        default="ci_artifacts/workflow_simulation",
        help="Directory containing simulation results",
    )
    parser.add_argument(
        "--versions-dir",
        default="ci_artifacts/workflow_versions",
        help="Directory containing version history",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="ci_artifacts/workflow_dashboard",
        help="Directory containing dashboard",
    )
    parser.add_argument(
        "--changed-files", nargs="+", help="List of changed files in the PR"
    )
    parser.add_argument(
        "--output-file",
        default="workflow_pr_comment.md",
        help="Output file for PR comment",
    )
    args = parser.parse_args()

    # If no changed files are provided, try to get them from environment variable
    changed_files = args.changed_files
    if not changed_files and os.environ.get("CHANGED_FILES"):
        changed_files = os.environ.get("CHANGED_FILES").split(",")

    if not changed_files:
        print(
            "No changed files provided. Please provide --changed-files or set CHANGED_FILES environment variable"
        )
        sys.exit(1)

    generator = PRCommentGenerator(
        validation_dir=args.validation_dir,
        requirements_dir=args.requirements_dir,
        tests_dir=args.tests_dir,
        simulation_dir=args.simulation_dir,
        versions_dir=args.versions_dir,
        dashboard_dir=args.dashboard_dir,
        changed_files=changed_files,
        output_file=args.output_file,
    )

    generator.generate_comment()


if __name__ == "__main__":
    main()
