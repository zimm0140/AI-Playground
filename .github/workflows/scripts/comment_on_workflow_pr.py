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
from typing import Optional, Any, Dict, List, Set, cast
import datetime
import re


class PRCommentGenerator:
    """Generates PR comments for ComfyUI workflow changes"""

    def __init__(self, validation_dir: str = "ci_artifacts/workflow_validation", requirements_dir: str = "ci_artifacts/workflow_requirements", tests_dir: str = "ci_artifacts/workflow_tests", simulation_dir: str = "ci_artifacts/workflow_simulation", versions_dir: str = "ci_artifacts/workflow_versions", dashboard_dir: str = "ci_artifacts/workflow_dashboard", changed_files: Optional[List[str]] = None, output_file: str = "workflow_pr_comment.md"):
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
            if f.startswith("WebUI/external/workflows/") and f.suffix == '.json')
        ]

        # Data storage
        self.validation_data = None
        self.requirements_data = None
        self.tests_data = None
        self.simulation_data = None
        self.versions_data = None
        self.dashboard_data = None

    def load_validation_data(self) -> Dict[str, Any]:
        """Load validation data from JSON files.
        
        Returns:
            Dictionary of validation results data by workflow name.
        """
        validation_data: Dict[str, Any] = {}

        # Check if validation directory exists
        if not os.path.exists(self.validation_dir):
            return validation_data
        
        # Load validation results for all workflows
        for file in os.listdir(self.validation_dir):
            if file.suffix == '.json'):
                file_path = os.path.join(self.validation_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        workflow_name = os.path.basename(file).replace(".json", "")
                        validation_data[workflow_name] = data
                except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                    print(f"Error loading validation data from {file}: {e}")
        
        return validation_data

    def load_requirements_data(self) -> Dict[str, Any]:
        """Load requirements data from JSON files.
        
        Returns:
            Dictionary of requirements results data by workflow name.
        """
        requirements_data: Dict[str, Any] = {}

        # Check if requirements directory exists
        if not os.path.exists(self.requirements_dir):
            return requirements_data
        
        # Load requirements results for all workflows
        for file in os.listdir(self.requirements_dir):
            if file.suffix == '.json'):
                file_path = os.path.join(self.requirements_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        workflow_name = os.path.basename(file).replace(".json", "")
                        requirements_data[workflow_name] = data
                except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                    print(f"Error loading requirements data from {file}: {e}")
        
        return requirements_data

    def load_tests_data(self) -> Dict[str, Any]:
        """Load test data from JSON files.
        
        Returns:
            Dictionary of test results data by workflow name.
        """
        tests_data: Dict[str, Any] = {}

        # Check if tests directory exists
        if not os.path.exists(self.tests_dir):
            return tests_data
        
        # Load test results for all workflows
        for file in os.listdir(self.tests_dir):
            if file.suffix == '.json'):
                file_path = os.path.join(self.tests_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        workflow_name = os.path.basename(file).replace(".json", "")
                        tests_data[workflow_name] = data
                except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                    print(f"Error loading test data from {file}: {e}")
        
        return tests_data

    def load_simulation_data(self) -> Dict[str, Any]:
        """Load simulation data from JSON files.
        
        Returns:
            Dictionary of simulation results data by workflow name.
        """
        simulation_data: Dict[str, Any] = {}

        # Check if simulation directory exists
        if not os.path.exists(self.simulation_dir):
            return simulation_data
        
        # Load simulation results for all workflows
        for file in os.listdir(self.simulation_dir):
            if file.suffix == '.json'):
                file_path = os.path.join(self.simulation_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        workflow_name = os.path.basename(file).replace(".json", "")
                        simulation_data[workflow_name] = data
                except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                    print(f"Error loading simulation data from {file}: {e}")
        
        return simulation_data

    def load_versions_data(self) -> Dict[str, Any]:
        """Load version data from JSON files.
        
        Returns:
            Dictionary of version history data by workflow name.
        """
        versions_data: Dict[str, Any] = {}

        # Check if versions directory exists
        if not os.path.exists(self.versions_dir):
            return versions_data
        
        # Load version histories for all workflows
        for file in os.listdir(self.versions_dir):
            if file.suffix == '.json'):
                file_path = os.path.join(self.versions_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        workflow_name = os.path.basename(file).replace(".json", "")
                        versions_data[workflow_name] = data
                except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
                    print(f"Error loading version data from {file}: {e}")
        
        return versions_data

    def get_workflow_filenames(self) -> Set[str]:
        """Extract workflow filenames from changed files paths"""
        return {os.path.basename(file) for file in self.changed_workflows}

    def get_workflow_validation_status(self, filename: str) -> Dict[str, Any]:
        """Get validation status for a specific workflow"""
        workflow_name = filename.replace(".json", "")
        validation_data = self.load_validation_data()
        
        if not validation_data or workflow_name not in validation_data:
            return cast(Dict[str, Any], {"valid": False, "errors": ["No validation data available"], "warnings": []})
        
        return cast(Dict[str, Any], validation_data[workflow_name])

    def get_workflow_requirements(self, filename: str) -> Dict[str, Any]:
        """Get requirements analysis for a specific workflow"""
        workflow_name = filename.replace(".json", "")
        requirements_data = self.load_requirements_data()
        
        if not requirements_data or workflow_name not in requirements_data:
            return cast(Dict[str, Any], {"gpu_memory": "Unknown", "cpu_memory": "Unknown", "disk_space": "Unknown"})
        
        return cast(Dict[str, Any], requirements_data[workflow_name])

    def get_workflow_test_results(self, filename: str) -> Dict[str, Any]:
        """Get test results for a specific workflow"""
        workflow_name = filename.replace(".json", "")
        tests_data = self.load_tests_data()
        
        if not tests_data or workflow_name not in tests_data:
            return cast(Dict[str, Any], {"passed": False, "errors": ["No test data available"], "warnings": []})
        
        return cast(Dict[str, Any], tests_data[workflow_name])

    def get_workflow_simulation_results(self, filename: str) -> Dict[str, Any]:
        """Get simulation results for a specific workflow"""
        workflow_name = filename.replace(".json", "")
        simulation_data = self.load_simulation_data()
        
        if not simulation_data or workflow_name not in simulation_data:
            return cast(Dict[str, Any], {"success": False, "errors": ["No simulation data available"], "outputs": []})
        
        return cast(Dict[str, Any], simulation_data[workflow_name])

    def get_workflow_version_history(self, filename: str) -> Dict[str, Any]:
        """Get version history for a specific workflow"""
        workflow_name = filename.replace(".json", "")
        versions_data = self.load_versions_data()
        
        if not versions_data or workflow_name not in versions_data:
            return cast(Dict[str, Any], {"has_history": False, "latest_changes": []})
        
        return cast(Dict[str, Any], versions_data[workflow_name])

    def generate_workflow_summary(self, filename: str) -> Dict[str, Any]:
        """Generate a complete summary for a workflow"""
        validation = self.get_workflow_validation_status(filename)
        requirements = self.get_workflow_requirements(filename)
        test = self.get_workflow_test_results(filename)
        simulation = self.get_workflow_simulation_results(filename)
        changes = self.get_workflow_version_history(filename)

        # Determine overall status
        statuses = [
            validation["valid"],
            requirements["gpu_memory"] != "Unknown" or requirements["cpu_memory"] != "Unknown" or requirements["disk_space"] != "Unknown",
            test["passed"],
            simulation["success"],
        ]

        if "fail" in statuses:
            overall_status = "fail"
        elif "warning" in statuses or changes.get("has_history", False):
            overall_status = "warning"
        elif all(status == "pass" for status in statuses):
            overall_status = "pass"
        else:
            overall_status = "unknown"

        return {
        }

    def _generate_status_summary(self, workflow_summaries: dict) -> str:
        """Generate a summary of the overall status of workflows."""
        filenames = list(workflow_summaries.keys())
        
        # Count different statuses
        status_counts = {
            "pass": sum(1 for w in workflow_summaries.values() if w["status"] == "pass"),
            "warning": sum(1 for w in workflow_summaries.values() if w["status"] == "warning"),
            "fail": sum(1 for w in workflow_summaries.values() if w["status"] == "fail"),
            "unknown": sum(1 for w in workflow_summaries.values() if w["status"] == "unknown"),
        }
        
        # Create appropriate summary text based on counts
        if status_counts["fail"] > 0:
            return f"❌ **{status_counts['fail']} of {len(filenames)} workflows have issues that need to be fixed**\n\n"
        elif status_counts["warning"] > 0:
            return f"⚠️ **{status_counts['warning']} of {len(filenames)} workflows have warnings to review**\n\n"
        else:
            return "✅ **All workflow changes look good!**\n\n"

    def _get_status_icon(self, status: str) -> str:
        """Convert a status to its corresponding icon."""
        if status == "pass":
            return "✅"
        elif status == "warning":
            return "⚠️"
        elif status == "fail":
            return "❌"
        else:
            return "❓"

    def _get_memory_text(self, min_mem: int) -> str:
        """Format memory requirement text."""
        if min_mem == 0:
            return "Unknown"
        memory_text = f"{min_mem}GB"
        
        # Highlight high memory requirements
        if min_mem > 16:
            memory_text += " ⚠️"
            
        return memory_text

    def _generate_workflow_table(self, workflow_summaries: dict) -> str:
        """Generate a table summarizing workflow statuses."""
        table = "### Changed Workflows Status\n\n"
        table += "| Workflow | Validation | Test | Simulation | Memory | Status |\n"
        table += "|----------|------------|------|------------|--------|--------|\n"
        
        for filename, summary in sorted(workflow_summaries.items()):
            # Get icons for each status
            validation_icon = self._get_status_icon(summary["validation"]["status"])
            test_icon = self._get_status_icon(summary["test"]["status"])
            simulation_icon = self._get_status_icon(summary["simulation"]["status"])
            
            # Get memory text
            mem_req = summary["requirements"]["memory_required"]["min"]
            memory_text = self._get_memory_text(mem_req)
            
            # Overall status text
            if summary["status"] == "pass":
                status_text = "✅ Pass"
            elif summary["status"] == "warning":
                status_text = "⚠️ Warning"
            elif summary["status"] == "fail":
                status_text = "❌ Fix Required"
            else:
                status_text = "❓ Unknown"
            
            # Add row to table
            table += f"| {filename} | {validation_icon} | {test_icon} | {simulation_icon} | {memory_text} | {status_text} |\n"
            
        return table

    def _generate_recommendations(self, workflow: dict) -> list:
        """Generate recommendations based on workflow issues."""
        recommendations = []
        
        if workflow["validation"]["status"] == "fail":
            recommendations.append("Fix JSON structure and validation issues")
            
        if workflow["test"]["status"] == "fail":
            recommendations.append("Ensure workflow graph is properly connected without circular dependencies")
            
        if workflow["simulation"]["status"] == "fail":
            recommendations.append("Review node implementation compatibility")
            
        if workflow["changes"].get("has_history", False):
            recommendations.append("Consider impact of breaking changes on existing users")
            
        if workflow["requirements"]["memory_required"]["min"] > 16:
            recommendations.append("Optimize for lower memory usage if possible")
            
        return recommendations

    def _format_validation_issues(self, workflow: dict) -> str:
        """Format validation issues for a workflow."""
        if workflow["validation"]["status"] != "fail":
            return ""
        result = "**Validation Issues:**\n\n"
        for issue in workflow["validation"]["issues"]:
            issue_type = issue.get("type", "").replace("_", " ").title()
            result += f"- {issue_type}: {issue.get('message', '')}\n"
        result += "\n"
        
        return result

    def _format_test_issues(self, workflow: dict) -> str:
        """Format test issues for a workflow."""
        if workflow["test"]["status"] != "fail":
            return ""
        result = "**Test Issues:**\n\n"
        for issue in workflow["test"]["issues"]:
            result += f"- {issue}\n"
        result += "\n"
        
        return result

    def _format_simulation_errors(self, workflow: dict) -> str:
        """Format simulation errors for a workflow."""
        if workflow["simulation"]["status"] != "fail":
            return ""
        result = "**Simulation Errors:**\n\n"
        errors = workflow["simulation"]["errors"]
        
        # Show at most 5 errors
        for error in errors[:5]:
            result += f"- {error}\n"
            
        if len(errors) > 5:
            result += f"- ... and {len(errors) - 5} more errors\n"
        result += "\n"
        
        return result

    def _format_breaking_changes(self, workflow: dict) -> str:
        """Format breaking changes for a workflow."""
        if not workflow["changes"].get("has_history", False):
            return ""
        result = "**Breaking Changes Detected:**\n\n"
        for change in workflow["changes"]["latest_changes"]:
            if change.startswith("Removed") or "type changed" in change:
                result += f"- ⚠️ {change}\n"
            else:
                result += f"- {change}\n"
        result += "\n"
        
        return result

    def _format_resource_requirements(self, workflow: dict) -> str:
        """Format resource requirements for a workflow."""
        result = "**Resource Requirements:**\n\n"
        result += f"- Minimum memory: {workflow['requirements']['memory_required']['min']}GB\n"
        result += f"- Recommended memory: {workflow['requirements']['memory_required']['recommended']}GB\n"
        
        if workflow["requirements"]["custom_nodes"]:
            result += f"- Custom nodes: {', '.join(workflow['requirements']['custom_nodes'])}\n"
            
        # Show node and link counts
        if workflow["changes"].get("node_count", 0) > 0:
            result += f"- Total nodes: {workflow['changes']['node_count']}\n"
            result += f"- Total connections: {workflow['changes']['link_count']}\n"
            
        result += "\n"
        
        return result

    def _format_single_workflow_issues(self, workflow: dict) -> str:
        """Format all issues for a single workflow."""
        result = f"#### {workflow['filename']}\n\n"
        
        # Add different issue sections
        result += self._format_validation_issues(workflow)
        result += self._format_test_issues(workflow)
        result += self._format_simulation_errors(workflow)
        result += self._format_breaking_changes(workflow)
        result += self._format_resource_requirements(workflow)
        
        # Add recommendations
        recommendations = self._generate_recommendations(workflow)
        if recommendations:
            result += "**Recommendations:**\n\n"
            for recommendation in recommendations:
                result += f"- {recommendation}\n"
            result += "\n"
            
        return result

    def _generate_workflow_issues(self, failing_workflows: list) -> str:
        """Generate detailed issues section for workflows with problems."""
        if not failing_workflows:
            return ""
        issues_text = "\n### Issues Requiring Attention\n\n"
        
        # Process each workflow with issues
        for workflow in failing_workflows:
            issues_text += self._format_single_workflow_issues(workflow)
                
        return issues_text

    def _generate_about_section(self) -> str:
        """Generate explanation about the workflow validation process."""
        about = "\n### About Workflow Validation\n\n"
        about += "This comment was automatically generated by the CI workflow validation process. It analyzes:\n\n"
        about += "1. **Structural Validation**: Checks JSON structure and node connections\n"
        about += "2. **Execution Testing**: Verifies workflow execution logic\n"
        about += "3. **Simulation**: Tests execution with simulated models (no GPU required)\n"
        about += "4. **Resource Analysis**: Estimates memory and dependency requirements\n"
        about += "5. **Version Tracking**: Detects breaking changes from previous versions\n\n"
        about += "For more details, see the CI artifacts from this PR build.\n"
        
        return about

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
        
        # Add overall status summary
        comment += self._generate_status_summary(workflow_summaries)
        
        # Add workflow summary table
        comment += self._generate_workflow_table(workflow_summaries)
        
        # Add details for workflows with issues
        failing_workflows = [w for w in workflow_summaries.values() if w["status"] in ["fail", "warning"]]
        comment += self._generate_workflow_issues(failing_workflows)
        
        # Add notes about workflow validation process
        comment += self._generate_about_section()

        # Save comment to file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(comment)

        print(f"Comment saved to {self.output_file}")
        return comment

def extract_branch_name(webhook_payload_path: Optional[str]) -> Optional[str]:
    """Extract branch name from GitHub webhook payload.
    
    Args:
        webhook_payload_path: Path to the webhook payload JSON file
        
    Returns:
        Branch name or None if not found/accessible
    """
    if not webhook_payload_path or not os.path.exists(webhook_payload_path):
        return None
    
    try:
        with open(webhook_payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        # Try to extract from different possible locations in the payload
        if "pull_request" in payload and "head" in payload["pull_request"]:
            branch = payload["pull_request"]["head"]["ref"]
            if branch:
                return cast(str, branch)
                
        if "ref" in payload:
            ref = payload["ref"]
            if ref and isinstance(ref, str) and ref.startswith("refs/heads/"):
                return cast(Optional[str], ref.replace("refs/heads/", ""))
    
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError, IOError) as e:
        print(f"Error extracting branch name: {e}")
    
    return None

def _get_title_from_json(file_path: str) -> str:
    """Get workflow title from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "title" in data and data["title"]:
                return cast(str, data["title"])
    except (json.JSONDecodeError, UnicodeDecodeError, IOError, FileNotFoundError):
        pass
    
    # Return filename as fallback
    return os.path.basename(file_path).replace(".json", "")

def get_last_modified_workflows(repo_path: str, days: int = 7) -> List[str]:
    """Get list of workflows modified in the last N days.
    
    Args:
        repo_path: Path to the repository
        days: Number of days to look back
        
    Returns:
        List of workflow files modified in the specified period
    """
    workflows_dir = os.path.join(repo_path, "WebUI/external/workflows")
    modified_workflows: List[str] = []
    
    if not os.path.exists(workflows_dir):
        return modified_workflows
    
    # Calculate the cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Check all workflow files
    for file in os.listdir(workflows_dir):
        if file.suffix == '.json'):
            file_path = os.path.join(workflows_dir, file)
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_time > cutoff_date:
                modified_workflows.append(os.path.join("WebUI/external/workflows", file))
    
    return modified_workflows

def _sanitize_filename(name: str) -> str:
    """Sanitize workflow filename, removing invalid characters."""
    # Remove any special characters that could cause issues
    sanitized = re.sub(r'[^\w\-\.]', '_', name)
    # Ensure it ends with .json
    if not sanitized.suffix == '.json'):
        sanitized += ".json"
    return sanitized

def main() -> None:
    """Run the PR comment generator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", help="Repository root directory")
    parser.add_argument("--branch", help="Branch name for the PR")
    parser.add_argument("--webhook-payload", help="Path to GitHub webhook payload JSON")
    parser.add_argument("--changed-files", help="Comma-separated list of changed files")
    parser.add_argument("--output-file", default="workflow_pr_comment.md", help="Output file for the comment")
    args = parser.parse_args()

    # Get branch name
    branch_name = args.branch
    if not branch_name and args.webhook_payload:
        branch_name = extract_branch_name(args.webhook_payload)

    # Parse changed files
    changed_files = []
    if args.changed_files:
        changed_files = args.changed_files.split(',')
    elif args.repo_root:
        # If no files specified, assume all recently changed files
        changed_files = get_last_modified_workflows(args.repo_root)

    # Generate comment
    generator = PRCommentGenerator(changed_files=changed_files, output_file=args.output_file)
    comment = generator.generate_comment()

    # Write to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(comment)

    print(f"PR comment generated successfully in {args.output_file}")


if __name__ == "__main__":
    main()