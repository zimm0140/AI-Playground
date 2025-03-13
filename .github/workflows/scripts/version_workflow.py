#!/usr/bin/env python3
"""
Workflow Version Manager

This script adds and updates version information in ComfyUI workflow files.
It also detects breaking changes between workflow versions.

Usage:
    python version_workflow.py --workflows-dir DIR [--bump major|minor|patch] [--workflow FILE]
"""

import argparse
import json
import os
import sys
from datetime import date


def load_workflow(workflow_file):
    """Load a workflow from a JSON file."""
    try:
        with open(workflow_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading workflow file: {str(e)}")
        return None


def save_workflow(workflow, workflow_file):
    """Save a workflow to a JSON file."""
    try:
        with open(workflow_file, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving workflow file: {str(e)}")
        return False


def parse_version(version_str):
    """Parse a version string into a tuple of integers."""
    try:
        if not version_str:
            return (0, 0, 0)
        parts = version_str.split(".")
        if len(parts) < 3:
            parts.extend(["0"] * (3 - len(parts)))
        return tuple(map(int, parts[:3]))
    except Exception:
        return (0, 0, 0)


def format_version(version_tuple):
    """Format a version tuple as a string."""
    return ".".join(map(str, version_tuple))


def bump_version(current_version, bump_type):
    """Bump a version according to the specified type."""
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        return format_version((major + 1, 0, 0))
    if bump_type == "minor":
        return format_version((major, minor + 1, 0))
    if bump_type == "patch":
        return format_version((major, minor, patch + 1))
    return current_version


def detect_breaking_changes(old_workflow, new_workflow):
    """
    Detect breaking changes between two versions of a workflow.
    Returns a list of potential breaking changes.
    """
    breaking_changes = []

    # Check if the workflow exists
    if not old_workflow or not new_workflow:
        return ["Unable to compare workflows - one or both are missing"]

    # Check for removed inputs
    old_inputs = {
        inp.get("nodeInput", ""): inp for inp in old_workflow.get("inputs", [])
    }
    new_inputs = {
        inp.get("nodeInput", ""): inp for inp in new_workflow.get("inputs", [])
    }

    removed_inputs = set(old_inputs.keys()) - set(new_inputs.keys())
    if removed_inputs:
        breaking_changes.append(f"Removed inputs: {', '.join(removed_inputs)}")

    # Check for changed input types
    for input_name in set(old_inputs.keys()) & set(new_inputs.keys()):
        old_type = old_inputs[input_name].get("type")
        new_type = new_inputs[input_name].get("type")
        if old_type != new_type:
            breaking_changes.append(
                f"Changed type of input '{input_name}' from '{old_type}' to '{new_type}'",
            )

    # Check for changed model requirements
    old_models = {
        model.get("model", "")
        for model in old_workflow.get("comfyUIRequirements", {}).get(
            "requiredModels", [],
        )
    }
    new_models = {
        model.get("model", "")
        for model in new_workflow.get("comfyUIRequirements", {}).get(
            "requiredModels", [],
        )
    }

    removed_models = old_models - new_models
    if removed_models:
        breaking_changes.append(
            f"Removed model requirements: {', '.join(removed_models)}",
        )

    # Check for changed backend
    if old_workflow.get("backend") != new_workflow.get("backend"):
        breaking_changes.append(
            f"Changed backend from '{old_workflow.get('backend')}' to '{new_workflow.get('backend')}'",
        )

    # Check for removed outputs
    old_outputs = {out.get("name", ""): out for out in old_workflow.get("outputs", [])}
    new_outputs = {out.get("name", ""): out for out in new_workflow.get("outputs", [])}

    removed_outputs = set(old_outputs.keys()) - set(new_outputs.keys())
    if removed_outputs:
        breaking_changes.append(f"Removed outputs: {', '.join(removed_outputs)}")

    # Check for changed output types
    for output_name in set(old_outputs.keys()) & set(new_outputs.keys()):
        old_type = old_outputs[output_name].get("type")
        new_type = new_outputs[output_name].get("type")
        if old_type != new_type:
            breaking_changes.append(
                f"Changed type of output '{output_name}' from '{old_type}' to '{new_type}'",
            )

    return breaking_changes


def update_workflow_version(workflow_file, bump_type=None):
    """Update version information for a workflow file."""
    workflow = load_workflow(workflow_file)
    if not workflow:
        return False

    # Get current version
    current_version = workflow.get("version", "0.0.0")

    # Determine new version
    new_version = current_version
    if bump_type:
        new_version = bump_version(current_version, bump_type)

    # If no version or changelog exists, initialize them
    if "version" not in workflow:
        workflow["version"] = new_version

    if "changeLog" not in workflow:
        workflow["changeLog"] = []

    # Add changelog entry if version changed
    if new_version != current_version:
        workflow["version"] = new_version

        # Check for existing entry with this version
        for entry in workflow["changeLog"]:
            if entry.get("version") == new_version:
                print(f"Version {new_version} already exists in changelog")
                return False

        # Add new changelog entry
        today = date.today().isoformat()
        changelog_entry = {
            "version": new_version,
            "date": today,
            "changes": ["Version updated"],
        }
        workflow["changeLog"].insert(0, changelog_entry)

    # Save updated workflow
    if save_workflow(workflow, workflow_file):
        print(f"Updated {workflow_file} to version {new_version}")
        return True

    return False


def find_previous_version(workflow_file):
    """Find the previous version of a workflow in version control."""
    # This is a placeholder. In a real implementation, this would use git or other VCS
    # to retrieve the previous version of the file.
    print(
        "Warning: Finding previous versions requires integration with version control.",
    )
    print("This feature is not implemented in this script.")


def process_all_workflows(workflows_dir, bump_type=None):
    """Process all workflow files in a directory."""
    success_count = 0
    error_count = 0

    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            workflow_file = os.path.join(workflows_dir, filename)
            print(f"Processing {filename}...")

            if update_workflow_version(workflow_file, bump_type):
                success_count += 1
            else:
                error_count += 1

    print(f"\nProcessed {success_count + error_count} workflow files")
    print(f"Successfully updated: {success_count}")
    print(f"Errors: {error_count}")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage versioning for ComfyUI workflow files",
    )
    parser.add_argument(
        "--workflows-dir", help="Directory containing workflow JSON files",
    )
    parser.add_argument("--workflow", help="Path to a specific workflow file to update")
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version number (major.minor.patch)",
    )
    parser.add_argument(
        "--check-breaking-changes",
        action="store_true",
        help="Check for breaking changes with previous version",
    )

    args = parser.parse_args()

    if args.workflow:
        # Process a single workflow file
        if not os.path.isfile(args.workflow):
            print(f"Error: Workflow file {args.workflow} not found")
            sys.exit(1)

        if args.check_breaking_changes:
            previous_workflow = find_previous_version(args.workflow)
            current_workflow = load_workflow(args.workflow)

            if previous_workflow and current_workflow:
                changes = detect_breaking_changes(previous_workflow, current_workflow)
                if changes:
                    print("Breaking changes detected:")
                    for change in changes:
                        print(f"  - {change}")

                    # Suggest version bump
                    print("\nBreaking changes detected. Consider a major version bump.")

        success = update_workflow_version(args.workflow, args.bump)
        sys.exit(0 if success else 1)

    elif args.workflows_dir:
        # Process all workflows in a directory
        if not os.path.isdir(args.workflows_dir):
            print(f"Error: Workflows directory {args.workflows_dir} not found")
            sys.exit(1)

        success = process_all_workflows(args.workflows_dir, args.bump)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
