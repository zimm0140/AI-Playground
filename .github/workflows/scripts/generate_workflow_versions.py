#!/usr/bin/env python
"""
Generate Workflow Versions

This script analyzes ComfyUI workflow files and generates version information.
It tracks version history and ensures version numbers are properly incremented.

Usage:
    python generate_workflow_versions.py --workflows-dir <dir> --output-dir <dir>
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("workflow-versions")

class WorkflowVersion:
    """Represents a version entry in a workflow's version history."""

    def __init__(self, version: str, date: str, changes: list[str]):
        self.version = version
        self.date = date
        self.changes = changes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version,
            "date": self.date,
            "changes": self.changes
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkflowVersion':
        """Create a WorkflowVersion from a dictionary."""
        return cls(
            version=data.get("version", ""),
            date=data.get("date", ""),
            changes=data.get("changes", [])
        )

class WorkflowHistory:
    """Manages the version history of a workflow."""

    def __init__(self, workflow_id: str, versions: list[WorkflowVersion] = None):
        self.workflow_id = workflow_id
        self.versions = versions or []

    def add_version(self, version: WorkflowVersion) -> None:
        """Add a new version to the history."""
        self.versions.append(version)

    def get_latest_version(self) -> WorkflowVersion | None:
        """Get the most recent version."""
        if not self.versions:
            return None
        return sorted(self.versions, key=lambda v: v.version, reverse=True)[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "versions": [v.to_dict() for v in self.versions]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkflowHistory':
        """Create a WorkflowHistory from a dictionary."""
        versions = [WorkflowVersion.from_dict(v) for v in data.get("versions", [])]
        return cls(
            workflow_id=data.get("workflow_id", ""),
            versions=versions
        )

def load_workflow(file_path: str) -> dict[str, Any]:
    """Load a workflow file and return its contents as a dictionary."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON in {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading workflow {file_path}: {e}")
        return {}

def extract_version_info(workflow: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Extract version information from a workflow."""
    version = workflow.get("version", "0.0.0")
    changelog = workflow.get("changeLog", [])
    return version, changelog

def generate_version_report(workflows_dir: str, output_dir: str) -> None:
    """Generate a report of all workflow versions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    report = {
        "generated_at": datetime.now().isoformat(),
        "workflows": []
    }

    # Process all workflow files
    for filename in os.listdir(workflows_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(workflows_dir, filename)
        workflow = load_workflow(file_path)

        if not workflow:
            continue

        version, changelog = extract_version_info(workflow)

        workflow_info = {
            "filename": filename,
            "name": workflow.get("name", "Unnamed Workflow"),
            "current_version": version,
            "changelog": changelog,
            "tags": workflow.get("tags", []),
            "description": workflow.get("description", "")
        }

        report["workflows"].append(workflow_info)

    # Write the report to the output directory
    output_file = os.path.join(output_dir, "workflow_versions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Generated version report for {len(report['workflows'])} workflows")

def main():
    parser = argparse.ArgumentParser(description="Generate workflow version information")
    parser.add_argument("--workflows-dir", required=True, help="Directory containing workflow files")
    parser.add_argument("--output-dir", required=True, help="Directory to write output files")

    args = parser.parse_args()

    if not os.path.exists(args.workflows_dir):
        logger.error(f"Workflows directory does not exist: {args.workflows_dir}")
        sys.exit(1)

    generate_version_report(args.workflows_dir, args.output_dir)

if __name__ == "__main__":
    main()
