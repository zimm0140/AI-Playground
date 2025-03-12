#!/usr/bin/env python3
"""
Ensure Unique Artifacts Script

This script scans all GitHub Actions workflow files (.yml) in the .github/workflows directory
and ensures that all artifact names are unique to prevent conflicts.
It adds job-specific prefixes to artifact names to make them unique.
"""

import glob
import os
import re


def ensure_unique_artifacts():
    """Scan all workflow files and ensure unique artifact names"""
    print("Ensuring unique artifact names in workflow files...")

    # Find all workflow files
    workflow_files = glob.glob(".github/workflows/*.yml") + glob.glob(
        ".github/workflows/*.yaml"
    )
    workflow_files = [
        f for f in workflow_files if not f.endswith("ensure_unique_artifacts.yml")
    ]

    if not workflow_files:
        print("No workflow files found.")
        return

    print(f"Found {len(workflow_files)} workflow files to process.")

    # Track all artifact names
    artifact_names = {}  # {artifact_name: [file_path, job_name]}

    # First pass - catalog all artifact names
    for file_path in workflow_files:
        with open(file_path) as f:
            content = f.read()

        # Extract workflow name
        workflow_name_match = re.search(r"name:\s*([^\n]+)", content)
        _ = (  # noqa: F841 (was workflow_id)
            workflow_name_match.group(1).strip()
            if workflow_name_match
            else os.path.basename(file_path)
        )

        # Extract jobs and their names
        current_job = None
        current_job_match = None
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Check for job definition
            job_match = re.search(r"^\s*(\w+):\s*$", line)
            if job_match and "jobs:" in "".join(lines[max(0, i - 5) : i]):
                current_job = job_match.group(1)

            # Check for name field within a job
            if current_job:
                name_match = re.search(r"^\s*name:\s*(.+)$", line)
                if name_match:
                    current_job_match = name_match.group(1).strip()

            # Check for artifact uploads
            if "uses: actions/upload-artifact" in line:
                # Look for name: parameter in subsequent lines
                for j in range(i + 1, min(i + 8, len(lines))):
                    artifact_match = re.search(r"^\s*name:\s*(.+)$", lines[j])
                    if artifact_match:
                        artifact_name = artifact_match.group(1).strip().strip("\"'")
                        # Store with file path and job name
                        job_name = current_job_match or current_job
                        if artifact_name in artifact_names:
                            # Already exists - potential conflict
                            artifact_names[artifact_name].append((file_path, job_name))
                        else:
                            artifact_names[artifact_name] = [(file_path, job_name)]
                        break

    # Find artifacts with potential conflicts
    conflicts = {
        name: locations
        for name, locations in artifact_names.items()
        if len(locations) > 1
    }

    if not conflicts:
        print("No artifact name conflicts found.")
        return

    print(f"Found {len(conflicts)} artifact names with potential conflicts:")
    for name, locations in conflicts.items():
        print(f"  - '{name}' used in:")
        for file_path, job_name in locations:
            print(f"      {file_path} (job: {job_name})")

    # Second pass - update artifact names to make them unique
    for file_path in workflow_files:
        updated = False
        with open(file_path) as f:
            lines = f.readlines()

        # Extract workflow name for prefixing
        workflow_name_match = re.search(r"name:\s*([^\n]+)", "".join(lines))
        workflow_prefix = (
            workflow_name_match.group(1).strip()
            if workflow_name_match
            else os.path.basename(file_path)
        )
        workflow_prefix = workflow_prefix.lower().replace(" ", "-")[
            :10
        ]  # Keep it short

        # Keep track of current job
        current_job = None
        current_job_match = None
        in_upload_section = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for job definition
            job_match = re.search(r"^\s*(\w+):\s*$", line)
            if job_match and i > 0 and "jobs:" in "".join(lines[max(0, i - 5) : i]):
                current_job = job_match.group(1)
                current_job_match = None

            # Check for name field within a job
            if current_job:
                name_match = re.search(r"^\s*name:\s*(.+)$", line)
                if name_match:
                    current_job_match = name_match.group(1).strip()

            # Check for start of upload-artifact action
            if "uses: actions/upload-artifact" in line:
                in_upload_section = True

            # Look for name: parameter in an upload-artifact section
            if in_upload_section:
                artifact_match = re.search(r"^\s*name:\s*(.+)$", line)
                if artifact_match:
                    artifact_name = artifact_match.group(1).strip().strip("\"'")

                    # Check if this artifact name has conflicts
                    if artifact_name in conflicts:
                        # Create a unique name by adding job name or workflow name
                        job_prefix = (
                            current_job_match or current_job or workflow_prefix
                        ).lower()
                        job_prefix = re.sub(
                            r"[^a-z0-9-]", "", job_prefix.replace(" ", "-")
                        )[:10]

                        # Only update if the name isn't already unique
                        if not artifact_name.endswith(
                            f"-{job_prefix}"
                        ) and not artifact_name.startswith(f"{job_prefix}-"):
                            if artifact_name in [
                                "build-artifacts",
                                "release-artifacts",
                                "test-results",
                                "coverage-data",
                                "pytest-results",
                                "tool-compatibility-report",
                                "hardware-support-matrix",
                                "api-documentation",
                            ]:
                                new_name = f"{artifact_name}-{job_prefix}"
                            else:
                                new_name = f"{job_prefix}-{artifact_name}"

                            # Update the line
                            indent = len(line) - len(line.lstrip())
                            lines[i] = " " * indent + f"name: {new_name}\n"
                            print(
                                f"  - Renamed '{artifact_name}' to '{new_name}' in {file_path}"
                            )
                            updated = True

                # Check for end of upload-artifact section
                if (
                    line.strip()
                    and not line.strip().startswith("-")
                    and not line.strip().startswith("name:")
                    and not line.strip().startswith("path:")
                    and not line.strip().startswith("retention-days:")
                    and not line.strip().startswith("if-no-files-found:")
                    and not line.strip().startswith("with:")
                ):
                    in_upload_section = False

            i += 1

        # Write back if changes were made
        if updated:
            with open(file_path, "w") as f:
                f.writelines(lines)
            print(f"Updated {file_path} with unique artifact names")

    print("Artifact name conflicts resolved.")


if __name__ == "__main__":
    ensure_unique_artifacts()
