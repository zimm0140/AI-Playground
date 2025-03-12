#!/usr/bin/env python
"""
Fix artifact name conflicts in workflow files.

This script updates artifact names in GitHub Actions workflow files to include job identifiers,
preventing conflicts when multiple jobs upload artifacts with the same name.
"""

import os
import re
import sys


def fix_artifact_names(file_path):
    """Fix artifact names in a workflow file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Keep track of the changes we make
    changes = []

    # Pattern to find job definitions
    job_pattern = re.compile(r"(\s+)(\w+):\s*\n\s+runs-on:")
    job_matches = list(job_pattern.finditer(content))

    # Process each job in the file
    for i, job_match in enumerate(job_matches):
        job_name = job_match.group(2)
        job_start = job_match.start()

        # Determine the end of the job section
        job_end = len(content)
        if i < len(job_matches) - 1:
            job_end = job_matches[i + 1].start()

        job_content = content[job_start:job_end]

        # Find artifact uploads within the job
        artifact_pattern = re.compile(
            r"(\s+)- name: Upload .*\n.*uses: actions/upload-artifact.*\n.*with:\s*\n\s+name: ([^\n]+)"
        )
        artifact_matches = list(artifact_pattern.finditer(job_content))

        for artifact_match in artifact_matches:
            _ = artifact_match.group(1)  # noqa: F841 (was artifact_indent)
            artifact_name = artifact_match.group(2)

            # Skip if the artifact name already includes a job identifier
            if f"-{job_name}" in artifact_name or job_name in artifact_name:
                continue

            # Create a new unique name with the job identifier
            new_artifact_name = f"{artifact_name}-{job_name}"

            # Replace the artifact name in the content
            pattern = re.compile(
                f"(\\s+name: Upload .*\\n.*uses: actions/upload-artifact.*\\n.*with:\\s*\\n\\s+name: ){re.escape(artifact_name)}"
            )
            new_content = pattern.sub(f"\\1{new_artifact_name}", content)

            if new_content != content:
                changes.append(
                    f"  - Changed artifact name '{artifact_name}' to '{new_artifact_name}'"
                )
                content = new_content

    # Only write the file if changes were made
    if changes:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {file_path}:")
        for change in changes:
            print(change)
        return True
    else:
        print(f"No changes needed in {file_path}")
        return False


def main():
    """Main function to process all workflow files."""
    workflow_dir = os.path.join(".github", "workflows")

    if not os.path.exists(workflow_dir):
        print(f"Error: Directory {workflow_dir} not found.")
        return 1

    updated_files = 0

    for filename in os.listdir(workflow_dir):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            file_path = os.path.join(workflow_dir, filename)
            if fix_artifact_names(file_path):
                updated_files += 1

    print(f"Updated {updated_files} workflow files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
