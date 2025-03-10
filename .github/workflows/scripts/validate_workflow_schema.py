#!/usr/bin/env python3
"""
Workflow Schema Validator

This script validates ComfyUI workflow JSON files against the defined schema.
It provides detailed error reports and can be integrated into CI/CD pipelines.

Usage:
    python validate_workflow_schema.py --workflows-dir DIR --schema-file FILE --output-file FILE
"""

import os
import sys
import json
import argparse
import jsonschema
from datetime import datetime

# Add the parent directory to the path so we can import the utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.workflow_parser import get_workflow_nodes


def validate_workflow(workflow_file, schema):
    """
    Validate a single workflow file against the schema.

    Args:
        workflow_file (str): Path to the workflow JSON file
        schema (dict): The loaded JSON schema

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # Use utility function to check for nodes in different formats
        nodes = get_workflow_nodes(workflow)

        # Log information about the workflow format for debugging
        if nodes:
            if "nodes" in workflow and workflow["nodes"] == nodes:
                print(
                    f"Validating {os.path.basename(workflow_file)} (top-level nodes format)"
                )
            elif "comfyUiApiWorkflow" in workflow:
                if (
                    "nodes" in workflow["comfyUiApiWorkflow"]
                    and workflow["comfyUiApiWorkflow"]["nodes"] == nodes
                ):
                    print(
                        f"Validating {os.path.basename(workflow_file)} (comfyUiApiWorkflow.nodes format)"
                    )
                else:
                    print(
                        f"Validating {os.path.basename(workflow_file)} (comfyUiApiWorkflow format)"
                    )
        else:
            print(
                f"Warning: {os.path.basename(workflow_file)} does not contain nodes in any recognized format"
            )

        jsonschema.validate(instance=workflow, schema=schema)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except jsonschema.exceptions.ValidationError as e:
        return (
            False,
            f"Schema validation error: {e.message} at {'/'.join([str(p) for p in e.path])}",
        )
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_all_workflows(workflows_dir, schema_file, output_file):
    """
    Validate all workflow files in a directory against the schema.

    Args:
        workflows_dir (str): Directory containing workflow JSON files
        schema_file (str): Path to the JSON schema file
        output_file (str): Path to write the validation report
    """
    # Load the schema
    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        print(f"Error loading schema file: {str(e)}")
        sys.exit(1)

    # Find workflow files
    workflow_files = []
    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            workflow_files.append(os.path.join(workflows_dir, filename))

    if not workflow_files:
        print(f"No workflow files found in {workflows_dir}")
        sys.exit(1)

    # Validate each workflow
    results = {}
    valid_count = 0
    invalid_count = 0

    for workflow_file in workflow_files:
        filename = os.path.basename(workflow_file)
        is_valid, error = validate_workflow(workflow_file, schema)

        if is_valid:
            valid_count += 1
            results[filename] = {"valid": True}
        else:
            invalid_count += 1
            results[filename] = {"valid": False, "error": error}

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "schema_file": schema_file,
        "workflows_directory": workflows_dir,
        "summary": {
            "total_workflows": len(workflow_files),
            "valid_workflows": valid_count,
            "invalid_workflows": invalid_count,
        },
        "results": results,
    }

    # Write report to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary to console
    print("\nWorkflow Schema Validation Summary:")
    print(f"Total workflows: {len(workflow_files)}")
    print(f"Valid workflows: {valid_count}")
    print(f"Invalid workflows: {invalid_count}")

    if invalid_count > 0:
        print("\nInvalid workflows:")
        for filename, result in results.items():
            if not result["valid"]:
                print(f"  {filename}: {result['error']}")

    print(f"\nDetailed report written to: {output_file}")

    # Return exit code based on validation result
    return 0 if invalid_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate ComfyUI workflow JSON files against schema"
    )
    parser.add_argument(
        "--workflows-dir",
        required=True,
        help="Directory containing workflow JSON files",
    )
    parser.add_argument(
        "--schema-file", required=True, help="Path to the JSON schema file"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to write the validation report"
    )

    args = parser.parse_args()

    exit_code = validate_all_workflows(
        args.workflows_dir, args.schema_file, args.output_file
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
