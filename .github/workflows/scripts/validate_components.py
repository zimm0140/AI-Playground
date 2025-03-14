from pathlib import Path

#!/usr/bin/env python3
"""
Component Validator

This script validates component JSON files to ensure they follow the required format
and contain all necessary information.

Usage:
    python validate_components.py --components-dir DIR --output-dir DIR
"""

import argparse
import json
import os
import sys
from datetime import datetime


def validate_component(component_file):
    """
    Validate a single component file.

    Args:
        component_file (str): Path to the component JSON file

    Returns:
        tuple: (is_valid, list of issues)
    """
    issues = []

    try:
        with open(component_file, encoding="utf-8") as f:
            component = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {str(e)}"]
    except Exception as e:
        return False, [f"Error reading file: {str(e)}"]

    # Check required fields
    required_fields = [
        "name",
        "description",
        "version",
        "type",
        "inputs",
        "outputs",
        "nodes",
    ]
    for field in required_fields:
        if field not in component:
            issues.append(f"Missing required field: {field}")

    # Check version format (semver)
    if "version" in component:
        version = component["version"]
        parts = version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            issues.append(
                f"Invalid version format: {version}. Should be semver (e.g., 1.0.0)",
            )

    # Check inputs
    if "inputs" in component and isinstance(component["inputs"], list):
        for i, input_item in enumerate(component["inputs"]):
            if not isinstance(input_item, dict):
                issues.append(f"Input {i} is not an object")
                continue

            # Check required input fields
            input_required_fields = ["name", "type", "description"]
            for field in input_required_fields:
                if field not in input_item:
                    issues.append(
                        f"Input {i} ({input_item.get('name', 'unnamed')}) missing required field: {field}",
                    )

    # Check outputs
    if "outputs" in component and isinstance(component["outputs"], list):
        for i, output_item in enumerate(component["outputs"]):
            if not isinstance(output_item, dict):
                issues.append(f"Output {i} is not an object")
                continue

            # Check required output fields
            output_required_fields = ["name", "type", "description"]
            for field in output_required_fields:
                if field not in output_item:
                    issues.append(
                        f"Output {i} ({output_item.get('name', 'unnamed')}) missing required field: {field}",
                    )

    # Check nodes
    if "nodes" in component and isinstance(component["nodes"], dict):
        for node_id, node in component["nodes"].items():
            if not isinstance(node, dict):
                issues.append(f"Node {node_id} is not an object")
                continue

            # Check required node fields
            if "class_type" not in node:
                issues.append(f"Node {node_id} missing required field: class_type")

            if "inputs" not in node:
                issues.append(f"Node {node_id} missing required field: inputs")

    # Check input mappings
    if "inputMappings" in component:
        if not isinstance(component["inputMappings"], dict):
            issues.append("inputMappings is not an object")
        else:
            for input_name, mapping in component["inputMappings"].items():
                if not isinstance(mapping, dict):
                    issues.append(f"Input mapping for {input_name} is not an object")
                    continue

                if "nodeId" not in mapping:
                    issues.append(
                        f"Input mapping for {input_name} missing required field: nodeId",
                    )

                if "inputName" not in mapping:
                    issues.append(
                        f"Input mapping for {input_name} missing required field: inputName",
                    )

    # Check output mappings
    if "outputMappings" in component:
        if not isinstance(component["outputMappings"], dict):
            issues.append("outputMappings is not an object")
        else:
            for output_name, mapping in component["outputMappings"].items():
                if not isinstance(mapping, dict):
                    issues.append(f"Output mapping for {output_name} is not an object")
                    continue

                if "nodeId" not in mapping:
                    issues.append(
                        f"Output mapping for {output_name} missing required field: nodeId",
                    )

                if "outputIndex" not in mapping:
                    issues.append(
                        f"Output mapping for {output_name} missing required field: outputIndex",
                    )

    return len(issues) == 0, issues


def validate_all_components(components_dir, output_dir):
    """
    Validate all component files in a directory.

    Args:
        components_dir (str): Directory containing component JSON files
        output_dir (str): Directory to write validation reports
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all component files
    component_files = []
    for filename in Path(components_dir).iterdir():
        if filename.endswith(".json") and filename != "schema.json":
            component_files.append(os.path.join(components_dir, filename))

    if not component_files:
        print(f"No component files found in {components_dir}")
        return True

    # Validate each component
    results = {
        "timestamp": datetime.now().isoformat(),
        "components_directory": components_dir,
        "summary": {
            "total_components": len(component_files),
            "valid_components": 0,
            "invalid_components": 0,
        },
        "components": [],
    }

    for component_file in component_files:
        filename = os.path.basename(component_file)
        is_valid, issues = validate_component(component_file)

        component_result = {
            "filename": filename,
            "path": component_file,
            "is_valid": is_valid,
            "issues": issues,
        }

        results["components"].append(component_result)

        if is_valid:
            results["summary"]["valid_components"] += 1
            print(f"✅ {filename} is valid")
        else:
            results["summary"]["invalid_components"] += 1
            print(f"❌ {filename} has {len(issues)} issues:")
            for issue in issues:
                print(f"   - {issue}")

    # Write results to file
    output_file = os.path.join(output_dir, "component_validation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Create a validation failed marker if any components are invalid
    if results["summary"]["invalid_components"] > 0:
        with open(os.path.join(output_dir, "validation_failed"), "w") as f:
            f.write("Component validation failed")

    # Print summary
    print("\nComponent Validation Summary:")
    print(f"Total components: {results['summary']['total_components']}")
    print(f"Valid components: {results['summary']['valid_components']}")
    print(f"Invalid components: {results['summary']['invalid_components']}")
    print(f"\nDetailed results written to: {output_file}")

    return results["summary"]["invalid_components"] == 0


def main():
    parser = argparse.ArgumentParser(description="Validate component JSON files")
    parser.add_argument(
        "--components-dir",
        required=True,
        help="Directory containing component JSON files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write validation reports",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.components_dir):
        print(f"Error: Components directory {args.components_dir} not found")
        sys.exit(1)

    success = validate_all_components(args.components_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
