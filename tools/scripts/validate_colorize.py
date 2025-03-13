import json
import sys
from pathlib import Path

import jsonschema


def validate_workflow():
    try:
        # Load schema and workflow files
        schema_file = "WebUI/external/schemas/workflow-schema.json"
        workflow_file = "WebUI/external/workflows/Colorize.json"

        print(f"Using Python from: {sys.executable}")
        print(f"Loading schema from: {schema_file}")
        print(f"Loading workflow from: {workflow_file}")

        with Path(schema_file).open() as f:
            schema = json.load(f)

        with Path(workflow_file).open() as f:
            workflow = json.load(f)

        # Validate the workflow against the schema
        jsonschema.validate(workflow, schema)
        print("Validation successful!")
        return 0
    except Exception as e:
        print(f"Validation error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(validate_workflow())
