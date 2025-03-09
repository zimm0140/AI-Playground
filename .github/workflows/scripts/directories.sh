#!/bin/bash
# Create required directories for workflow system

# Main directories
mkdir -p WebUI/external/schemas
mkdir -p WebUI/external/workflows
mkdir -p WebUI/external/docs/workflows

# CI artifacts directories
mkdir -p ci_artifacts/workflow_validation
mkdir -p ci_artifacts/workflow_requirements
mkdir -p ci_artifacts/workflow_tests
mkdir -p ci_artifacts/workflow_versions
mkdir -p ci_artifacts/workflow_dashboard
mkdir -p ci_artifacts/workflow_docs

echo "Created all required directories for the workflow system." 