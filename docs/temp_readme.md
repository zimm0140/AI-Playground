
# ComfyUI Workflow Repository

This directory contains curated and validated ComfyUI workflows that have been tested and verified for compatibility with our system.

## Workflow Validation System

All workflows in this repository undergo a comprehensive validation process to ensure quality, compatibility, and proper documentation. Our CI system includes:

### 1. Structural Validation

- Schema validation for proper ComfyUI JSON format
- Node syntax checking
- Connection validation between nodes
- Required input checking

### 2. Requirements Analysis

- Model dependency detection (checkpoints, VAEs, LoRAs)
- Extension dependency detection
- Hardware requirements estimation (VRAM, CPU, disk space)
- Python package dependency analysis

### 3. Execution Simulation

- Pathway analysis for workflow execution
- Execution order validation
- Error detection in workflow logic
- Dead node identification (nodes that don't contribute to output)

### 4. Model Simulation

- Tensor-based simulation without requiring full models
- Runtime error detection with specific debugging information
- Data flow validation through the execution pipeline
- Resource usage estimation

### 5. Version Tracking

- Automatic version incrementation for modified workflows
- Breaking change detection
- Feature addition/removal tracking
- Compatibility matrices across versions

### 6. Dashboard Generation

- Comprehensive workflow dashboard with key metrics
- Status indicators for validation, analysis, and simulation
- Resource requirement visualization
- Compatibility information

### 7. PR Integration

- Automatic PR checks for workflow modifications
- Detailed comments with validation results
- Recommendations for fixing issues
- Breaking change warnings

## Using Workflows

Each workflow is provided as a JSON file that can be imported directly into ComfyUI. To use a workflow:

1. Download the JSON file


2. Open ComfyUI in your browser


2. Right-click anywhere in the canvas


2. Select "Load" and choose the downloaded JSON file

## Required Models

Most workflows require specific checkpoint models, VAEs, or LoRAs. Check the dashboard or workflow documentation for specific requirements.

## Contributing Workflows

To contribute a new workflow:

2. Create a fork of this repository


2. Add your workflow JSON file to this directory


2. Create a pull request


2. Our CI system will automatically validate your workflow


2. Address any issues identified in the validation


2. Once all checks pass, your workflow will be reviewed for merging

## Documentation

For more detailed information about the workflow validation system, see [docs/comfyui_workflow_validation.md](../../docs/comfyui_workflow_validation.md)
