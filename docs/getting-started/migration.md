
# Migration Guide

This guide helps users migrate from previous versions of AI-Playground to the current version, highlighting important changes and providing step-by-step instructions for a smooth
transition.

## Migrating from v1.x to v2.x

Version 2.x introduces significant changes to hardware detection, environment management, and workflow processing. Follow these steps to migrate your existing setup:

### Step 1: Update Your Repository

\`\`\`text\`bash
git pull origin main

```text`text

### Step 2: Clean Your Environment

It's recommended to create a fresh environment for v2.x:

```bash

## Remove old environment

rm -rf .venv

## Set up new environment with hardware detection

python setup_hardware_env.py --clean

```text

### Step 3: Update Configuration Files

Configuration files have changed format in v2.x. If you have custom configuration files, you'll need to update them:

1. Update `uvfast.json` to the new schema:

   ```json

   {

```text

 "project_name": "ai-playground",
 "python_version": "3.10",
 "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
 "default_hardware": "base",
 "requirements": {
   "base": "requirements.txt",
   "dev": "requirements-dev.txt",
   "hardware": {

```text

 "base": "requirements-hardware-base.txt",
 "acm": "requirements-hardware-acm.txt"

```text

   }
 }

```text

   }

   ```text

1. Workflows now use the new format in `v2.x`. To migrate existing workflows:

   ```bash

   ## Convert old workflow to new format

   python service/tools/convert_workflow.py --input old_workflow.json --output new_workflow.json

   ```text

### Step 4: API Changes

If you're using the API, note these changes:

1. The base URL has changed from `/api/v1` to `/api/v2`

1. The workflow submission format has been updated

1. Authentication now requires an API key

Example of updated API calls:

```python

## Old v1.x API call

response = requests.post("<http://localhost:8000/api/v1/workflow",> json=workflow_data)

## New v2.x API call

headers = {"X-API-Key": "your_api_key"}
response = requests.post("<http://localhost:8000/api/v2/workflow",> headers=headers, json=workflow_data)

```text

### Step 5: Hardware Optimization Changes

Hardware detection is now more advanced:

1. The system now auto-detects Intel Arc, Battlemage, Meteor Lake, and Lunar Lake devices

1. Optimized packages are installed based on your hardware

1. OpenVINO integration is improved

To manually set hardware type:

```bash

python setup_hardware_env.py --hardware acm

```text

## Breaking Changes

Be aware of these breaking changes in v2.x:

1. Python 3.9 is no longer supported; minimum requirement is Python 3.10

1. Config file format has changed and is not backward compatible

1. The CLI interface has been redesigned with new command syntax

1. Workflow format has been updated for better performance and flexibility

1. Hardware detection now uses a different approach

## Troubleshooting Migration Issues

If you encounter issues during migration:

1. *_Missing dependencies__: Run `python setup_hardware_env.py --dev` to install all dependencies

1. **Configuration errors**: Delete your `uvfast.json` file to regenerate the default configuration

1. **Workflow compatibility**: Use the provided conversion tool for workflows

For further assistance, please [open an issue](https://github.com/intel/AI-Playground/issues) with details about your environment and the problems you're experiencing.

---
**Previous**: [Installation Guide](installation.md) | **Next**: [Hardware Overview](../hardware/overview.md) | __See also_*: [Troubleshooting](../reference/troubleshooting.md)


```text`

```text`
