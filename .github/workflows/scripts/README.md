# GitHub Workflow Scripts

This directory contains scripts used by GitHub Actions workflows.

## Important Note

These scripts are copies of scripts from the `tools` directory. The originals are now located at:

- `tools/linting/fix_markdown_lint.py` - Script for fixing common markdown linting issues

## Maintenance

When updating scripts in the `tools` directory, please remember to also update the copies in this directory to keep them in sync. This ensures that both local development and CI/CD workflows use the same logic.

The duplication is necessary because GitHub Actions workflows reference these scripts by path, and changing all workflows to reference the new paths would require significant changes to multiple workflow files.

## Usage

These scripts are primarily called by GitHub Actions workflows defined in the `.github/workflows` directory. They should not be called directly by users or developers, who should instead use the versions in the `tools` directory.
