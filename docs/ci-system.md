
# CI System Documentation {#ci-system-documentation}

This document provides information about the CI (Continuous Integration) system used in the AI-Playground project, how to use it, and how to troubleshoot common issues.

## Table of Contents {#table-of-contents}

- [Overview](#overview)

- [Workflow Files](#workflow-files)

- [Pre-commit Hooks](#pre-commit-hooks)

- [CI Scripts](#ci-scripts)

- [Best Practices](#best-practices)

- [Troubleshooting](#troubleshooting)

- [Contributing to CI](#contributing-to-ci)

## Overview {#overview}

The AI-Playground CI system is designed to:

1. *_Validate Code Quality__: Ensure code meets quality standards before merging.

1. **Run Tests**: Verify that the codebase functions correctly on various platforms.

1. **Generate Documentation**: Keep documentation up-to-date with code changes.

1. **Create Artifacts**: Build and package artifacts for distribution.

1. **Validate ComfyUI Workflows**: Test and validate workflows for compatibility.

The CI system uses GitHub Actions for automation and includes pre-commit hooks for catching issues early in the development process.

## Workflow Files {#workflow-files}

The main workflow files are:

| Workflow File | Purpose |
|---------------|---------|
| `ci.yml` | Main CI workflow, runs tests and validation |
| `ci-cleanup.yml` | Optimizes and cleans CI workflow files |
| `ci-health-check.yml` | Validates the health of the CI system |
| `comfyui-workflow-validation.yml` | Validates ComfyUI workflow files |
| `comfyui-pr-checks.yml` | Runs checks on PRs that modify ComfyUI workflows |
| `ruff.yml` | Runs Python linting using Ruff |

## Pre-commit Hooks {#pre-commit-hooks}

Pre-commit hooks are used to catch issues before they're committed to the repository. They run automatically when you commit changes.

### Setup {#setup}

To set up pre-commit hooks:

\`\`\`text\`bash

## On Unix/Linux/macOS or Git Bash {#on-unixlinuxmacos-or-git-bash}

./.github/setup-hooks.sh

## On Windows with PowerShell {#on-windows-with-powershell}

..github\\setup-hooks.ps1

````text

### Available Hooks {#available-hooks}

- `pre-commit`: Runs linting checks on Python files that are being committed
  - Detects OS and runs the appropriate script (bash or PowerShell)
  - Validates code against common issues like unused imports, bad regex patterns

### Skipping Hooks {#skipping-hooks}

In case you need to bypass hooks temporarily:

```b
as
h

git commit --no-verify

```

##

CI {#ci}

 Scripts {#ci-scripts}

The CI system includes several utility scripts that help maintain code quality and workflow efficiency:

| Script | Purpose |
|--------|---------|
| `fix_ci_issues.py` | Fixes common issues in CI like indentation in test files |
| `ensure_unique_artifacts.py` | Ensures artifact names are unique across workflows |
| `optimize_ci.py` | Optimizes CI workflow files for better performance |
| `lint_python_files.py` | Lints Python files for common issues |
| `remove_duplicate_sections.py` | Removes duplicate sections in workflow files |

## Best Practices {#best-practices}

To ensure smooth CI operation:

1. **Keep workflow files organized**: Each workflow should have a single responsibility

1. **Use unique artifact names**: Append job name or matrix variables to artifact names

1. **Include conditionals**: Use `if: always()` for artifact uploads to ensure they run even if tests fail

1. **Optimize cache usage**: Use dependency hashing and OS-specific cache paths

1. **Keep workflows lean**: Combine similar steps and use job dependencies

1. **Run pre-commit hooks locally**: Catch issues before pushing to remote

## Troubleshooting {#troubleshooting}

### Common Issues {#common-issues}

#### Artifact Name Conflicts {#artifact-name-conflicts}

**Symptom**: CI job fails with `Error: Failed to CreateArtifact: Received non-retryable error: Failed request: (409) Conflict`

**Solution**: Run the CI cleanup workflow which will ensure unique artifact names:

``
`b
ash

## Via GitHub Actions web UI {#via-github-actions-web-ui}

## Go to Actions > CI Cleanup and Optimization > Run workflow {#go-to-actions-ci-cleanup-and-optimization-run-workflow}

```

#
##

# Linting Errors {#linting-errors}

**Symptom**: Ruff or linting check fails with errors like `F401 import xxx is unused`

**Solution**: Run the lint script locally to identify and fix issues:

``
`bash

python .github/workflows/scripts/lint_python_files.py path/to/file.py

```

#

### Indentation Errors in Python Files {#indentation-errors-in-python-files}

**Symptom**: CI fails with indentation errors, particularly in `try/except` blocks

**Solution**: Run the fix_ci_issues script:

```bash

python .github/workflows/scripts/fix_ci_issues.py

```

#### Windows Path Issues {#windows-path-issues}

**Symptom**: Backslash escaping issues in regex patterns

__Solution_*: Always use raw strings (`r"pattern"`) for regex patterns and double backslashes (`\\\\`) in string templates.

## Contributing to CI {#contributing-to-ci}

When contributing to the CI system:

1. Test changes locally before pushing

1. Document any new workflows or scripts

1. Update this documentation if you add/modify CI capabilities

1. Keep backwards compatibility in mind

1. Consider cross-platform compatibility (Windows, Linux, macOS)

### Adding a New Workflow {#adding-a-new-workflow}

1. Use existing workflows as templates

1. Ensure proper error handling

1. Use conditionals to control when jobs run

1. Provide clear job and step names

1. Optimize for performance (use caching, fetch-depth: 1, etc.)

1. Add status badges for visibility

## CI Performance Metrics {#ci-performance-metrics}

The CI system tracks performance metrics to help identify bottlenecks and improve efficiency over time. These metrics are available as artifacts in the CI job outputs and include:

- Job duration

- Step duration

- Cache hit rates

- Resource utilization

## Further Reading {#further-reading}

- [GitHub Actions Documentation](https://docs.github.com/en/actions)

- [Pre-commit Hooks Guide](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)

- [Ruff Documentation](https://docs.astral.sh/ruff/)

````

````