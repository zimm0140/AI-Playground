# Migration Guide for AI Playground

This guide helps you migrate to the modern Python development workflow using uv and Python 3.10+.

## Table of Contents

1. [Migrating from pip to uv](#migrating-from-pip-to-uv)
2. [Updating Type Annotations for Python 3.10+](#updating-type-annotations-for-python-310)
3. [Using Lockfiles for Reproducible Environments](#using-lockfiles-for-reproducible-environments)
4. [Working with Docker](#working-with-docker)
5. [CI/CD Pipeline Updates](#cicd-pipeline-updates)
6. [Migration FAQs](#migration-faqs)

## Migrating from pip to uv

### Why Migrate to uv?

- *_Speed__: uv is 10-100x faster than pip for dependency resolution
- __Reliability__: Improved dependency resolution and conflict handling
- __Features__: Better support for modern Python packaging standards
- __Lockfiles__: Native support for lockfile generation and updating

### Step-by-Step Migration

1. __Install uv__:

   \`\`\`text\`bash

   ## Unix/Linux/macOS

   curl -LsSf <https://astral.sh/uv/install.sh> | sh

   ## Windows

   powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"


   ````text

   ```text`

   ````

1. __Migrate existing environments__:

   \`\`\`text\`bash

   ## Generate lockfiles from your existing requirements

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock

   ## Create a new environment using uv

   uv venv

   ## Install using lockfiles

   uv pip sync requirements.lock requirements-dev.lock

   ````text

   ```text`

   ````

1. __Use the helper scripts__:

   We've provided convenient script wrappers in `scripts/run_with_uv.sh` (Unix/macOS) and `scripts/run_with_uv.ps1` (Windows).

   \`\`\`text\`bash

   ## Run tests

   ./scripts/run_with_uv.sh test

   ## Run linters

   ./scripts/run_with_uv.sh lint

   ````text

   ```text`

   ````

## Updating Type Annotations for Python 3.10+

Python 3.10 introduced new type annotation syntax. We've provided a helper script to identify type annotations that can be updated:

\`\`\`text\`bash

## Scan the entire project

python scripts/fix_type_annotations.py .

## Scan a specific file

python scripts/fix_type_annotations.py path/to/file.py

````text

For detailed guidance on type compatibility issues and solutions, see the [Type Compatibility Guide](TYPE_COMPATIBILITY.md).

### Common Type Annotation Updates

1. __Union Types__:

   Before (Python 3.9 and earlier):
   ```python
   from typing import Union

   def func(x: Union[int, str]) -> Union[float, None]:

```text

   ...

```text
   ```text

   After (Python 3.10+):
   ```python
   def func(x: int | str) -> float | None:


```text

   ...

```text
   ```text

1. __Optional Types__:

   Before:
   ```python
   from typing import Optional

   def func(x: Optional[int] = None) -> Optional[str]:

```text

   ...

```text
   ```text

   After:
   ```python
   def func(x: int | None = None) -> str | None:


```text

   ...

```text
   ```text

## Using Lockfiles for Reproducible Environments

The project now uses lockfiles to ensure reproducible environments:

1. __Sync your environment__ using the lockfiles:
   ```bash
   uv pip sync requirements.lock requirements-dev.lock
   ```text

1. __Update lockfiles__ when dependencies change:
   ```bash
   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock
   ```text

## Working with Docker

The project includes a Dockerfile optimized for uv:

```bash

## Build and run the development image

docker build --target development -t ai-playground-dev .
docker run -p 5000:5000 -v $(pwd):/app ai-playground-dev

## Build and run the production image

docker build --target production -t ai-playground .
docker run -p 5000:5000 ai-playground

```text

### Benefits of the uv-based Dockerfile

- __Faster builds__: uv's speed dramatically reduces build times
- __Reproducible environments__: Using lockfiles ensures consistent environments
- __Multi-stage builds__: Separate development and production images
- __Smaller images__: Only necessary dependencies are included

## CI/CD Pipeline Updates

The CI/CD pipeline has been updated to use uv for faster and more reliable builds:

1. __Testing across Python versions__: CI tests against Python 3.10, 3.11, and 3.13
2. __Dual testing__: Tests both traditional and modern installation methods
3. __Caching__: Optimized caching of dependencies to speed up CI runs
4. __Markdown linting_*: Automated linting of markdown files

## Migration FAQs

### Q: Do I need to uninstall pip?

A: No. uv works alongside pip and doesn't replace it completely. The helper scripts will install uv if needed.

### Q: Will my existing scripts still work?

A: Yes. We maintain backward compatibility with traditional workflows while offering improved alternatives.

### Q: How do I add a new dependency?

A: Add it to `requirements.txt` or `requirements-dev.txt`, then run:
   ```bash
   uv pip compile requirements.txt --output-file requirements.lock
   ```text

### Q: Can I still use requirements.txt?

A: Yes. We maintain compatibility with requirements.txt while leveraging uv's improved handling.

### Q: Will these changes affect existing installations?

A: No. Users installing via pip will still be able to do so. These changes enhance the development experience without breaking compatibility.

### Q: What if I encounter type checking errors after migration?

A: Use the `scripts/fix_type_annotations.py` script to help identify and fix type annotation issues.

```text`

````
