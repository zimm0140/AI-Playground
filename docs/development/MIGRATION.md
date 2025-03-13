
# Migration Guide for AI Playground

This guide helps you migrate to the modern Python development workflow using uv and Python 3.10+.

## Table of Contents

- [Migration Guide for AI Playground](#migration-guide-for-ai-playground)
    - [Table of Contents](#table-of-contents)
    - [Migrating from pip to uv {#migrating-from-pip-to-uv}](#migrating-from-pip-to-uv-migrating-from-pip-to-uv)
        - [Why Migrate to uv?](#why-migrate-to-uv)
        - [Step-by-Step Migration](#step-by-step-migration)
    - [Unix/Linux/macOS](#unixlinuxmacos)
    - [Windows](#windows)
    - [Generate lockfiles from your existing requirements](#generate-lockfiles-from-your-existing-requirements)
    - [Create a new environment using uv](#create-a-new-environment-using-uv)
    - [Install using lockfiles](#install-using-lockfiles)
    - [Run tests](#run-tests)
    - [Run linters](#run-linters)
    - [Updating Type Annotations for Python 3.10+ {#updating-type-annotations-for-python-310}](#updating-type-annotations-for-python-310-updating-type-annotations-for-python-310)
    - [Scan the entire project {#scan-the-entire-project}](#scan-the-entire-project-scan-the-entire-project)
    - [Scan a specific file {#scan-a-specific-file}](#scan-a-specific-file-scan-a-specific-file)
        - [Common Type Annotation Updates](#common-type-annotation-updates)
    - [Using Lockfiles for Reproducible Environments {#using-lockfiles-for-reproducible-environments}](#using-lockfiles-for-reproducible-environments-using-lockfiles-for-reproducible-environments)
    - [Working with Docker {#working-with-docker}](#working-with-docker-working-with-docker)
    - [Build and run the development image {#build-and-run-the-development-image}](#build-and-run-the-development-image-build-and-run-the-development-image)
    - [Build and run the production image {#build-and-run-the-production-image}](#build-and-run-the-production-image-build-and-run-the-production-image)
        - [Benefits of the uv-based Dockerfile](#benefits-of-the-uv-based-dockerfile)
    - [CI/CD Pipeline Updates {#cicd-pipeline-updates}](#cicd-pipeline-updates-cicd-pipeline-updates)
    - [Migration FAQs {#migration-faqs}](#migration-faqs-migration-faqs)
        - [Q: Do I need to uninstall pip?](#q-do-i-need-to-uninstall-pip)
        - [Q: Will my existing scripts still work?](#q-will-my-existing-scripts-still-work)
        - [Q: How do I add a new dependency?](#q-how-do-i-add-a-new-dependency)
        - [Q: Can I still use requirements.txt?](#q-can-i-still-use-requirementstxt)
        - [Q: Will these changes affect existing installations?](#q-will-these-changes-affect-existing-installations)
        - [Q: What if I encounter type checking errors after migration?](#q-what-if-i-encounter-type-checking-errors-after-migration)


## Migrating from pip to uv {#migrating-from-pip-to-uv}

### Why Migrate to uv?

- **Speed**: uv is 10-100x faster than pip for dependency resolution

- **Reliability**: Improved dependency resolution and conflict handling

- **Features**: Better support for modern Python packaging standards

- **Lockfiles**: Native support for lockfile generation and updating

### Step-by-Step Migration

1. **Install uv**:

   ```bash


   ## Unix/Linux/macOS

   curl -LsSf https://astral.sh/uv/install.sh | sh

   ## Windows

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   ```text

1. **Migrate existing environments**:

   ```bash


   ## Generate lockfiles from your existing requirements

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock

   ## Create a new environment using uv

   uv venv

   ## Install using lockfiles

   uv pip sync requirements.lock requirements-dev.lock

   ```text

1. **Use the helper scripts**:

   We've provided convenient script wrappers in `scripts/run_with_uv.sh` (Unix/macOS) and `scripts/run_with_uv.ps1` (Windows).

   ```bash


   ## Run tests

   ./scripts/run_with_uv.sh test

   ## Run linters

   ./scripts/run_with_uv.sh lint

   ```text

## Updating Type Annotations for Python 3.10+ {#updating-type-annotations-for-python-310}

Python 3.10 introduced new type annotation syntax. We've provided a helper script to identify type annotations that can be updated:

```bash


## Scan the entire project {#scan-the-entire-project}

python scripts/fix_type_annotations.py .

## Scan a specific file {#scan-a-specific-file}

python scripts/fix_type_annotations.py path/to/file.py

```bash

For detailed guidance on type compatibility issues and solutions, see the [Type Compatibility Guide](TYPE_COMPATIBILITY.md).

### Common Type Annotation Updates

1. **Union Types**:

   Before (Python 3.9 and earlier):

   ```python

   from typing import Union

   def func(x: Union[int, str]) -> Union[float, None]:

   ```bash

   ...

   ```bash


   ```bash

   After (Python 3.10+):

   ```python

   def func(x: int | str) -> float | None:

   ```bash

   ...

   ```bash


   ```bash

1. **Optional Types**:

   Before:

   ```python

   from typing import Optional

   def func(x: Optional[int] = None) -> Optional[str]:

   ```bash

   ...

   ```bash


   ```bash

   After:

   ```python

   def func(x: int | None = None) -> str | None:

   ```bash

   ...

   ```bash


   ```bash

## Using Lockfiles for Reproducible Environments {#using-lockfiles-for-reproducible-environments}

The project now uses lockfiles to ensure reproducible environments:

1. **Sync your environment** using the lockfiles:

   ```bash

   uv pip sync requirements.lock requirements-dev.lock

   ```bash

1. **Update lockfiles** when dependencies change:

   ```bash

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock

   ```bash

## Working with Docker {#working-with-docker}

The project includes a Dockerfile optimized for uv:

```bash


## Build and run the development image {#build-and-run-the-development-image}

docker build --target development -t ai-playground-dev .
docker run -p 5000:5000 -v $(pwd):/app ai-playground-dev

## Build and run the production image {#build-and-run-the-production-image}

docker build --target production -t ai-playground .
docker run -p 5000:5000 ai-playground

```bash

### Benefits of the uv-based Dockerfile

- **Faster builds**: uv's speed dramatically reduces build times

- **Reproducible environments**: Using lockfiles ensures consistent environments

- **Multi-stage builds**: Separate development and production images

- **Smaller images**: Only necessary dependencies are included

## CI/CD Pipeline Updates {#cicd-pipeline-updates}

The CI/CD pipeline has been updated to use uv for faster and more reliable builds:

1. **Testing across Python versions**: CI tests against Python 3.10, 3.11, and 3.13


1. **Dual testing**: Tests both traditional and modern installation methods


1. **Caching**: Optimized caching of dependencies to speed up CI runs


1. __Markdown linting_*: Automated linting of markdown files

## Migration FAQs {#migration-faqs}

### Q: Do I need to uninstall pip?

A: No. uv works alongside pip and doesn't replace it completely. The helper scripts will install uv if needed.

### Q: Will my existing scripts still work?

A: Yes. We maintain backward compatibility with traditional workflows while offering improved alternatives.

### Q: How do I add a new dependency?

A: Add it to `requirements.txt` or `requirements-dev.txt`, then run:

   ```bash

   uv pip compile requirements.txt --output-file requirements.lock

   ```bash

### Q: Can I still use requirements.txt?

A: Yes. We maintain compatibility with requirements.txt while leveraging uv's improved handling.

### Q: Will these changes affect existing installations?

A: No. Users installing via pip will still be able to do so. These changes enhance the development experience without breaking compatibility.

### Q: What if I encounter type checking errors after migration?

A: Use the `scripts/fix_type_annotations.py` script to help identify and fix type annotation issues.

```bash

```bash
