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

- **Speed**: uv is 10-100x faster than pip for dependency resolution
- **Reliability**: Improved dependency resolution and conflict handling
- **Features**: Better support for modern Python packaging standards
- **Lockfiles**: Native support for lockfile generation and updating

### Step-by-Step Migration

1. **Install uv**:
   ```bash
   # Unix/Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Migrate existing environments**:
   ```bash
   # Generate lockfiles from your existing requirements
   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock
   
   # Create a new environment using uv
   uv venv
   
   # Install using lockfiles
   uv pip sync requirements.lock requirements-dev.lock
   ```

3. **Use the helper scripts**:
   
   We've provided convenient script wrappers in `scripts/run_with_uv.sh` (Unix/macOS) and `scripts/run_with_uv.ps1` (Windows).
   
   ```bash
   # Run tests
   ./scripts/run_with_uv.sh test
   
   # Run linters
   ./scripts/run_with_uv.sh lint
   ```

## Updating Type Annotations for Python 3.10+

Python 3.10 introduced new type annotation syntax. We've provided a helper script to identify type annotations that can be updated:

```bash
# Scan the entire project
python scripts/fix_type_annotations.py .

# Scan a specific file
python scripts/fix_type_annotations.py path/to/file.py
```

### Common Type Annotation Updates

1. **Union Types**:
   
   Before (Python 3.9 and earlier):
   ```python
   from typing import Union
   
   def func(x: Union[int, str]) -> Union[float, None]:
       ...
   ```
   
   After (Python 3.10+):
   ```python
   def func(x: int | str) -> float | None:
       ...
   ```

2. **Optional Types**:
   
   Before:
   ```python
   from typing import Optional
   
   def func(x: Optional[int] = None) -> Optional[str]:
       ...
   ```
   
   After:
   ```python
   def func(x: int | None = None) -> str | None:
       ...
   ```

## Using Lockfiles for Reproducible Environments

The project now uses lockfiles to ensure reproducible environments:

1. **Sync your environment** using the lockfiles:
   ```bash
   uv pip sync requirements.lock requirements-dev.lock
   ```

2. **Update lockfiles** when dependencies change:
   ```bash
   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock
   ```

## Working with Docker

The project includes a Dockerfile optimized for uv:

```bash
# Build and run the development image
docker build --target development -t ai-playground-dev .
docker run -p 5000:5000 -v $(pwd):/app ai-playground-dev

# Build and run the production image
docker build --target production -t ai-playground .
docker run -p 5000:5000 ai-playground
```

### Benefits of the uv-based Dockerfile

- **Faster builds**: uv's speed dramatically reduces build times
- **Reproducible environments**: Using lockfiles ensures consistent environments
- **Multi-stage builds**: Separate development and production images
- **Smaller images**: Only necessary dependencies are included

## CI/CD Pipeline Updates

The CI/CD pipeline has been updated to use uv for faster and more reliable builds:

1. **Testing across Python versions**: CI tests against Python 3.10, 3.11, and 3.13
2. **Dual testing**: Tests both traditional and modern installation methods
3. **Caching**: Optimized caching of dependencies to speed up CI runs
4. **Markdown linting**: Automated linting of markdown files

## Migration FAQs

**Q: Do I need to uninstall pip?**
A: No. uv works alongside pip and doesn't replace it completely. The helper scripts will install uv if needed.

**Q: Will my existing scripts still work?**
A: Yes. We maintain backward compatibility with traditional workflows while offering improved alternatives.

**Q: How do I add a new dependency?**
A: Add it to `requirements.txt` or `requirements-dev.txt`, then run:
   ```bash
   uv pip compile requirements.txt --output-file requirements.lock
   ```

**Q: Can I still use requirements.txt?**
A: Yes. We maintain compatibility with requirements.txt while leveraging uv's improved handling.

**Q: Will these changes affect existing installations?**
A: No. Users installing via pip will still be able to do so. These changes enhance the development experience without breaking compatibility.

**Q: What if I encounter type checking errors after migration?**
A: Use the `scripts/fix_type_annotations.py` script to help identify and fix type annotation issues.
