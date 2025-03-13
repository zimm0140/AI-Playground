

# Migration Guide for AI Playground {#migration-guide-for-ai-playground}

This guide helps you migrate to the modern Python development workflow using uv and Python 3.10+.

## Table of Contents {#table-of-content {#table-of-contents-table-of-content}

s}

- [Migration Guide for AI Playground {#migration-guide-for-ai-playground}](#migration-guide-for-ai-playground-migration-guide-for-ai-playground)

  - [Table of Contents {#table-of-content {#table-of-contents-table-of-content}](#table-of-contents-table-of-content-table-of-contents-table-of-content)

  - [Migrating from pip to uv {#migrating-from-pip-to {#migrating-from-pip-to-uv-migrating-from-pip-to}](#migrating-from-pip-to-uv-migrating-from-pip-to-migrating-from-pip-to-uv-migrating-from-pip-to)

    - [Why Migrate to uv? {#why-migrate- {#why-migrate-to-uv-why-migrate-}](#why-migrate-to-uv-why-migrate--why-migrate-to-uv-why-migrate-)

    - [Step-by-Step Migration {#step-by-step-mi {#step-by-step-migration-step-by-step-mi}](#step-by-step-migration-step-by-step-mi-step-by-step-migration-step-by-step-mi)

  - [Updating Type Annotations for Python 3.10+ {#updating-type-annotations-for-p {#updating-type-annotations-for-python-310-updating-type-annotations-for-p}](#updating-type-annotations-for-python-310-updating-type-annotations-for-p-updating-type-annotations-for-python-310-updating-type-annotations-for-p)

  - [Scan the entire project {#scan-the-ent {#scan-the-entire-project-scan-the-ent}](#scan-the-entire-project-scan-the-ent-scan-the-entire-project-scan-the-ent)

  - [Scan a specific file {#scan-a- {#scan-a-specific-file-scan-a-}](#scan-a-specific-file-scan-a--scan-a-specific-file-scan-a-)

  - [Working with Docker {# {#working-with-docker-}](#working-with-docker--working-with-docker-)

  - [Build and run the development image {#build-and-run- {#build-and-run-the-development-image-build-and-run-}](#build-and-run-the-development-image-build-and-run--build-and-run-the-development-image-build-and-run-)

  - [Build and run the production image {#build-and-r {#build-and-run-the-production-image-build-and-r}](#build-and-run-the-production-image-build-and-r-build-and-run-the-production-image-build-and-r)

    - [Benefits of the uv-based Dockerfile {#benefits-o {#benefits-of-the-uv-based-dockerfile-benefits-o}](#benefits-of-the-uv-based-dockerfile-benefits-o-benefits-of-the-uv-based-dockerfile-benefits-o)

  - [CI/CD Pipeline Upda {#cicd-pipeline-upda}](#cicd-pipeline-upda-cicd-pipeline-upda)

  - [Mi {#mi}](#mi-mi)

    - [Q: Do I need to uninstall pip?](#q-do-i-need-to-uninstall-pip)

    - [Q: Will my existing scripts still work? {#q-wi {#q-will-my-existing-scripts-still-work-q-wi}](#q-will-my-existing-scripts-still-work-q-wi-q-will-my-existing-scripts-still-work-q-wi)

    - [Q: How do I add a new dependency {#q-how-do-i-add-a-new-dependency}](#q-how-do-i-add-a-new-dependency-q-how-do-i-add-a-new-dependency)

    - [Q: Can I still use requirements.txt {#q-can-i-still-use-requirementstxt}](#q-can-i-still-use-requirementstxt-q-can-i-still-use-requirementstxt)

    - [Q: Will these changes affect existing installations? {#q-will-thes {#q-will-these-changes-affect-existing-installations-q-will-thes}](#q-will-these-changes-affect-existing-installations-q-will-thes-q-will-these-changes-affect-existing-installations-q-will-thes)

    - [Q: What if I encounter type checking errors after migration? {#q-what-if-i-encou {#q-what-if-i-encounter-type-checking-errors-after-migration-q-what-if-i-encou}](#q-what-if-i-encounter-type-checking-errors-after-migration-q-what-if-i-encou-q-what-if-i-encounter-type-checking-errors-after-migration-q-what-if-i-encou)

n)

## Migrating from pip to uv {#migrating-from-pip-to {#migrating-from-pip-to-uv-migrating-from-pip-to}

-uv}

### Why Migrate to uv? {#why-migrate- {#why-migrate-to-uv-why-migrate-}

to-uv}

- **Speed**: uv is 10-100x faster than pip for dependency resolution

- **Reliability**: Improved dependency resolution and conflict handlin

g

- **Features**: Better support for modern Python packaging standar

ds

- **Lockfiles**: Native support for lockfile generation and u

pdating

### Step-by-Step Migration {#step-by-step-mi {#step-by-step-migration-step-by-step-mi}

gration}

1. **Install uv**:

   ```bash

   ## Unix/Linux/macOS

   curl -LsSf https://astral.sh/uv/install.sh | sh

   ## Windows

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"


   ```

1. **Migrate existing environments**:

   ```bash

   ## Generate lockfiles from your existing requirements

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock

   ## Create a new environment using uv

   uv venv

   ## Install using lockfiles

   uv pip sync requirements.lock requirements-dev.lock

   ```

1. **Use the helper scripts**:

   We've provided convenient script wrappers in `scripts/run_with_uv.sh` (Unix/macOS) and `scripts/run_with_uv.ps1` (Windows).

   ```bash

   ## Run tests

   ./scripts/run_with_uv.sh test

   ## Run linters

   ./scripts/run_with_uv.sh lint

   ```

## Updating Type Annotations for Python 3.10+ {#updating-type-annotations-for-p {#updating-type-annotations-for-python-310-updating-type-annotations-for-p}

ython-310}

Python 3.10 introduced new type annotation syntax. We've provided a helper script to identify type annotations that can be updated
:

``
`b
ash

## Scan the entire project {#scan-the-ent {#scan-the-entire-project-scan-the-ent}

ire-project}

python scripts/fix_type_anno
tations.py .

## Scan a specific file {#scan-a- {#scan-a-specific-file-scan-a-}

specific-file}

python scripts/fix_type_annotations.py path/to/file.py

``
`ba
s
h

For detailed guidance on type compatibility issues and solutions, see the [Type Compatibility Guide](TYPE_COMP
ATIBILITY.md).

### Common Type Annotation Updates {#common-type-ann {#common-type-annotation-updates-common-type-ann}

otation-update
s}

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

   ```ba

sh

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

   ``

`bash

   ```bash

## Using Lockfiles for Reproducible Environments {#using-lockfiles-for-reproduc {#using-lockfiles-for-reproducible-environments-using-lockfiles-for-reproduc}

ible-environments}

The project now uses lockfiles to ensure reproducible environmen
ts:

1. **Sync your environment** using the lockfiles:

   ```bash

   uv pip sync requirements.lock requirements-dev.lock

   ```

bash

1. **Update lockfiles** when dependencies change:

   ```bash

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-de

v.lock

   ```bash

## Working with Docker {# {#working-with-docker-}

working-with-docker}

The project includes a Dockerfile optimi
zed for uv:

``
`bash

## Build and run the development image {#build-and-run- {#build-and-run-the-development-image-build-and-run-}

the-development-image}

docker build --target development -t ai-playground-dev .
docker run -p 5000:5000 -v $(pwd):
/app ai-playground-dev

## Build and run the production image {#build-and-r {#build-and-run-the-production-image-build-and-r}

un-the-production-image}

docker build --target production -t ai-playground .
docker run -p 5000:500
0 ai-playground

```b
a

sh

### Benefits of the uv-based Dockerfile {#benefits-o {#benefits-of-the-uv-based-dockerfile-benefits-o}

f-the-uv-based-dockerf
ile}

- **Faster builds**: uv's speed dramatically reduces build t

imes

- **Reproducible environments**: Using lockfiles ensures consistent environ

ments

- **Multi-stage builds**: Separate development and production

images

- **Smaller images**: Only necessary
 dependencies are included

## CI/CD Pipeline Upda {#cicd-pipeline-upda}

tes {#cicd-pipeline-updates}

The CI/CD pipeline has been updated to use uv for faster and more reliable bu
ilds:

1. **Testing across Python versions**: CI tests against Python 3.10, 3.11, an

d 3.13

1. **Dual testing**: Tests both traditional and modern installation

methods

1. **Caching**: Optimized caching of dependencies to speed up
 CI runs

1. __Markdown linting_*: Automat

ed linting of markdown files

## Mi {#mi}

gration FAQs {#migration-faqs}

### Q: Do I need to uninstall pip?

 {#q-do-i-need-to-uninstall-pip}

A: No. uv works alongside pip and doesn't replace it completely. The helper sc
ripts will install uv if needed.

### Q: Will my existing scripts still work? {#q-wi {#q-will-my-existing-scripts-still-work-q-wi}

ll-my-existing-scripts-still-work}
A: Yes. We maintain backward compatibility with traditional workflows whi
le offering improved alternatives.

### Q: How do I add a new dependency {#q-how-do-i-add-a-new-dependency}

? {#q-how-do-i-add-a-new-dependency}

A: Add it to `requirements.txt` or `requirements-dev.txt`, then run:

   ```bash

   uv pip compile requirements.txt --outpu
t-file requirements.lock

   ```bash

### Q: Can I still use requirements.txt {#q-can-i-still-use-requirementstxt}

? {#q-can-i-still-use-requirementstxt}

A: Yes. We maintain compatibility with requirements.txt wh
ile leveraging uv's improved handling.

### Q: Will these changes affect existing installations? {#q-will-thes {#q-will-these-changes-affect-existing-installations-q-will-thes}

e-changes-affect-existing-installations}
A: No. Users installing via pip will still be able to do so. These changes enhance the development ex
perience without breaking compatibility.

### Q: What if I encounter type checking errors after migration? {#q-what-if-i-encou {#q-what-if-i-encounter-type-checking-errors-after-migration-q-what-if-i-encou}

nter-type-checking-errors-after-migration}
A: Use the `scripts/fix_type_annotations.py` script to help identify and fix type annotation issues.

```bash

```

bash
