
# Linting Guide {#linting-guide}

This guide outlines the linting practices used in the AI-Playground project to maintain code quality and consistency.

## Linting Tools {#linting-tools}

AI-Playground primarily uses the following linting tools:

- *_[Ruff](https://github.com/astral-sh/ruff)__: A fast Python linter that combines multiple linting tools

- **[mypy](https://mypy.readthedocs.io/)**: A static type checker for Python

- **[markdownlint](https://github.com/DavidAnson/markdownlint)**: A linter for Markdown files

## Python Linting Configuration {#python-linting-configuration}

### Ruff Configuration {#ruff-configuration}

The project uses Ruff with the following settings:

\`\`\`text\`toml

## in pyproject.toml {#in-pyprojecttoml}

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "F", "I", "W", "N", "B", "C4", "UP", "T20"]
ignore = ["E501"]
extend-exclude = \[".git", ".github", ".venv", "venv", "**pycache**", "build", "dist"\]

````text

#### Key Rules {#key-rules}

- **E**: Style errors (from pycodestyle)

- **F**: Logical/syntax errors and undefined names (from Pyflakes)

- **I**: Import sorting (from isort)

- **W**: Warnings (from pycodestyle)

- **N**: Naming conventions (from pep8-naming)

- **B**: Bug detection (from flake8-bugbear)

- **C4**: Comprehension complexity (from flake8-comprehensions)

- **UP**: Python upgrade suggestions (from pyupgrade)

- **T20**: Print statement detection (from flake8-print)

### Type Checking with mypy {#type-checking-with-mypy}

For static type checking, we use mypy with these settings:

```toml

## in pyproject.toml {#in-pyprojecttoml}

[too
l.mypy
]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
strict_optional = true

```

## Running Linters Locally {#running-linters-locally}

### Us {#us}

ing the Provided Scripts {#using-the-provided-scripts}

1. For Windows users:

   ```powershell

   .\.github\workflows\scripts\fix_ruff_windows.ps1

   ```

1. For Linux/Mac users:

   ```bash

   python .github/workflows/scripts/fix_ruff_issues_local.py

   ```

### Manual Linting {#manual-linting}

To run Ruff manually:

```bash

## Install Ruff {#install-ruff}

pip install
 ruff

## Check for issues {#check-for-issues}

ruff check .

## Fix issues automatically {#fix-issues-automatically}

ruff check --fix .

```
To run mypy:

```bash

## Install mypy {#install-myp

y}

pi
p inst
all myp
y

## Run type checking {#run-type-checking}

mypy .

```
To run markdownlint on Markdown files:

```bash

#

# Inst {#inst}

all markdownlint (requi
res Node.js) {#i

nstall-markdownlint-requires-nodejs}

npm install -g markdownlint-cli

## Check Markdown files {#check-markdown-files}

markdownlint "__/_.md"

```

## Common Linting Issues and Fixes {#common-linti

ng-iss {#common-linting-issues-and-fixes-common-linting-iss}

ues-and-fixes}

### Unused Imports (F401) {#unused-imports-f401}

An import that's not used in the file:

```python

import os  # Unused import

```
*_Fix
__: Ei
ther remove the import or add a `#
noqa: F

401` comment if it's needed for side effects:

```python

import os  # noqa: F401

```

### M

issing {#missing}

 Whitespace (E2xx) {#
missing-whitespace-
e2xx}

Missing spaces around operators or after commas:

```python

x=1+2  # Missing spaces

def func(
a,b):
  # Missing space after comma

```
**Fix**: Add appropriate spacing:

```python

x =
 1 + 2  # Correct spacing

d
ef fun
c(a, b):  # Space after comma

```

### Type Annotation Issues {#type-annotati

on-is {#type-annotation-issues-type-annotation-is}

sues}

Missing or incorrect type annotations:

```python

def process_data(data):  # Mis

sing

type annotations

```
return data + 1

```

```
**Fix**: A
dd pr
oper type
annota
tions
:

```p
ython

def process_data(data: in
t) ->
int:

```
return data + 1

```

```

### H

ardwa {#hardwa}

re-Specific
Impo
rt Issues {#h

ardware-specific-import-issues}

Importing hardware-specific modules that might not be available:

```python

import intel_extension
_for_
pytorch  # May not be available on all systems

```
**Fix**: Use conditional imports
:

``
`python

try:

```
import intel_ex
tension_for
_pytorch

HAS_INTEL_EXTENSION = True

```
except ImportError:

```
HAS_I

NTEL_EXTENSION = Fa
lse

```t
ext

```

## CI Integration

{#ci

-integration}

{#ci-integrat

ion}

The project's CI system uses GitHub Actions to run linters on all files. The configuration is maintained in the following files:

- `.github/workflows/ruff-integration.yml` (for Ruff)

- `.github/workflows/type-check.yml` (for mypy)

- `.github/workflows/docs-check.yml` (for markdownlint)

The CI will:

1. Check for linting issues

1. Generate a report

1. Comment on PRs if issues are found

1. Provide instructions for fixing the issues

## Pre-commit Hooks {#pre-commit-hooks}

To ensure code quality before committing, you can set up pre-commit hooks locally:

```bash

## On Linux/macOS/Gi {#on-

linuxmacosgi}

t Bash {#on-linuxmacosgit-bash}

./.github/setup-hooks.sh

## On Windows PowerShell {#on-windows-powershell}

.\.github\setup-hooks.ps1

```
This will check your
code for
 linting issues before each commit.

## Temporary Disabling of Linter Rules {#temporary-disabling-of-linter-rules}

There are cases where linter rules need to be temporarily disabled:

```python

## In situations {#in-si

tuations}

 where a line is necessarily long {#in-situations-where-a-line-is-necessarily-long}

long_url = "<https://very-long-url-that-cannot-be-split.com/path/to/resource">  # noqa: E501

## When using a variable name that doesn't match conventions {#when-using-a-variable-name-that-doesnt-match-conventions}

def connect_to_API():  # noqa: N802

```
pass

```

```
Use `# n

oqa:`

comments sp
aringly and only when necessary.

## Hardware-Specific Linting Considerations {#hardware-specific-linting-considerations}

When writing hardware-specific code:

1. Use conditional imports for hardware-specific dependencies

1. Consider using feature checking rather than relying on specific hardware

1. Add appropriate comments where hardware specifics affect code structure

1. Use type annotations that reflect hardware-specific considerations

```python

def optimize_
for_ha
rdware(model: torch.nn.Module, hardware_type: str) -> torch.nn.Module:

```
"""
Optimize mo
del for
specific hardware.

```

```
Args:

``
`
m

odel: The
 PyTor
ch model
hardware_type: One of "acm", "bmg", or "base"

```

```
```
Returns

:

```
Opt
imize
d mo
del

```
"""
if hardw
are
_type == "acm":

```
try:

`
``
import int
el_
extension_for_pytorch as ipex  # noqa: F401

```

```
```
`

``

```

```
m

odel =
 ipex.optimize(model)

```
ex
c
ept ImportError:

```
pass
 # Fall back to unoptimized model

```t
ext

``
`
``

`text

```
r

eturn model

```

```

## Additional Resources {#additional-resources}

- [Ruff Documentation](https://docs.astral.sh/ruff/)

- [mypy Documentation](https://mypy.readthedocs.io/)

- [markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)

- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

- [Code Quality Standards](code-quality.md)

---
**Previous**: [Testing Guide](testing.md) | **Next**: [Project Architecture](../architecture/overview.md) | __See also_*: [Code Quality Standards](code-quality.md)

````

````