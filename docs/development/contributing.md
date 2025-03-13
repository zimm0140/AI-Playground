
# Contributing to AI Playground {#contributing-to-ai-playground}

First off, thank you for considering contributing to AI Playground. It's people like you that make this project such a great tool. We welcome contributions from everyone as long
as they follow the guidelines below.

## Contributing Guidelines {#contributing-guidelines}

Thank you for your interest in contributing to AI-Playground! This document outlines the environment setup process and best practices.

## Development Environment Setup {#development-environment-setup}

### Option 1: Using Conda (Recommended) {#option-1-using-conda-recommended}

1. *_Install Conda__:

   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

1. **Create and activate the environment**:

\`\`\`text\`bash

## Create a new Conda environment {#create-a-new-conda-environment}

conda create -n ai-playground-env python=3.9 -y

## Activate the environment {#activate-the-environment}

conda activate ai-playground-env

## Install required packages {#install-required-packages}

pip install -r requirements.txt

```text`text

1. **Verify the environment**:

```bash

## Check w

hich Python is being used (should point to your conda environment) {#check-which-python-is-being-used-should-point-to-your-conda-environment}

python -c "import sys; print(sys.executable)"

## Test that jsonschema is installed {#test-that-jsonschema-is-installed}

python -c "import jsonschema; print(f'jsonschema version: {jsonschema.**version**}')"

```text

### Option 2: Using

 venv {#option-2-using-venv}

1. **Create and activate the environment**:

```bash

## On Wi

ndows {#on-windows}

python -m venv .venv
.\.venv\Scripts\activate

## On macOS/Linux {#on-macoslinux}

python -m venv .venv
source .venv/bin/activate

## Install required packages {#install-required-packages}

pip install -r requirements.txt

```text

## Environment Ma

nagement Best Practices {#environment-management-best-practices}

1. **Always activate your environment before working on the project**:

```bash

## For

 Conda {#for-conda}

conda activate ai-playground-env

## For venv on Windows {#for-venv-on-windows}

.\.venv\Scripts\activate

## For venv on macOS/Linux {#for-venv-on-macoslinux}

source .venv/bin/activate

```text
1. **Add new de
pendencies to requirements.txt**:

When adding a new package, update the requirements.txt file:

```bash

## A

fter installing a new package {#after-installing-a-new-package}

pip freeze > requirements.txt

## Or manually add it with a specific version {#or-manually-add-it-with-a-specific-version}

echo "package-name==1.2.3" >> requirements.txt

```text
1. __Never co
mmit environment directories_*:

The .gitignore file is set up to exclude environment directories (.venv/, env/, etc.).
Do not manually commit these directories.

## Validation Workflow {#validation-workflow}

To validate JSON files against schemas:

```bash

##
 Ensure you're in your activated environment {#ensure-youre-in-your-activated-environment}

python validate_colorize.py

```text

## Addition

al Guidelines {#additional-guidelines}

## Code of Conduct {#code-of-conduct}

AI Playground has adopted a Code of Conduct that we expect project participants to adhere to. Please read [the full text](./CODE_OF_CONDUCT.md) so that you can understand what
actions will and will not be tolerated.

## How to Contribute {#how-to-contribute}

There are many ways to contibute to this project, from writing tutorials or blog posts ([Show and tell in Discussion](https://github.com/intel/AI-Playground/discussions)),
improving the documentation, submitting bug reports and feature requests, or writing code which can be incorporated into this project itself.

### Reporting Bugs {#reporting-bugs}

Before submitting bug reports, please check the [issue](https://github.com/intel/AI-Playground/issues) tracker to make sure the bug hasn't been reported before. If it is a new bug,
please provide as much detail as possible, including:

- A clear and descriptive title

- The exact steps which reproduce the problem

- Your environment (OS, GPU, CPU, etc.)

- Any related files or screenshots

### Suggesting Enhancements {#suggesting-enhancements}

Enhancement suggestions (feature requests) are also tracked as GitHub [issues](https://github.com/intel/AI-Playground/issues). When suggesting an enhancement, please:

- Use a clear and descriptive title

- Provide a step-by-step description of the suggested enhancement

- Explain why this enhancement would be useful to most AI Playground users

### Pull Requests {#pull-requests}

If you are not familiar with pull requests, you could get started with GitHub tutorials: [Collaborate with pull requests](https://docs.github.
com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models).

The process described here has several goals:

- Maintain the quality of this project

- Fix problems that are important to users

- Engage the community in working toward the best possible AI Playground

- Enable a sustainable system for AI Playground's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Always set base branch to [dev](https://github.com/intel/AI-Playground/tree/dev), do NOT make pull requests to the main branch without a strong reason.

1. Follow all instructions in [the template](/.github/pull_request_template.md).

1. After you submit your pull request, verify that all [status checks](https://docs.github.

com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks) are passing.

1. [Sign your work](/CONTRIBUTING.md#sign-your-work).

While the prerequisites above must be satifisfied prior to have your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes
before your pull request can be ultimately accepted.

## Community {#community}

Discussions about AI Playground take place on this repository's [Issues](https://github.com/intel/AI-Playground/issues), [Pull Requests](https://github.
com/intel/AI-Playground/pulls) and [Discussions](https://github.com/intel/AI-Playground/discussions). Anybody is welcome to join these conversations.

## License {#license}

AI Playground is licensed under the terms in [LICENSE](/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your
contribution under these terms.

## Sign your work {#sign-your-work}

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The
rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```text
Develo
per Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I

```text
have the
right to submit it under the open source license
indicated in the file; or

```text
(b)
The contribution is based upon previous work that, to the best

```text
of my k
nowledge, is covered under an appropriate open source
license and I have the right under that license to submit that
work with modifications, whether created in whole or in part
by me, under the same open source license (unless I am
permitted to submit under a different license), as indicated
in the file; or

```text
(c
) The contribution was provided directly to me by some other

```text
perso
n who certified (a), (b) or (c) and I have not modified
it.

```text
(d) I understand and agree that this project and the contribution

```text
are
 public and that a record of the contribution (including all
personal information I submit with it, including my sign-off) is
maintained indefinitely and may be redistributed consistent with
this project or the open source license(s) involved.

``
`text

```text
T
hen you just add a line to every git commit message:

```text
Signed-off-by: Joe Smith <joe.smith@email.com>

```text
Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

-----

Again, thanks for your interest in contributing to this project. We appreciate your efforts to make our project even better!

```text`

```text`
