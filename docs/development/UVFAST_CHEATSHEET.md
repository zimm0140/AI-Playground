
# uvfast Command Cheat Sheet {#uvfast-command-cheat-sheet}

This document provides a quick reference for common `uvfast.py` commands.

## Basic Commands {#basic-commands}

### Setup Environment {#setup-environment}

\`\`\`text\`bash

## Auto-detect hardware and set up environment {#auto-detect-hardware-and-set-up-environment}

python uvfast.py setup

## With development dependencies {#with-development-dependencies}

python uvfast.py setup --dev

## For specific hardware {#for-specific-hardware}

python uvfast.py setup --hardware acm
python uvfast.py setup --hardware ovino
python uvfast.py setup --hardware mtl

## Skip lockfile generation/usage {#skip-lockfile-generationusage}

python uvfast.py setup --no-lockfile

````

### Show Environment Information {#show-environment-information}

```bash

## Di

s {#dis}

play hardware detection and environment info {#display-hardware-detection-and-environment-info}

python uvfast.py info

```

### Generate L

o {#generate-lo}

ckfiles {#generate-lockfiles}

```bash

##

G {#g}

enerate lockfiles for all hardware types {#generate-lockfiles-for-all-hardware-types}

python uvfast.py lockfiles --all

## Generate for specific hardware {#generate-for-specific-hardware}

python uvfast.py lockfiles --hardware acm
python uvfast.py lockfiles --hardware acm --dev

```

### Run Comm

a {#run-comma}

nds {#run-commands}

```bash

#

#
 Run pytest {#run-pytest}

python uvfast.py run pytest

## Run with specific test path {#run-with-specific-test-path}

python uvfast.py run pytest tests/test_api.py

## Run linting {#run-linting}

python uvfast.py run ruff check .
python uvfast.py run ruff format .

## Run type checking {#run-type-checking}

python uvfast.py run mypy

## Run any command {#run-any-command}

python uvfast.py run python -m your_module

```

### Legacy

 {#legacy}

Installation {#legacy-installation}

```bash

## Install using traditional approach but with uv speed {#install-using-traditional-approach-but-with-uv-speed}

python uvfast.py legacy-install --dev

```

## Using

 {#using}

Wrapper Scripts {#using-wrapper-scripts}

### Linux/macOS {#linuxmacos}

```bash

## Make script executable {#make-script-executable}

chmod +x scripts/uvfast.sh

## Run commands through the wrapper {#run-commands-through-the-wrapper}

./scripts/uvfast.sh setup --dev
./scripts/uvfast.sh run pytest

```

### Wi

n {#win}

dows {#windows}

```po
w
ershell

## Run commands through the PowerShell wrapper {#run-commands-through-the-powershell-wrapper}

.\scripts\uvfast.ps1 setup --dev
.\scripts\uvfast.ps1 run pytest

```

## H

a {#ha}

rdware-Specific Tips {#hardware-specific-tips}

### Intel Arc GPUs (acm) {#intel-arc-gpus-acm}

```
b
ash

## Set up for Arc GPUs {#set-up-for-arc-gpus}

python uvfast.py setup --hardware acm --dev

## Run GPU-specific tests {#run-gpu-specific-tests}

python uvfast.py run pytest tests/hardware/test_gpu.py

```

##

#
 OpenVINO (ovino) {#openvino-ovino}

``
`bash

## Set up for OpenVINO {#set-up-for-openvino}

python uvfast.py setup --hardware ovino --dev

## Run OpenVINO-specific tests {#run-openvino-specific-tests}

python uvfast.py run pytest tests/openvino/

``
`

#

# Configuration {#configuration}

Edit `uvfast.json` to customize settings:

```json

{

```
"hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino"],
"lockfiles_dir": ".lockfiles",
"venv_dir": ".venv",
"cache_dir": ".uvcache",
"parallel_jobs": 4

```
}

````

````
