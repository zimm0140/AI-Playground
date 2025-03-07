#!/bin/bash
# Tool Compatibility Check Script
# This script checks compatibility with different tool versions and generates reports

echo "Checking compatibility with different tool versions..."

# Create a directory for storing compatibility reports
mkdir -p ci_artifacts/tool_compat

# Check Python package versions
python -m pip freeze > ci_artifacts/tool_compat/pip_freeze.txt

# Generate a report for key packages
python -c "
import sys
import platform
import subprocess

# Basic system info
report = [
    '# Tool Compatibility Report',
    '',
    '## System Information',
    f'- Python version: {platform.python_version()}',
    f'- OS: {platform.system()} {platform.release()}',
    '',
    '## Key Package Versions',
]

# Check key packages
packages = ['torch', 'numpy', 'pip', 'setuptools']
for pkg in packages:
    try:
        pkg_ver = subprocess.check_output([sys.executable, '-m', 'pip', 'show', pkg]).decode()
        version = [line for line in pkg_ver.split('\\n') if line.startswith('Version:')][0].split(': ')[1]
        report.append(f'- {pkg}: {version}')
    except:
        report.append(f'- {pkg}: Not installed')

# Write report
with open('ci_artifacts/tool_compat/report.md', 'w') as f:
    f.write('\\n'.join(report))
"

echo "Tool compatibility check complete. Report saved to ci_artifacts/tool_compat/report.md" 