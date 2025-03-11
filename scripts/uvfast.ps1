# Convenience wrapper script for uvfast.py on Windows
# PowerShell script

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Change to project root directory
Set-Location -Path $ProjectRoot

# Check if uvfast.py exists
if (-not (Test-Path "uvfast.py")) {
    Write-Error "Error: uvfast.py not found in $ProjectRoot"
    exit 1
}

# Pass all arguments to uvfast.py
python uvfast.py $args 