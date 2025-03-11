# PowerShell script to help set up the development environment on Windows

# Stop on first error
$ErrorActionPreference = "Stop"

# Get script directory and repository root
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Get-Item "$ScriptsDir\..\..\").FullName

Write-Host "Setting up development environment..." -ForegroundColor Cyan

# Ensure tomli and tomli_w are installed for the dependency sync script
Write-Host "Installing required packages for dependency sync..." -ForegroundColor Cyan
pip install tomli tomli_w
Write-Host "✓ Installed tomli and tomli_w" -ForegroundColor Green

# Install pre-commit if not installed
if (-not (Get-Command pre-commit -ErrorAction SilentlyContinue)) {
    Write-Host "Installing pre-commit..." -ForegroundColor Cyan
    pip install pre-commit
    Write-Host "✓ Installed pre-commit" -ForegroundColor Green
} else {
    Write-Host "✓ pre-commit already installed" -ForegroundColor Green
}

# Install pre-commit hooks
Write-Host "Installing pre-commit hooks..." -ForegroundColor Cyan
Set-Location $RepoRoot
pre-commit install
Write-Host "✓ Installed pre-commit hooks" -ForegroundColor Green

# Sync dependencies (ensure setup.py and pyproject.toml are in sync)
Write-Host "Syncing dependencies..." -ForegroundColor Cyan
python "$RepoRoot\.github\sync_dependencies.py"
Write-Host "✓ Synced dependencies" -ForegroundColor Green

# Run markdown fixer
Write-Host "Fixing markdown issues..." -ForegroundColor Cyan
python "$ScriptsDir\fix_markdown_lint.py" "$RepoRoot"
Write-Host "✓ Fixed markdown issues" -ForegroundColor Green

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "You can now run tests with: pytest" -ForegroundColor Cyan
Write-Host "Or with Rye (if installed): rye run pytest" -ForegroundColor Cyan
Write-Host "To run pre-commit checks: pre-commit run --all-files" -ForegroundColor Cyan 