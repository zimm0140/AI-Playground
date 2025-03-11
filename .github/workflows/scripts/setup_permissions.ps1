# PowerShell script to help set up the development environment on Windows

# Stop on first error
$ErrorActionPreference = "Stop"

# Get script directory and repository root
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Get-Item "$ScriptsDir\..\..\").FullName

Write-Host "Setting up development environment..." -ForegroundColor Cyan

# Install uv if not already installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (recommended)..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "✓ Installed uv" -ForegroundColor Green
} else {
    Write-Host "✓ uv already installed" -ForegroundColor Green
}

# Ensure tomli and tomli_w are installed for the dependency sync script
Write-Host "Installing required packages for dependency sync..." -ForegroundColor Cyan
uv pip install tomli tomli_w
Write-Host "✓ Installed tomli and tomli_w" -ForegroundColor Green

# Install pre-commit if not installed
if (-not (Get-Command pre-commit -ErrorAction SilentlyContinue)) {
    Write-Host "Installing pre-commit..." -ForegroundColor Cyan
    uv pip install pre-commit
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
Write-Host "You can now run tests with:" -ForegroundColor Cyan
Write-Host "  uv run pytest" -ForegroundColor Green
Write-Host "Or using traditional method:" -ForegroundColor Cyan
Write-Host "  pytest" -ForegroundColor Green
Write-Host "To run pre-commit checks: pre-commit run --all-files" -ForegroundColor Cyan 