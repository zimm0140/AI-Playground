# PowerShell script to run commands with uv
# Helper script to run commands with uv on Windows systems

# Set error handling
$ErrorActionPreference = "Stop"

# Ensure uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing..." -ForegroundColor Yellow
    Invoke-Expression "powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'"
}

# Function to display help
function Show-Help {
    Write-Host "Usage: .\run_with_uv.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  run <script.py>        Run a Python script in an isolated environment" -ForegroundColor Cyan
    Write-Host "  test                   Run pytest" -ForegroundColor Cyan
    Write-Host "  lint                   Run linters (ruff, mypy)" -ForegroundColor Cyan
    Write-Host "  format                 Format code with ruff" -ForegroundColor Cyan
    Write-Host "  sync                   Sync dependencies from lockfiles" -ForegroundColor Cyan
    Write-Host "  audit                  Run security audit" -ForegroundColor Cyan
    Write-Host "  tool <tool_name>       Install and run a tool" -ForegroundColor Cyan
    Write-Host "  clean                  Clean temporary files" -ForegroundColor Cyan
    Write-Host "  help                   Show this help message" -ForegroundColor Cyan
    Write-Host ""
    Exit 0
}

# Check if no arguments were passed
if ($args.Count -eq 0) {
    Show-Help
}

# Process commands
switch ($args[0]) {
    "run" {
        if ($args.Count -lt 2) {
            Write-Host "Error: No script specified." -ForegroundColor Red
            Write-Host "Usage: .\run_with_uv.ps1 run <script.py>"
            Exit 1
        }
        Write-Host "Running $($args[1]) with uv..." -ForegroundColor Green
        uv run $args[1]
    }
    "test" {
        $testArgs = $args | Select-Object -Skip 1
        Write-Host "Running tests with uv..." -ForegroundColor Green
        uv run pytest $testArgs
    }
    "lint" {
        Write-Host "Running linters with uv..." -ForegroundColor Green
        uv run ruff check .
        uv run mypy .
    }
    "format" {
        Write-Host "Formatting code with uv..." -ForegroundColor Green
        uv run ruff format .
    }
    "sync" {
        Write-Host "Syncing dependencies with uv..." -ForegroundColor Green
        uv pip sync requirements.lock requirements-dev.lock
    }
    "audit" {
        Write-Host "Running security audit with uv..." -ForegroundColor Green
        uv pip audit
    }
    "tool" {
        if ($args.Count -lt 2) {
            Write-Host "Error: No tool specified." -ForegroundColor Red
            Write-Host "Usage: .\run_with_uv.ps1 tool <tool_name>"
            Exit 1
        }
        $toolArgs = $args | Select-Object -Skip 1
        Write-Host "Running tool with uv: $toolArgs" -ForegroundColor Green
        uv tool run $toolArgs
    }
    "clean" {
        Write-Host "Cleaning temporary files..." -ForegroundColor Green
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build/, dist/, *.egg-info/, .pytest_cache/, .ruff_cache/
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
        Get-ChildItem -Path . -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown command: $($args[0])" -ForegroundColor Red
        Show-Help
    }
} 