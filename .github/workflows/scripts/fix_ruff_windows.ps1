#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fixes Ruff issues in the service directory on Windows systems.

.DESCRIPTION
    This script runs Ruff on the service directory to find and fix common linting issues.
    It provides a detailed report of what was fixed and what still needs manual attention.

.EXAMPLE
    .\fix_ruff_windows.ps1

.NOTES
    Requires Ruff to be installed: pip install ruff
#>

function Write-Header {
    param ([string]$Message)
    
    $line = "=" * 80
    Write-Host "`n$line"
    Write-Host (" " + $Message + " ").PadLeft(40 + $Message.Length / 2).PadRight(80, "=")
    Write-Host "$line`n"
}

# Make sure Ruff is installed
try {
    $null = & ruff --version
} catch {
    Write-Host "Ruff is not installed or not in your PATH. Please install it with:" -ForegroundColor Red
    Write-Host "pip install ruff" -ForegroundColor Yellow
    exit 1
}

Write-Header "Ruff Issue Fixer for Windows"

# Find the service directory
$serviceDir = "service"
if (-not (Test-Path $serviceDir)) {
    Write-Host "Service directory not found in current location. Looking elsewhere..." -ForegroundColor Yellow
    
    # Try to find the service directory
    $possibleLocations = @(
        "service",
        "..\service", 
        "WebUI\service",
        (Join-Path $PWD.Path "service")
    )
    
    foreach ($loc in $possibleLocations) {
        if (Test-Path $loc) {
            $serviceDir = $loc
            Write-Host "Found service directory at: $((Resolve-Path $serviceDir).Path)" -ForegroundColor Green
            break
        }
    }
    
    if (-not (Test-Path $serviceDir)) {
        Write-Host "Could not find service directory in any of these locations: $($possibleLocations -join ', ')" -ForegroundColor Red
        exit 1
    }
}

# Find Python files
Write-Header "Finding Python Files"
$pythonFiles = Get-ChildItem -Path $serviceDir -Filter "*.py" -Recurse -File | Select-Object -ExpandProperty FullName
if ($pythonFiles.Count -eq 0) {
    Write-Host "No Python files found in service directory!" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($pythonFiles.Count) Python files" -ForegroundColor Green

# Define common Ruff arguments
$ruffArgs = @(
    "--select=E,F",
    "--ignore=E501",
    "--extend-exclude=.git,.github,.venv,venv,__pycache__,build,dist",
    "--line-length=100"
)

# Check for issues
Write-Header "Checking for Issues"
$checkOutput = & ruff check @ruffArgs --statistics $pythonFiles 2>&1
$checkExitCode = $LASTEXITCODE

if ($checkExitCode -eq 0) {
    Write-Host "`n✅ No issues found! Ruff is happy with your code." -ForegroundColor Green
    exit 0
}

# Show initial issues
Write-Host $checkOutput -ForegroundColor Yellow

# Fix issues
Write-Header "Fixing Issues"
$fixOutput = & ruff check @ruffArgs --fix $pythonFiles 2>&1
Write-Host $fixOutput

# Check again after fixes
Write-Header "Checking Again After Fixes"
$recheckOutput = & ruff check @ruffArgs --statistics $pythonFiles 2>&1
$recheckExitCode = $LASTEXITCODE

if ($recheckExitCode -eq 0) {
    Write-Host "`n✅ All issues fixed!" -ForegroundColor Green
    exit 0
}

# Try more specific fixes
Write-Header "Trying More Specific Fixes"

# Fix unused imports
Write-Host "`n📌 Fixing unused imports (F401)..." -ForegroundColor Cyan
$unusedImportsOutput = & ruff check --select=F401 --ignore=E501 --line-length=100 --fix $pythonFiles 2>&1

# Fix other formatting issues
Write-Host "`n📌 Fixing formatting issues (E)..." -ForegroundColor Cyan
$formattingOutput = & ruff check --select=E --ignore=E501 --line-length=100 --fix $pythonFiles 2>&1

# Final check
Write-Header "Final Check"
$finalOutput = & ruff check @ruffArgs --statistics $pythonFiles 2>&1
$finalExitCode = $LASTEXITCODE

if ($finalExitCode -eq 0) {
    Write-Host "`n✅ All issues fixed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠️ Some issues still need manual attention" -ForegroundColor Yellow
    
    # Show remaining issues in a more readable way
    Write-Header "Issues Needing Manual Attention"
    $detailedOutput = & ruff check @ruffArgs --format=text $pythonFiles 2>&1
    Write-Host $detailedOutput -ForegroundColor Yellow
    
    # Suggestions for manual fixes
    Write-Header "Suggestions for Manual Fixes"
    Write-Host @"
Common issues that need manual attention:

1. Unused imports (F401):
   - Remove unused imports from the top of the file
   - Or add '# noqa: F401' at the end of the import line if needed

2. Undefined names (F821):
   - Make sure variables are defined before use
   - Check for typos in variable names

3. Missing whitespace (E2xx):
   - Add spaces around operators
   - Add spaces after commas in lists/dicts
"@ -ForegroundColor Cyan
    
    exit 1
} 