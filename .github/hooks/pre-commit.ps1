# PowerShell pre-commit hook for Windows users
# Save this file to .git/hooks/pre-commit.ps1 and call it from pre-commit hook

Write-Host "Running pre-commit checks..."

# Get the list of staged Python files
$StagedFiles = git diff --cached --name-only --diff-filter=d
$StagedPyFiles = $StagedFiles | Where-Object { $_ -match '\.py$' }

if (-not $StagedPyFiles) {
    Write-Host "No Python files to check, skipping linting."
    exit 0
}

# Run the linter on the staged Python files
$ErrorCount = 0
try {
    # Convert array to space-separated string
    $FilesArg = $StagedPyFiles -join ' '
    
    # Run the lint script
    python .github/workflows/scripts/lint_python_files.py $StagedPyFiles
    $ErrorCount = $LASTEXITCODE
} catch {
    Write-Host "Error running linting script: $_" -ForegroundColor Red
    exit 1
}

# If lint_python_files.py returns a non-zero code, abort the commit
if ($ErrorCount -ne 0) {
    Write-Host "❌ Pre-commit check failed! Fix the linting issues before committing." -ForegroundColor Red
    exit 1
}

Write-Host "✅ All pre-commit checks passed!" -ForegroundColor Green
exit 0 