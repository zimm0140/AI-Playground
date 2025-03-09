# PowerShell script to set up Git hooks for the project
# Run this script from the project root to set up pre-commit hooks on Windows

$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow

Write-Host "Setting up Git hooks for AI-Playground project..." -ForegroundColor $Yellow

# Ensure the hooks directory exists
$HooksDir = ".git/hooks"
if (-not (Test-Path $HooksDir)) {
    New-Item -ItemType Directory -Path $HooksDir -Force | Out-Null
}

# Copy the pre-commit hooks
Copy-Item ".github/hooks/pre-commit" -Destination "$HooksDir/pre-commit" -Force
Copy-Item ".github/hooks/pre-commit.ps1" -Destination "$HooksDir/pre-commit.ps1" -Force

# Make the hooks executable (not needed on Windows but good for cross-platform compatibility)
Write-Host "Making hooks executable..."
# Try Git Bash's chmod if available
try {
    & bash -c "chmod +x .git/hooks/pre-commit" 2>$null
} catch {
    Write-Host "Could not make hooks executable using bash. This is OK on Windows." -ForegroundColor $Yellow
}

Write-Host "Git hooks successfully installed!" -ForegroundColor $Green
Write-Host "Pre-commit hooks will now run automatically when you commit."
Write-Host "To bypass hooks temporarily, use" -NoNewline
Write-Host " git commit --no-verify" -ForegroundColor $Yellow 