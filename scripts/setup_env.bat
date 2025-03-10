@echo off
echo Running AI-Playground Environment Setup...

:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.6 or higher.
    pause
    exit /b 1
)

:: Run the setup script
python scripts/setup_env.py

pause 