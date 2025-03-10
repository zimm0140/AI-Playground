#!/usr/bin/env python
"""
Environment setup and validation script for AI-Playground
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def ensure_directory_exists(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def check_python_version():
    """Check Python version is 3.6+"""
    print(f"Using Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    if sys.version_info < (3, 6):
        print("ERROR: Python 3.6 or higher is required")
        sys.exit(1)
    else:
        print("✓ Python version OK")

def check_environment():
    """Check if running in a virtual environment"""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("WARNING: Not running in a virtual environment!")
        print("It's recommended to use a virtual environment (venv or conda)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set up a virtual environment first.")
            print("See CONTRIBUTING.md for instructions.")
            sys.exit(1)
    else:
        print(f"✓ Using virtual environment: {sys.prefix}")

def install_dependencies():
    """Install dependencies from requirements.txt"""
    if Path("requirements.txt").exists():
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
    else:
        print("ERROR: requirements.txt not found")
        sys.exit(1)

def check_dependencies():
    """Check that key dependencies are installed"""
    try:
        import jsonschema
        print(f"✓ jsonschema version: {jsonschema.__version__}")
    except ImportError:
        print("ERROR: jsonschema not installed. Run 'pip install -r requirements.txt'")
        sys.exit(1)

def main():
    """Main function"""
    print("\n=== AI-Playground Environment Setup ===\n")
    
    # Create scripts directory if run directly
    if __name__ == "__main__":
        ensure_directory_exists("scripts")
    
    # Check Python version
    check_python_version()
    
    # Check for virtual environment
    check_environment()
    
    # Install dependencies if needed
    should_install = input("Install/update dependencies? (y/n): ")
    if should_install.lower() == 'y':
        install_dependencies()
    
    # Verify dependencies
    check_dependencies()
    
    print("\n=== Setup Complete ===")
    print("\nYou're ready to contribute to AI-Playground!")
    print("Remember to activate your virtual environment when working on this project.")

if __name__ == "__main__":
    main() 