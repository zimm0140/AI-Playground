#!/usr/bin/env python3
"""
Fix remaining linting issues in service directory.

This script focuses specifically on the service directory to fix any
remaining linting issues detected by the CI pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def fix_linting_issues():
    """Run Ruff to fix linting issues in the service directory."""
    service_dir = Path("service")
    
    if not service_dir.exists():
        print(f"Error: {service_dir} directory does not exist.")
        return 1
    
    # Run Ruff with the same parameters as the CI
    print("Running Ruff with the same parameters as the CI...")
    cmd = [
        sys.executable, "-m", "ruff", "check", "service",
        "--ignore=F401,W291,F821,N801,N802,N803", 
        "--fix",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}", file=sys.stderr)
        
        # Check for remaining issues
        print("\nChecking for remaining issues...")
        check_cmd = [
            sys.executable, "-m", "ruff", "check", "service",
            "--ignore=F401,W291,F821,N801,N802,N803",
        ]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
        
        if check_result.returncode == 0:
            print("✅ All linting issues fixed!")
            return 0
        else:
            print("❌ Some issues remain:")
            print(check_result.stdout)
            return 1
    
    except subprocess.SubprocessError as e:
        print(f"Error running Ruff: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix linting issues in service directory")
    args = parser.parse_args()
    
    return fix_linting_issues()


if __name__ == "__main__":
    sys.exit(main()) 