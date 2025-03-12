#!/usr/bin/env python3
"""
Script to automatically fix linting issues reported by Ruff.
This script will run ruff with the --fix option on the directories
with reported issues.
"""

import os
import subprocess
import sys

def main():
    """Run ruff with --fix on specified directories."""
    directories = [
        ".github/workflows/scripts/",
        "tests/",
        "LlamaCPP/",
        "OpenVINO/"
    ]
    
    print("🔍 Running Ruff auto-fixes on codebase...")
    success = True
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ Directory {directory} does not exist, skipping.")
            continue
            
        print(f"🛠️ Fixing issues in {directory}...")
        try:
            result = subprocess.run(
                ["ruff", "check", "--fix", directory],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to fix all issues in {directory}")
                print(f"Error output: {result.stderr}")
                success = False
            else:
                print(f"✅ Successfully fixed issues in {directory}")
                
            # Also run formatter
            format_result = subprocess.run(
                ["ruff", "format", directory],
                capture_output=True,
                text=True
            )
            
            if format_result.returncode != 0:
                print(f"❌ Failed to format {directory}")
                print(f"Error output: {format_result.stderr}")
                success = False
            else:
                print(f"✅ Successfully formatted {directory}")
                
        except Exception as e:
            print(f"❌ Error processing {directory}: {str(e)}")
            success = False
    
    if success:
        print("✅ All fixable linting issues have been addressed.")
        print("Note: Some issues may require manual attention.")
        return 0
    else:
        print("⚠️ Some linting issues could not be automatically fixed.")
        print("Please review the output and fix remaining issues manually.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 