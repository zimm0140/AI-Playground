#!/usr/bin/env python3
"""
Auto Fix High-Priority Issues

This script automatically applies Ruff's --fix mode to address high-priority linting issues
such as:
- F401: Unused imports
- F821: Undefined names
- F841: Unused variables
- W291: Trailing whitespace

Usage:
    python tools/auto_fix_high_priority.py [--check-only]
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


# High priority rules to fix
HIGH_PRIORITY_RULES = [
    "F401",  # Unused imports
    "F821",  # Undefined names
    "F841",  # Unused variables
    "W291",  # Trailing whitespace
]


def find_python_files(root_dir: str = ".") -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def get_rule_violations(file_path: str, rules: List[str]) -> List[Dict]:
    """Get all rule violations for the specified rules in a file."""
    try:
        cmd = ["ruff", "check", "--select", ",".join(rules), "--output-format=json", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 and result.stdout:
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return []


def fix_violations(file_path: str, rules: List[str], check_only: bool = False) -> Tuple[bool, int]:
    """Apply Ruff's auto-fix to the file for specified rules."""
    try:
        violations = get_rule_violations(file_path, rules)
        if not violations:
            return False, 0
        
        print(f"Found {len(violations)} issues in {file_path}")
        
        if check_only:
            for v in violations:
                print(f"  {v['code']}: {v['message']} (Line {v['location']['row']})")
            return True, len(violations)
        
        cmd = ["ruff", "check", "--fix", "--select", ",".join(rules), file_path]
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Check if fixes were successful
        remaining = get_rule_violations(file_path, rules)
        fixed_count = len(violations) - len(remaining)
        
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} issues in {file_path}")
        if remaining:
            print(f"⚠️ {len(remaining)} issues remain in {file_path}")
            for v in remaining:
                print(f"  {v['code']}: {v['message']} (Line {v['location']['row']})")
        
        return True, fixed_count
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(description="Auto-fix high priority linting issues")
    parser.add_argument("--check-only", action="store_true", help="Only check for issues without fixing")
    args = parser.parse_args()
    
    python_files = find_python_files()
    print(f"Scanning {len(python_files)} Python files for high-priority issues...")
    
    total_files_with_issues = 0
    total_fixed_issues = 0
    
    for file_path in python_files:
        had_issues, fixed_count = fix_violations(file_path, HIGH_PRIORITY_RULES, args.check_only)
        if had_issues:
            total_files_with_issues += 1
            total_fixed_issues += fixed_count
    
    mode = "Checked" if args.check_only else "Fixed"
    print(f"\n{mode} {total_fixed_issues} high-priority issues in {total_files_with_issues} files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())