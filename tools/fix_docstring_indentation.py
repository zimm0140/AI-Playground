#!/usr/bin/env python3
"""
Docstring Indentation Fixer

This script fixes indentation issues with docstrings in Python files.
It ensures that docstrings are properly indented after function/method definitions.
"""

import os
import re
import sys
import argparse
from typing import List, Tuple


def fix_docstring_indentation(content: str) -> str:
    """Fix indentation issues with docstrings in Python code.
    
    Args:
        content: The content of the Python file.
        
    Returns:
        The fixed content with properly indented docstrings.
    """
    lines = content.splitlines()
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for function or method definition
        if re.match(r'^(\s*)def\s+\w+\s*\(.*\)\s*:', line):
            indent_level = len(line) - len(line.lstrip())
            expected_indent = ' ' * (indent_level + 4)  # 4 spaces indentation
            
            # Look ahead for docstring
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_line_stripped = next_line.strip()
                
                # Check if next line is a docstring
                if next_line_stripped.startswith('"""') or next_line_stripped.startswith("'''"):
                    # Check if it's a single-line docstring
                    if (next_line_stripped.endswith('"""') and next_line_stripped.count('"""') == 2) or \
                       (next_line_stripped.endswith("'''") and next_line_stripped.count("'''") == 2):
                        # Single-line docstring
                        if not next_line.startswith(expected_indent):
                            lines[i + 1] = expected_indent + next_line.lstrip()
                    else:
                        # Multi-line docstring
                        docstring_start = i + 1
                        docstring_end = find_docstring_end(lines, docstring_start)
                        
                        # Fix indentation for all lines in the docstring
                        for j in range(docstring_start, docstring_end + 1):
                            if j < len(lines):
                                # Skip empty lines
                                if lines[j].strip():
                                    if not lines[j].startswith(expected_indent):
                                        lines[j] = expected_indent + lines[j].lstrip()
                        
                        # Skip the docstring lines since we've already processed them
                        i = docstring_end
        
        i += 1
    
    return '\n'.join(lines)


def find_docstring_end(lines: List[str], start_idx: int) -> int:
    """Find the end index of a docstring.
    
    Args:
        lines: List of code lines.
        start_idx: Starting index of the docstring.
        
    Returns:
        The index of the line where the docstring ends.
    """
    if start_idx >= len(lines):
        return start_idx
    
    # Determine the docstring delimiter (''' or """)
    first_line = lines[start_idx].strip()
    if first_line.startswith("'''"):
        delimiter = "'''"
    else:
        delimiter = '"""'
    
    # If the first line also ends with the delimiter and has more than just the delimiter
    if first_line.endswith(delimiter) and first_line.count(delimiter) > 1:
        return start_idx
    
    # Find the closing delimiter
    for i in range(start_idx + 1, len(lines)):
        if delimiter in lines[i]:
            return i
    
    # If no closing delimiter found, assume it ends at the last line
    return len(lines) - 1


def process_file(file_path: str, dry_run: bool = False) -> Tuple[bool, int]:
    """Process a single Python file to fix docstring indentation.
    
    Args:
        file_path: Path to the Python file.
        dry_run: If True, don't modify the file, just report changes.
        
    Returns:
        A tuple of (success, number of changes).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_docstring_indentation(content)
        
        if fixed_content != content:
            print(f"Fixed docstring indentation in {file_path}")
            
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
            
            return True, 1
        else:
            return True, 0
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False, 0


def main():
    """Run the docstring indentation fixer."""
    parser = argparse.ArgumentParser(description="Fix docstring indentation in Python files")
    parser.add_argument('--path', required=True, help='File or directory to process')
    parser.add_argument('--dry-run', action='store_true', help="Don't modify files, just show what would be changed")
    
    args = parser.parse_args()
    
    path = args.path
    dry_run = args.dry_run
    
    if os.path.isfile(path) and path.endswith('.py'):
        success, changes = process_file(path, dry_run)
        print(f"Fixed {changes} files")
    
    elif os.path.isdir(path):
        total_changes = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    _, changes = process_file(file_path, dry_run)
                    total_changes += changes
        
        print(f"Fixed {total_changes} files")
    
    else:
        print(f"Invalid path: {path}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 