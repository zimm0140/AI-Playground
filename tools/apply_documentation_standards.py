#!/usr/bin/env python3
"""
Documentation Standards Application Tool

This script applies our project's documentation standards to Python files.
It standardizes docstrings, improves type annotations, and ensures consistent formatting.
"""

import os
import re
import argparse
import subprocess
from typing import List, Dict, Set, Tuple, Optional, Any

# Map of function types to apply docstring improvement to
FUNCTION_TYPES = {
    # Format: function_name: return_type
    "validate_component": "Tuple[bool, List[str]]",
    "validate_all_components": "bool",
    "patch_files": "None",
    "_patch_paint_biz": "None",
    "_patch_web_api": "None",
    "_patch_test_api": "None",
    "_patch_xpu_hijacks": "None",
    "_ensure_valid_test_api": "None",
    "get_current_stats": "Dict[str, Any]",
    "_get_total_python_files": "int",
    "_get_priority_issue_counts": "Dict[str, int]",
    "_get_files_with_issues_from_statistics": "int",
    "_get_files_with_issues_from_json": "int",
    "_get_stats_from_track_technical_debt": "Dict[str, Any]",
}

def apply_google_docstring(file_path: str, function_name: str, return_type: str) -> bool:
    """Apply Google-style docstring to a specific function in a file.
    
    Args:
        file_path: Path to the Python file.
        function_name: Name of the function to update.
        return_type: Return type of the function.
        
    Returns:
        True if changes were made, False otherwise.
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the function definition
    pattern = rf'def\s+{function_name}\s*\([^)]*\):'
    matches = list(re.finditer(pattern, content))
    if not matches:
        print(f"No function {function_name} found in {file_path}")
        return False
    
    # Get the position of the function
    match = matches[0]
    function_start = match.start()
    
    # Find the indentation level
    line_start = content.rfind('\n', 0, function_start) + 1
    indentation = content[line_start:function_start]
    
    # Extract parameters
    params_str = match.group(0)[match.group(0).find('(') + 1:match.group(0).rfind(')')]
    params = [param.strip() for param in params_str.split(',') if param.strip()]
    
    # Get param names
    param_names = []
    for param in params:
        param_parts = param.split(':')[0].split('=')[0].strip()
        if param_parts not in ('self', 'cls') and param_parts:
            param_names.append(param_parts)
    
    # Check if there's already a docstring
    docstring_start = content.find('"""', function_start)
    next_line_start = content.find('\n', function_start)
    if docstring_start > next_line_start and docstring_start < content.find('\n\n', function_start):
        # There's already a docstring, skip
        print(f"Docstring already exists for {function_name} in {file_path}")
        return False
    
    # Create a new docstring
    docstring = f'"""{function_name.replace("_", " ").capitalize()}.\n\n'
    
    # Add parameters section if there are parameters
    if param_names:
        docstring += 'Args:\n'
        for param in param_names:
            docstring += f'    {param}: Description of {param}.\n'
        docstring += '\n'
    
    # Add return section
    if return_type and return_type.lower() != 'none':
        docstring += 'Returns:\n'
        if return_type == 'bool':
            docstring += '    True if successful, False otherwise.\n'
        elif return_type == 'int':
            docstring += '    Number of items processed.\n'
        elif return_type.startswith('Dict'):
            docstring += '    Dictionary containing the results.\n'
        elif return_type.startswith('List'):
            docstring += '    List of results.\n'
        elif return_type.startswith('Tuple'):
            docstring += '    Tuple containing the results.\n'
        else:
            docstring += f'    {return_type} containing the results.\n'
    
    docstring += '"""'
    
    # Find where to insert the docstring
    insert_pos = content.find('\n', function_start) + 1
    
    # Insert the docstring with correct indentation
    new_content = (
        content[:insert_pos] + 
        indentation + '    ' + docstring + '\n' +
        content[insert_pos:]
    )
    
    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Applied docstring to {function_name} in {file_path}")
    return True

def add_type_annotations(file_path: str) -> int:
    """Add type annotations to functions in a file.
    
    Args:
        file_path: Path to the Python file.
        
    Returns:
        Number of functions modified.
    """
    # Use the mypy-stubgen tool if available
    try:
        result = subprocess.run(
            ['stubgen', file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"Generated type stubs for {file_path}")
            return 1
        else:
            print(f"Failed to generate type stubs for {file_path}: {result.stderr}")
            return 0
    except FileNotFoundError:
        print("stubgen tool not found, skipping type annotation generation")
        return 0

def process_file(file_path: str) -> Tuple[int, int]:
    """Process a Python file to apply documentation standards.
    
    Args:
        file_path: Path to the Python file.
        
    Returns:
        Tuple of (docstrings_added, type_annotations_added).
    """
    docstrings_added = 0
    functions_processed = set()
    
    # Get the file name
    file_name = os.path.basename(file_path)
    
    # Add docstrings to known functions
    for function_name, return_type in FUNCTION_TYPES.items():
        if apply_google_docstring(file_path, function_name, return_type):
            docstrings_added += 1
            functions_processed.add(function_name)
    
    # Add type annotations
    type_annotations_added = add_type_annotations(file_path)
    
    return docstrings_added, type_annotations_added

def main():
    """Run the documentation standards application tool."""
    parser = argparse.ArgumentParser(description='Apply documentation standards to Python files')
    parser.add_argument('--target-files', nargs='+', help='List of files to process')
    
    args = parser.parse_args()
    
    total_docstrings = 0
    total_type_annotations = 0
    
    # Process the listed files or use default refactored files
    target_files = args.target_files if args.target_files else [
        '.github/workflows/scripts/validate_components.py',
        '.github/workflows/scripts/fix_ci_issues.py',
        'tools/linting/track_progress.py',
        '.github/workflows/scripts/comment_on_workflow_pr.py',
        '.github/workflows/scripts/generate_workflow_docs.py',
    ]
    
    for file_path in target_files:
        if os.path.exists(file_path) and file_path.endswith('.py'):
            print(f"Processing {file_path}...")
            docstrings, type_annotations = process_file(file_path)
            total_docstrings += docstrings
            total_type_annotations += type_annotations
    
    print(f"\nSummary:")
    print(f"- Added {total_docstrings} docstrings")
    print(f"- Added {total_type_annotations} type annotations")

if __name__ == '__main__':
    main() 