#!/usr/bin/env python3
"""
Type Annotation Application Tool

This script applies type annotations to Python files by using generated stub files.
It improves type safety and code documentation.
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional

def extract_annotations_from_stub(stub_file: str) -> Dict[str, Dict[str, str]]:
    """Extract function type annotations from a stub file.
    
    Args:
        stub_file: Path to the stub file (.pyi).
        
    Returns:
        Dictionary mapping function names to their parameter and return types.
    """
    annotations = {}
    
    try:
        with open(stub_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract function signatures
        # Match both regular functions and methods
        pattern = r'(?:^|\s)(?:def|async def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*->\s*([^:]+):'
        matches = re.finditer(pattern, content, re.MULTILINE)
        
        for match in matches:
            func_name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3).strip()
            
            # Parse parameters
            params = {}
            if params_str:
                param_list = params_str.split(',')
                for param in param_list:
                    param = param.strip()
                    if ':' in param:
                        # Parameter has type annotation
                        name, type_ann = param.split(':', 1)
                        name = name.strip()
                        if name != 'self' and name != 'cls':
                            params[name] = type_ann.strip()
            
            annotations[func_name] = {
                'params': params,
                'return': return_type
            }
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading stub file {stub_file}: {str(e)}")
        return {}
    
    return annotations

def apply_annotations_to_file(file_path: str, annotations: Dict[str, Dict[str, str]], dry_run: bool = False) -> int:
    """Apply type annotations to a Python file.
    
    Args:
        file_path: Path to the Python file.
        annotations: Dictionary containing function annotations.
        dry_run: If True, don't modify the file, just report changes. Defaults to False.
        
    Returns:
        Number of annotations applied.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return 0
    
    # Track changes
    changes = 0
    modified_content = content
    
    # Find function definitions and update them
    pattern = r'(?:^|\s)(?:def|async def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)(?:\s*->\s*[^:]+)?:'
    
    # Process in reverse order to avoid position shifts
    matches = list(re.finditer(pattern, content, re.MULTILINE))
    for match in reversed(matches):
        func_name = match.group(1)
        
        if func_name in annotations:
            # Found annotation for this function
            func_ann = annotations[func_name]
            old_params_str = match.group(2)
            old_sig = match.group(0)
            
            # Parse existing parameters
            old_params = [p.strip() for p in old_params_str.split(',') if p.strip()]
            new_params = []
            
            # Update parameter types
            for param in old_params:
                param_name = param.split('=')[0].split(':')[0].strip()
                if param_name in func_ann['params'] and ':' not in param:
                    # Add type annotation
                    if '=' in param:
                        name, default = param.split('=', 1)
                        new_params.append(f"{name.strip()}: {func_ann['params'][param_name]} = {default.strip()}")
                    else:
                        new_params.append(f"{param}: {func_ann['params'][param_name]}")
                else:
                    new_params.append(param)
            
            # Construct new signature
            new_params_str = ', '.join(new_params)
            new_sig = old_sig.replace(f"({old_params_str})", f"({new_params_str})")
            
            # Add return type if not present
            if '->' not in new_sig and func_ann['return'] != 'None':
                new_sig = new_sig.replace(':', f" -> {func_ann['return']}:")
            
            # Apply change
            if new_sig != old_sig:
                modified_content = modified_content[:match.start()] + new_sig + modified_content[match.end():]
                changes += 1
                print(f"Updated signature for {func_name} in {file_path}")
                if dry_run:
                    print(f"  From: {old_sig.strip()}")
                    print(f"  To  : {new_sig.strip()}")
    
    # Write changes if not a dry run
    if changes > 0 and not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        except IOError as e:
            print(f"Error writing to file {file_path}: {str(e)}")
            return 0
    
    return changes

def main():
    """Run the type annotation application tool."""
    parser = argparse.ArgumentParser(description="Apply type annotations to Python files")
    parser.add_argument('--stub-dir', default='out', help='Directory containing stub files')
    parser.add_argument('--target-files', nargs='+', help='List of files to process')
    parser.add_argument('--dry-run', action='store_true', help="Don't modify files, just show changes")
    
    args = parser.parse_args()
    
    # Determine target files
    target_files = args.target_files if args.target_files else [
        '.github/workflows/scripts/comment_on_workflow_pr.py',
        '.github/workflows/scripts/generate_workflow_docs.py',
    ]
    
    total_annotations = 0
    
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        base_name = os.path.basename(file_path)
        module_name = os.path.splitext(base_name)[0]
        stub_file = os.path.join(args.stub_dir, f"{module_name}.pyi")
        
        if not os.path.exists(stub_file):
            print(f"Stub file not found: {stub_file}")
            continue
        
        print(f"Processing {file_path} with stub {stub_file}...")
        
        # Extract annotations from stub
        annotations = extract_annotations_from_stub(stub_file)
        
        if not annotations:
            print(f"No annotations found in {stub_file}")
            continue
        
        # Apply annotations
        changes = apply_annotations_to_file(file_path, annotations, args.dry_run)
        total_annotations += changes
    
    print(f"\nSummary: Applied {total_annotations} type annotations")

if __name__ == '__main__':
    main() 