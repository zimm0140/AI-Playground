#!/usr/bin/env python3
"""
Common Typing Error Fixer

This script fixes common typing errors that occur in Python files:
1. Using built-in types in annotations (dict, list, tuple) instead of typing.Dict, typing.List, typing.Tuple
2. Using | syntax for unions in Python < 3.10
3. Missing Optional for None default values
4. Path.endswith() vs Path.suffix
5. Sequence vs List for append operations
"""

import re
import os
import sys
import argparse
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path


def fix_builtin_types(content: str) -> str:
    """Replace built-in type subscripts with typing equivalents.
    
    Args:
        content: File content
        
    Returns:
        Updated content
    """
    # Patterns for subscripted built-in types in annotations
    patterns = [
        (r'(\W)dict\[(.*?)\]', r'\1Dict[\2]'),
        (r'(\W)list\[(.*?)\]', r'\1List[\2]'),
        (r'(\W)tuple\[(.*?)\]', r'\1Tuple[\2]'),
        (r'(\W)set\[(.*?)\]', r'\1Set[\2]'),
    ]
    
    # Import statements to add
    imports_to_add = set()
    
    modified_content = content
    for pattern, replacement in patterns:
        if re.search(pattern, modified_content):
            if 'Dict' in replacement:
                imports_to_add.add('Dict')
            elif 'List' in replacement:
                imports_to_add.add('List')
            elif 'Tuple' in replacement:
                imports_to_add.add('Tuple')
            elif 'Set' in replacement:
                imports_to_add.add('Set')
                
            modified_content = re.sub(pattern, replacement, modified_content)
    
    # Add required imports if not already present
    if imports_to_add and 'from typing import' not in modified_content:
        # Find existing typing import
        typing_import_match = re.search(r'from\s+typing\s+import\s+(.*?)(?:\n|$)', modified_content)
        if typing_import_match:
            # Extract existing imports
            existing_imports = typing_import_match.group(1).split(',')
            existing_imports = [imp.strip() for imp in existing_imports]
            
            # Add new imports to existing ones
            for imp in imports_to_add:
                if imp not in existing_imports:
                    existing_imports.append(imp)
            
            # Replace the old import with the updated one
            new_import = 'from typing import ' + ', '.join(sorted(existing_imports))
            modified_content = modified_content.replace(typing_import_match.group(0), new_import + '\n')
        else:
            # No existing typing import, add a new one at the top
            import_statement = 'from typing import ' + ', '.join(sorted(imports_to_add)) + '\n'
            # Try to add after any module docstring
            docstring_end = re.search(r'""".*?"""\s*\n', modified_content, re.DOTALL)
            if docstring_end:
                pos = docstring_end.end()
                modified_content = modified_content[:pos] + import_statement + modified_content[pos:]
            else:
                # Add at the top
                modified_content = import_statement + modified_content
    
    return modified_content


def fix_union_syntax(content: str) -> str:
    """Replace | syntax for unions with Union[] for Python < 3.10.
    
    Args:
        content: File content
        
    Returns:
        Updated content
    """
    # Find annotation with | syntax, excluding strings
    pattern = r'(\w+)\s*:\s*([^\'\"]*?\w+\s*\|\s*\w+[^\'\"]*?)(\s*=|\s*\)|\s*,|\s*$)'
    
    # Check if there are any union syntax usages
    if not re.search(pattern, content):
        return content
    
    # Add Union to imports
    modified_content = content
    if 'Union' not in modified_content:
        if 'from typing import' in modified_content:
            typing_import_match = re.search(r'from\s+typing\s+import\s+(.*?)(?:\n|$)', modified_content)
            if typing_import_match:
                # Extract existing imports
                existing_imports = typing_import_match.group(1).split(',')
                existing_imports = [imp.strip() for imp in existing_imports]
                
                # Add Union if not already present
                if 'Union' not in existing_imports:
                    existing_imports.append('Union')
                
                # Replace the old import with the updated one
                new_import = 'from typing import ' + ', '.join(sorted(existing_imports))
                modified_content = modified_content.replace(typing_import_match.group(0), new_import + '\n')
        else:
            # No existing typing import, add a new one at the top
            import_statement = 'from typing import Union\n'
            # Try to add after any module docstring
            docstring_end = re.search(r'""".*?"""\s*\n', modified_content, re.DOTALL)
            if docstring_end:
                pos = docstring_end.end()
                modified_content = modified_content[:pos] + import_statement + modified_content[pos:]
            else:
                # Add at the top
                modified_content = import_statement + modified_content
    
    # Replace | syntax with Union[]
    def replace_union(match):
        param_name = match.group(1)
        type_expr = match.group(2)
        rest = match.group(3)
        
        # Convert a | b | c to Union[a, b, c]
        union_types = [t.strip() for t in type_expr.split('|')]
        union_expr = f"Union[{', '.join(union_types)}]"
        
        return f"{param_name}: {union_expr}{rest}"
    
    modified_content = re.sub(pattern, replace_union, modified_content)
    
    return modified_content


def fix_implicit_optional(content: str) -> str:
    """Add Optional[] for parameters with None default values.
    
    Args:
        content: File content
        
    Returns:
        Updated content
    """
    # Find parameter definitions with None defaults but without Optional[]
    pattern = r'(\w+)\s*:\s*([^=]+?)\s*=\s*None'
    
    # Check if there are any implicit optionals
    if not re.search(pattern, content):
        return content
    
    # Add Optional to imports
    modified_content = content
    if 'Optional' not in modified_content:
        if 'from typing import' in modified_content:
            typing_import_match = re.search(r'from\s+typing\s+import\s+(.*?)(?:\n|$)', modified_content)
            if typing_import_match:
                # Extract existing imports
                existing_imports = typing_import_match.group(1).split(',')
                existing_imports = [imp.strip() for imp in existing_imports]
                
                # Add Optional if not already present
                if 'Optional' not in existing_imports:
                    existing_imports.append('Optional')
                
                # Replace the old import with the updated one
                new_import = 'from typing import ' + ', '.join(sorted(existing_imports))
                modified_content = modified_content.replace(typing_import_match.group(0), new_import + '\n')
        else:
            # No existing typing import, add a new one at the top
            import_statement = 'from typing import Optional\n'
            # Try to add after any module docstring
            docstring_end = re.search(r'""".*?"""\s*\n', modified_content, re.DOTALL)
            if docstring_end:
                pos = docstring_end.end()
                modified_content = modified_content[:pos] + import_statement + modified_content[pos:]
            else:
                # Add at the top
                modified_content = import_statement + modified_content
    
    # Replace type with Optional[type] for parameters with None defaults
    def replace_optional(match):
        param_name = match.group(1)
        type_expr = match.group(2).strip()
        
        # Skip if it's already Optional[]
        if type_expr.startswith('Optional['):
            return match.group(0)
        
        return f"{param_name}: Optional[{type_expr}] = None"
    
    modified_content = re.sub(pattern, replace_optional, modified_content)
    
    return modified_content


def fix_path_operations(content: str) -> str:
    """Fix common Path operations issues.
    
    Args:
        content: File content
        
    Returns:
        Updated content
    """
    # Fix Path.endswith
    modified_content = content
    
    if 'endswith' in modified_content and 'Path' in modified_content:
        # Replace .endswith() with .suffix checks
        endswith_pattern = r'(\w+)\.endswith\([\'\"](.*?)[\'\"]'
        
        def replace_endswith(match):
            path_var = match.group(1)
            extension = match.group(2)
            
            # Make sure extension starts with a dot
            if not extension.startswith('.'):
                extension = '.' + extension
                
            return f"{path_var}.suffix == '{extension}'"
        
        modified_content = re.sub(endswith_pattern, replace_endswith, modified_content)
    
    return modified_content


def fix_sequence_operations(content: str) -> str:
    """Fix operations on Sequence that should use MutableSequence or List.
    
    Args:
        content: File content
        
    Returns:
        Updated content
    """
    # Replace Sequence with List for variables that use .append()
    modified_content = content
    
    # Look for variables typed as Sequence that use append
    sequence_pattern = r'(\w+)\s*:\s*(?:Sequence|Collection)\[(.*?)\]'
    sequence_vars = re.findall(sequence_pattern, modified_content)
    
    for var_name, inner_type in sequence_vars:
        # Check if this variable uses append method
        append_pattern = rf'\b{var_name}\.append\('
        if re.search(append_pattern, modified_content):
            # Replace Sequence with List
            modified_content = re.sub(
                rf'{var_name}\s*:\s*Sequence\[{re.escape(inner_type)}\]', 
                f'{var_name}: List[{inner_type}]', 
                modified_content
            )
            modified_content = re.sub(
                rf'{var_name}\s*:\s*Collection\[{re.escape(inner_type)}\]', 
                f'{var_name}: List[{inner_type}]', 
                modified_content
            )
            
            # Make sure List is imported
            if 'List' not in modified_content:
                if 'from typing import' in modified_content:
                    typing_import_match = re.search(r'from\s+typing\s+import\s+(.*?)(?:\n|$)', modified_content)
                    if typing_import_match:
                        # Extract existing imports
                        existing_imports = typing_import_match.group(1).split(',')
                        existing_imports = [imp.strip() for imp in existing_imports]
                        
                        # Add List if not already present
                        if 'List' not in existing_imports:
                            existing_imports.append('List')
                        
                        # Replace the old import with the updated one
                        new_import = 'from typing import ' + ', '.join(sorted(existing_imports))
                        modified_content = modified_content.replace(typing_import_match.group(0), new_import + '\n')
                else:
                    # No existing typing import, add a new one at the top
                    import_statement = 'from typing import List\n'
                    # Try to add after any module docstring
                    docstring_end = re.search(r'""".*?"""\s*\n', modified_content, re.DOTALL)
                    if docstring_end:
                        pos = docstring_end.end()
                        modified_content = modified_content[:pos] + import_statement + modified_content[pos:]
                    else:
                        # Add at the top
                        modified_content = import_statement + modified_content
    
    return modified_content


def fix_typing_issues(file_path: str, verbose: bool = False) -> int:
    """Fix common typing issues in a Python file.
    
    Args:
        file_path: Path to the Python file
        verbose: Whether to print verbose output
        
    Returns:
        Number of changes made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply fixes
        modified_content = original_content
        modified_content = fix_builtin_types(modified_content)
        modified_content = fix_union_syntax(modified_content)
        modified_content = fix_implicit_optional(modified_content)
        modified_content = fix_path_operations(modified_content)
        modified_content = fix_sequence_operations(modified_content)
        
        # Check if changes were made
        if modified_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            changes = sum(1 for a, b in zip(original_content.splitlines(), modified_content.splitlines()) if a != b)
            if verbose:
                print(f"Fixed typing issues in {file_path} ({changes} changes)")
            return changes
        else:
            if verbose:
                print(f"No typing issues to fix in {file_path}")
            return 0
            
    except Exception as e:
        print(f"Error fixing typing issues in {file_path}: {str(e)}")
        return 0


def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Find all Python files in a directory recursively.
    
    Args:
        directory: Directory to search in
        exclude_patterns: List of regex patterns to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = []
        
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                excluded = any(re.search(pattern, file_path) for pattern in exclude_patterns)
                if not excluded:
                    python_files.append(file_path)
                    
    return python_files


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix common typing issues in Python code")
    parser.add_argument(
        "--path", "-p", required=True, help="Path to a Python file or directory to process"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--exclude", "-e", action="append", default=[], 
        help="Regex patterns to exclude (can be specified multiple times)"
    )
    
    args = parser.parse_args()
    path = args.path
    verbose = args.verbose
    exclude_patterns = args.exclude
    
    if os.path.isfile(path) and path.endswith(".py"):
        files = [path]
    elif os.path.isdir(path):
        files = find_python_files(path, exclude_patterns)
    else:
        print(f"Error: {path} is not a valid Python file or directory")
        return
    
    total_changes = 0
    total_files_fixed = 0
    
    for file_path in files:
        changes = fix_typing_issues(file_path, verbose)
        if changes > 0:
            total_changes += changes
            total_files_fixed += 1
            
    print(f"Fixed {total_changes} typing issues in {total_files_fixed} files")


if __name__ == "__main__":
    main() 