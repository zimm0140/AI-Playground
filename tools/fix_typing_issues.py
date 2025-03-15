#!/usr/bin/env python3
"""
Automated Typing Issues Fixer

This script identifies and fixes common typing issues in Python files.
It adds type annotations based on mypy error messages, function usage,
and common patterns.
"""

import os
import re
import sys
import argparse
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path


def run_mypy_on_file(file_path: str) -> Tuple[bool, str]:
    """Run mypy on a single file and return the result.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple containing success status and output
    """
    result = subprocess.run(
        ["python", "-m", "mypy", "--config-file", "mypy.ini", file_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout


def parse_mypy_errors(output: str) -> List[Dict[str, Any]]:
    """Parse mypy error messages into structured data.
    
    Args:
        output: mypy output text
        
    Returns:
        List of error dictionaries with line, message and error type
    """
    errors = []
    
    # Regular expression to match mypy error lines
    # Format: file:line: error: message [error-code]
    pattern = r'(?P<file>.*?):(?P<line>\d+): (?P<level>\w+): (?P<message>.*?)(?:\s+\[(?P<code>[a-z0-9-]+)\])?$'
    
    for line in output.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            errors.append({
                'file': match.group('file'),
                'line': int(match.group('line')),
                'level': match.group('level'),
                'message': match.group('message'),
                'code': match.group('code')
            })
    
    return errors


def get_function_signature(file_lines: List[str], line_num: int) -> Tuple[int, str]:
    """Extract function signature from the file.
    
    Args:
        file_lines: List of file lines
        line_num: Line number where the function is defined
        
    Returns:
        Tuple of (start_line, signature)
    """
    # Look for function definition
    line = file_lines[line_num - 1]
    
    # Check if this is already the function definition line
    if re.match(r'\s*def\s+\w+\s*\(', line):
        # Find the end of the signature (which might span multiple lines)
        signature = line
        i = line_num
        open_parens = line.count('(') - line.count(')')
        
        while open_parens > 0 and i < len(file_lines):
            i += 1
            if i < len(file_lines):
                signature += '\n' + file_lines[i - 1]
                open_parens += file_lines[i - 1].count('(') - file_lines[i - 1].count(')')
        
        return line_num - 1, signature
    
    # If not, search backwards for the function definition
    for i in range(line_num - 2, -1, -1):
        if re.match(r'\s*def\s+\w+\s*\(', file_lines[i]):
            # Found the function definition
            signature = file_lines[i]
            start_line = i
            
            # Check if signature spans multiple lines
            open_parens = signature.count('(') - signature.count(')')
            j = i
            
            while open_parens > 0 and j < len(file_lines) - 1:
                j += 1
                signature += '\n' + file_lines[j]
                open_parens += file_lines[j].count('(') - file_lines[j].count(')')
                
            return start_line, signature
    
    # Function definition not found
    return -1, ""


def get_missing_return_type(error: Dict[str, Any]) -> Optional[str]:
    """Extract missing return type from error message.
    
    Args:
        error: Error dictionary
        
    Returns:
        Return type as string or None if can't be determined
    """
    message = error['message']
    
    # "Function is missing a return type annotation" or
    # "Function is missing a type annotation"
    if "missing" in message and "return type" in message:
        # Try to infer from other parts of the message
        return_match = re.search(r'the return type is "([^"]+)"', message)
        if return_match:
            return return_match.group(1)
        
        # Default to Any if we can't determine
        return "Any"
    
    # "Returning Any from function declared to return "X""
    ret_match = re.search(r'Returning Any from function declared to return "([^"]+)"', message)
    if ret_match:
        return ret_match.group(1)
    
    return None


def fix_missing_return_type(file_path: str, error: Dict[str, Any], file_lines: List[str]) -> bool:
    """Fix missing return type annotation in function.
    
    Args:
        file_path: Path to the file
        error: Error dictionary
        file_lines: List of file lines
        
    Returns:
        True if file was modified, False otherwise
    """
    line_num = error['line']
    start_line, signature = get_function_signature(file_lines, line_num)
    
    if start_line == -1:
        print(f"  Could not find function definition for error on line {line_num}")
        return False
    
    return_type = get_missing_return_type(error)
    if not return_type:
        print(f"  Could not determine return type for error on line {line_num}")
        return False
    
    # Parse the signature to add the return type
    sig_lines = signature.split('\n')
    last_line = sig_lines[-1]
    
    # Check if there's already a return type annotation
    if "->" in last_line:
        print(f"  Function already has a return type annotation on line {start_line + len(sig_lines) - 1}")
        return False
    
    # Add return type annotation
    if ":" in last_line:
        # Function has type annotations already
        colon_pos = last_line.rfind(":")
        sig_lines[-1] = last_line[:colon_pos] + f" -> {return_type}:"
    else:
        # Function needs return type added
        if ')' in last_line:
            closing_paren_pos = last_line.rfind(")")
            sig_lines[-1] = last_line[:closing_paren_pos + 1] + f" -> {return_type}:" + last_line[closing_paren_pos + 1:]
    
    # Update the file lines
    for i, sig_line in enumerate(sig_lines):
        file_lines[start_line + i] = sig_line
    
    print(f"  Added return type '{return_type}' to function on line {start_line + 1}")
    return True


def get_variable_type(error: Dict[str, Any]) -> Optional[str]:
    """Extract variable type from error message.
    
    Args:
        error: Error dictionary
        
    Returns:
        Variable type as string or None if can't be determined
    """
    message = error['message']
    
    # "Need type annotation for 'x' (hint: "x: List[<type>] = ...")"
    type_hint_match = re.search(r'hint: "(?:\w+): ([^"=]+)', message)
    if type_hint_match:
        return type_hint_match.group(1)
    
    # "Incompatible types in assignment (expression has type "X", variable has type "Y")"
    incomp_match = re.search(r'expression has type "([^"]+)", variable has type', message)
    if incomp_match:
        return incomp_match.group(1)
    
    # Default for common patterns
    if "dict" in message.lower():
        return "Dict[str, Any]"
    elif "list" in message.lower():
        return "List[Any]"
    elif "string" in message.lower() or "str" in message.lower():
        return "str"
    elif "int" in message.lower():
        return "int"
    elif "float" in message.lower():
        return "float"
    elif "bool" in message.lower():
        return "bool"
    elif "optional" in message.lower():
        return "Optional[Any]"
    
    return None


def fix_missing_variable_type(file_path: str, error: Dict[str, Any], file_lines: List[str]) -> bool:
    """Fix missing variable type annotation.
    
    Args:
        file_path: Path to the file
        error: Error dictionary
        file_lines: List of file lines
        
    Returns:
        True if file was modified, False otherwise
    """
    line_num = error['line']
    line = file_lines[line_num - 1]
    
    # Extract variable name from error message
    var_match = re.search(r'Need type annotation for \'([^\']+)\'', error['message'])
    if not var_match:
        print(f"  Could not extract variable name from error message on line {line_num}")
        return False
    
    var_name = var_match.group(1)
    var_type = get_variable_type(error)
    
    if not var_type:
        # Default to Any if we can't determine the type
        var_type = "Any"
    
    # Check if variable is being assigned
    assign_pattern = rf'(\s*)({re.escape(var_name)})\s*='
    assign_match = re.search(assign_pattern, line)
    
    if assign_match:
        # Add type annotation to variable
        indent = assign_match.group(1)
        new_line = f"{indent}{var_name}: {var_type} = {line.split('=', 1)[1]}"
        file_lines[line_num - 1] = new_line
        print(f"  Added type annotation '{var_type}' to variable '{var_name}' on line {line_num}")
        return True
    
    print(f"  Could not find variable assignment for '{var_name}' on line {line_num}")
    return False


def fix_item_none_attribute(file_path: str, error: Dict[str, Any], file_lines: List[str]) -> bool:
    """Fix 'Item "None" of "Optional[X]" has no attribute "Y"' errors.
    
    Args:
        file_path: Path to the file
        error: Error dictionary
        file_lines: List of file lines
        
    Returns:
        True if file was modified, False otherwise
    """
    line_num = error['line']
    line = file_lines[line_num - 1]
    
    # Extract information from error message
    none_attr_match = re.search(r'Item "None" of "Optional\[([^\]]+)\]" has no attribute "([^"]+)"', error['message'])
    if not none_attr_match:
        print(f"  Could not parse None attribute error on line {line_num}")
        return False
    
    type_name = none_attr_match.group(1)
    attr_name = none_attr_match.group(2)
    
    # Look for patterns like "x.y" where x might be None
    # We'll add a check: "if x is not None and x.y"
    for var_name in re.findall(r'(\w+)\.'+re.escape(attr_name), line):
        # Find the position of the variable in the line
        var_pos = line.find(f"{var_name}.{attr_name}")
        if var_pos == -1:
            continue
        
        # Check if we're already inside an "if var_name is not None" block
        is_already_checked = False
        
        # Look back a few lines for an if statement
        for i in range(max(0, line_num - 5), line_num):
            prev_line = file_lines[i - 1]
            if f"if {var_name} is not None" in prev_line or f"if {var_name} is None" in prev_line:
                is_already_checked = True
                break
        
        if is_already_checked:
            print(f"  Line {line_num} already has a None check for '{var_name}'")
            continue
        
        # Add a None check
        indent = re.match(r'(\s*)', line).group(1)
        if " and " in line or " or " in line:
            # If the line already has logical operators, just add another condition
            pat = rf'(\w+\s*[=!<>]+\s*){var_name}\.{attr_name}'
            replacement = rf'\1{var_name} is not None and {var_name}.{attr_name}'
            new_line = re.sub(pat, replacement, line)
            
            # If we couldn't do a simple replacement, try a more complex one
            if new_line == line:
                new_line = line.replace(f"{var_name}.{attr_name}", f"{var_name} is not None and {var_name}.{attr_name}")
        else:
            # If it's a simple line, add an if statement around it
            new_lines = [
                f"{indent}if {var_name} is not None:",
                f"{indent}    {line.strip()}"
            ]
            
            # Replace the current line with the if block
            file_lines[line_num - 1] = new_lines[0]
            file_lines.insert(line_num, new_lines[1])
            print(f"  Added None check for '{var_name}' on line {line_num}")
            return True
        
        # Update the line if changed
        if new_line != line:
            file_lines[line_num - 1] = new_line
            print(f"  Added None check for '{var_name}' on line {line_num}")
            return True
    
    print(f"  Could not safely fix None attribute error on line {line_num}")
    return False


def fix_common_typing_issues(file_path: str) -> int:
    """Fix common typing issues in a single file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Number of issues fixed
    """
    # Run mypy to identify issues
    print(f"Running mypy on {file_path}...")
    success, output = run_mypy_on_file(file_path)
    
    if success:
        print(f"No typing issues found in {file_path}")
        return 0
    
    # Parse mypy errors
    errors = parse_mypy_errors(output)
    print(f"Found {len(errors)} typing issues")
    
    if not errors:
        print("No fixable typing issues found")
        return 0
    
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    
    # Track if we've made changes
    changes_made = 0
    
    # Process each error
    for error in errors:
        try:
            fixed = False
            message = error['message']
            
            if "missing a return type annotation" in message or "missing a type annotation" in message:
                fixed = fix_missing_return_type(file_path, error, file_lines)
            elif "Need type annotation for" in message:
                fixed = fix_missing_variable_type(file_path, error, file_lines)
            elif 'Item "None" of "Optional' in message and 'has no attribute' in message:
                fixed = fix_item_none_attribute(file_path, error, file_lines)
            
            if fixed:
                changes_made += 1
        except Exception as e:
            print(f"Error fixing issue on line {error['line']}: {str(e)}")
    
    # Write changes back to the file if any were made
    if changes_made > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(file_lines)
    
    print(f"Fixed {changes_made} typing issues in {file_path}")
    return changes_made


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix common typing issues in Python files")
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Path to the Python file or directory to fix",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process directory recursively",
    )
    
    args = parser.parse_args()
    
    path = args.path
    recursive = args.recursive
    
    total_files = 0
    total_fixes = 0
    
    if os.path.isfile(path) and path.endswith(".py"):
        fixes = fix_common_typing_issues(path)
        total_files = 1
        total_fixes = fixes
    elif os.path.isdir(path):
        for root, _, files in os.walk(path) if recursive else [(path, None, os.listdir(path))]:
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    fixes = fix_common_typing_issues(file_path)
                    total_files += 1
                    total_fixes += fixes
                    
                # If not recursive, break after processing the top-level directory
                if not recursive:
                    break
    else:
        print(f"Error: {path} is not a valid Python file or directory")
        sys.exit(1)
    
    print(f"Fixed {total_fixes} issues in {total_files} file(s)")


if __name__ == "__main__":
    main() 