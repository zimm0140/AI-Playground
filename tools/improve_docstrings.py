#!/usr/bin/env python3
"""
Docstring Improvement Tool

This script analyzes Python files and improves docstrings by adding proper
formatting and type annotations based on our project standards.
"""

import argparse
import ast
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union

# Regular expressions for parsing existing docstrings
PARAM_PATTERN = re.compile(r"(?:^|\n)\s*:param\s+(\w+):\s*(.*?)(?=(?:\n\s*:param|\n\s*:return|\n\s*:raises|\Z))", re.DOTALL)
RETURN_PATTERN = re.compile(r"(?:^|\n)\s*:return(?:s)?:\s*(.*?)(?=(?:\n\s*:param|\n\s*:raises|\Z))", re.DOTALL)
RAISES_PATTERN = re.compile(r"(?:^|\n)\s*:raises\s+(\w+):\s*(.*?)(?=(?:\n\s*:param|\n\s*:return|\n\s*:raises|\Z))", re.DOTALL)

def parse_docstring(docstring: str) -> Dict:
    """Parse an existing docstring to extract information.
    
    Args:
        docstring: The docstring to parse.
        
    Returns:
        A dictionary containing extracted information like description,
        parameters, return value, and exceptions.
    """
    if not docstring:
        return {"description": "", "params": {}, "returns": "", "raises": {}}
    
    # Clean up the docstring
    docstring = docstring.strip()
    
    # Extract description (everything before first :param, :return, or :raises)
    description_end = min([
        docstring.find(":param") if docstring.find(":param") != -1 else len(docstring),
        docstring.find(":return") if docstring.find(":return") != -1 else len(docstring),
        docstring.find(":raises") if docstring.find(":raises") != -1 else len(docstring)
    ])
    description = docstring[:description_end].strip()
    
    # Extract parameters
    params = {}
    for match in PARAM_PATTERN.finditer(docstring):
        name, desc = match.groups()
        params[name] = desc.strip()
    
    # Extract return value
    returns = ""
    return_match = RETURN_PATTERN.search(docstring)
    if return_match:
        returns = return_match.group(1).strip()
    
    # Extract exceptions
    raises = {}
    for match in RAISES_PATTERN.finditer(docstring):
        name, desc = match.groups()
        raises[name] = desc.strip()
    
    return {
        "description": description,
        "params": params,
        "returns": returns,
        "raises": raises
    }

def infer_type_from_default(default_value) -> str:
    """Infer type from default value.
    
    Args:
        default_value: The default value AST node.
        
    Returns:
        A string representing the Python type.
    """
    if isinstance(default_value, ast.Constant):
        if default_value.value is None:
            return "Optional"
        elif isinstance(default_value.value, bool):
            return "bool"
        elif isinstance(default_value.value, int):
            return "int"
        elif isinstance(default_value.value, float):
            return "float"
        elif isinstance(default_value.value, str):
            return "str"
    elif isinstance(default_value, ast.List):
        return "list"
    elif isinstance(default_value, ast.Dict):
        return "dict"
    elif isinstance(default_value, ast.Set):
        return "set"
    elif isinstance(default_value, ast.Tuple):
        return "tuple"
    return "Any"

def generate_google_docstring(
    old_info: Dict, 
    func_name: str, 
    args: List[ast.arg], 
    returns: Optional[ast.Expr],
    defaults: List
) -> str:
    """Generate a Google-style docstring.
    
    Args:
        old_info: Information extracted from the old docstring.
        func_name: The name of the function.
        args: List of function arguments.
        returns: Return annotation if available.
        defaults: Default values for arguments.
        
    Returns:
        A formatted Google-style docstring.
    """
    description = old_info.get("description", "")
    if not description:
        description = f"{func_name.replace('_', ' ').capitalize()}."
    
    # Split into short and long description
    description_lines = description.split("\n")
    short_desc = description_lines[0]
    long_desc = "\n".join(description_lines[1:]).strip()
    
    # Start building the docstring
    docstring = f'"""{short_desc}'
    if long_desc:
        docstring += f"\n\n{long_desc}"
    
    # Add parameters
    param_info = old_info.get("params", {})
    if args and args[0].arg == "self":
        args = args[1:]  # Skip 'self' for methods
    
    if args:
        if docstring[-1] != "\n":
            docstring += "\n"
        docstring += "\nArgs:\n"
        
        # Calculate default value offsets
        num_defaults = len(defaults)
        num_args = len(args)
        default_offset = num_args - num_defaults
        
        for i, arg in enumerate(args):
            name = arg.arg
            param_desc = param_info.get(name, f"The {name.replace('_', ' ')}.")
            
            # Check if parameter has a default value
            has_default = i >= default_offset
            default_index = i - default_offset if has_default else -1
            default_value = defaults[default_index] if has_default else None
            
            # Check for type annotation or infer from default
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    if isinstance(arg.annotation.value, ast.Name):
                        arg_type = arg.annotation.value.id
                        if arg_type == "Optional":
                            has_default = True
                    else:
                        arg_type = "complex type"
                else:
                    arg_type = "complex type"
            elif has_default:
                arg_type = infer_type_from_default(default_value)
            else:
                arg_type = ""
            
            # Format the parameter line
            if arg_type:
                if has_default:
                    docstring += f"    {name} ({arg_type}, optional): {param_desc}"
                    if default_value and isinstance(default_value, ast.Constant) and default_value.value is not None:
                        docstring += f" Defaults to {ast.unparse(default_value)}."
                    docstring += "\n"
                else:
                    docstring += f"    {name} ({arg_type}): {param_desc}\n"
            else:
                docstring += f"    {name}: {param_desc}\n"
    
    # Add return information
    returns_info = old_info.get("returns", "")
    return_type = ""
    if returns and isinstance(returns, ast.expr):
        if isinstance(returns, ast.Name):
            return_type = returns.id
        elif isinstance(returns, ast.Subscript):
            if isinstance(returns.value, ast.Name):
                return_type = returns.value.id
            else:
                return_type = "complex type"
    
    if returns_info or return_type:
        docstring += "\nReturns:\n"
        if return_type:
            docstring += f"    {return_type}: "
        else:
            docstring += "    "
        if returns_info:
            docstring += f"{returns_info}\n"
        else:
            docstring += "The result of the operation.\n"
    
    # Add exception information
    raises_info = old_info.get("raises", {})
    if raises_info:
        docstring += "\nRaises:\n"
        for exc_type, exc_desc in raises_info.items():
            docstring += f"    {exc_type}: {exc_desc}\n"
    
    docstring += '"""'
    return docstring

def process_function(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Tuple[bool, str]:
    """Process a function node to improve its docstring.
    
    Args:
        node: The AST function node.
        
    Returns:
        A tuple containing whether the docstring was changed and the new docstring.
    """
    old_docstring = ast.get_docstring(node)
    old_info = parse_docstring(old_docstring) if old_docstring else {}
    
    # Get arguments
    args = node.args.args
    defaults = node.args.defaults
    
    # Get return annotation
    returns = node.returns
    
    # Generate new docstring
    new_docstring = generate_google_docstring(old_info, node.name, args, returns, defaults)
    
    # Check if the docstring has changed
    if old_docstring == new_docstring:
        return False, old_docstring
    
    return True, new_docstring

def update_file_docstrings(file_path: str, dry_run: bool = True, debug: bool = False) -> int:
    """Update docstrings in a Python file.
    
    Args:
        file_path: Path to the Python file.
        dry_run: If True, don't actually modify the file. Defaults to True.
        debug: If True, print additional debug information. Defaults to False.
        
    Returns:
        Number of docstrings updated.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse the file
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {str(e)}")
        if debug:
            print(f"Error details: line {e.lineno}, offset {e.offset}")
            if e.text:
                print(f"Error line: {e.text.strip()}")
            import traceback
            traceback.print_exc()
        return 0
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return 0
    
    # Track changes
    changes = []
    
    # Process functions and methods
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if debug:
                print(f"Processing function: {node.name}")
            
            changed, new_docstring = process_function(node)
            if changed:
                if debug:
                    print(f"  Docstring changed for {node.name}")
                
                # Extract the position of the old docstring
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    old_docstring_node = node.body[0]
                    start_line = old_docstring_node.lineno - 1  # 0-indexed for file lines
                    end_line = old_docstring_node.end_lineno
                    changes.append((start_line, end_line, new_docstring))
                else:
                    # No existing docstring, add after the function definition
                    indent = " " * (node.col_offset + 4)  # Indentation for the docstring
                    changes.append((node.lineno, node.lineno, f"\n{indent}{new_docstring}"))
    
    if not changes:
        return 0
    
    # Apply changes (in reverse order to avoid line position changes)
    if not dry_run:
        lines = content.split("\n")
        for start_line, end_line, new_docstring in sorted(changes, reverse=True):
            # Calculate indentation from the first line of the function
            indent = ""
            if start_line > 0:
                function_line = lines[start_line - 1]
                indent = " " * (len(function_line) - len(function_line.lstrip()))
            
            # Format the new docstring with proper indentation
            formatted_docstring = "\n".join(f"{indent}{line}" for line in new_docstring.split("\n"))
            
            # Replace the lines
            lines[start_line:end_line] = [formatted_docstring]
            
            if debug:
                print(f"  Updated lines {start_line} to {end_line}")
        
        # Write the updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    print(f"Updated {len(changes)} docstrings in {file_path}")
    return len(changes)

def main():
    """Run the docstring improvement tool."""
    parser = argparse.ArgumentParser(description="Improve docstrings in Python files")
    parser.add_argument("--path", default=".", help="Directory or file to process")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report changes")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    # Track total changes
    total_changes = 0
    files_changed = 0
    
    # Process files
    if os.path.isfile(args.path) and args.path.endswith(".py"):
        changes = update_file_docstrings(args.path, args.dry_run, args.debug)
        if changes > 0:
            files_changed += 1
            total_changes += changes
    elif os.path.isdir(args.path):
        for root, _, files in os.walk(args.path) if args.recursive else [(args.path, None, os.listdir(args.path))]:
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    changes = update_file_docstrings(file_path, args.dry_run, args.debug)
                    if changes > 0:
                        files_changed += 1
                        total_changes += changes
    
    # Report results
    mode = "Dry run: " if args.dry_run else ""
    print(f"{mode}Updated {total_changes} docstrings in {files_changed} files")

if __name__ == "__main__":
    main() 