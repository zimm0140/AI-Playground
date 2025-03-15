#!/usr/bin/env python3
"""
Auto Type Annotation Fixer

This script automatically adds type annotations to Python functions based on:
1. Static analysis of the code
2. Inference from usage patterns
3. Basic type annotations for common patterns

It's designed to handle the most common typing issues:
- Missing return type annotations
- Missing parameter type annotations
"""

import os
import re
import ast
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Iterator, Counter as CounterType
from pathlib import Path


class FunctionTypeCollector(ast.NodeVisitor):
    """AST visitor to collect information about function usage and types."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions_defined: Dict[str, Dict[str, Any]] = {}
        self.parameter_usages: Dict[str, Dict[str, Set[str]]] = {}
        self.return_values: Dict[str, Set[str]] = {}
        self.current_function: Optional[str] = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to collect type information."""
        function_name = node.name
        self.current_function = function_name
        
        # Collect function definition information
        param_info: Dict[str, Dict[str, Any]] = {}
        for arg in node.args.args:
            param_name = arg.arg
            param_info[param_name] = {
                "has_annotation": arg.annotation is not None,
                "annotation": None if arg.annotation is None else ast.unparse(arg.annotation),
                "usages": set()
            }
        
        self.functions_defined[function_name] = {
            "params": param_info,
            "has_return_annotation": node.returns is not None,
            "return_annotation": None if node.returns is None else ast.unparse(node.returns),
            "docstring": ast.get_docstring(node),
            "line": node.lineno
        }
        
        # Initialize parameter usage tracking
        self.parameter_usages[function_name] = {param: set() for param in param_info}
        self.return_values[function_name] = set()
        
        # Visit function body
        self.generic_visit(node)
        self.current_function = None
    
    def visit_Name(self, node: ast.Name) -> None:
        """Visit name references to track parameter usage."""
        if isinstance(node.ctx, ast.Load) and self.current_function:
            # Check if this is a parameter being used
            if (self.current_function in self.parameter_usages and 
                node.id in self.parameter_usages[self.current_function]):
                # Record the context of usage
                if isinstance(node.parent, ast.Call) and node.parent.func == node:
                    # Parameter used as a function
                    self.parameter_usages[self.current_function][node.id].add("callable")
                elif (isinstance(node.parent, ast.BinOp) and 
                      (node.parent.left == node or node.parent.right == node)):
                    # Parameter used in a binary operation
                    if isinstance(node.parent.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        self.parameter_usages[self.current_function][node.id].add("numeric")
                    elif isinstance(node.parent.op, (ast.Mod)):
                        self.parameter_usages[self.current_function][node.id].add("string")
                elif isinstance(node.parent, ast.Subscript) and node.parent.value == node:
                    # Parameter used as a container to be indexed
                    self.parameter_usages[self.current_function][node.id].add("subscriptable")
                elif isinstance(node.parent, ast.Attribute) and node.parent.value == node:
                    # Parameter used as an object with attributes
                    self.parameter_usages[self.current_function][node.id].add("object")
                elif isinstance(node.parent, ast.Compare):
                    # Parameter used in a comparison
                    self.parameter_usages[self.current_function][node.id].add("comparable")
        
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statements to infer return types."""
        if self.current_function and node.value:
            return_type = self._infer_type_from_node(node.value)
            if return_type:
                self.return_values[self.current_function].add(return_type)
        
        self.generic_visit(node)
    
    def _infer_type_from_node(self, node: ast.AST) -> Optional[str]:
        """Infer a Python type from an AST node."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict" 
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.NameConstant):
            if node.value is None:
                return "None"
            return str(type(node.value).__name__)
        elif isinstance(node, ast.Name):
            if node.id == "True" or node.id == "False":
                return "bool"
            # Can't infer the type just from the name
            return None
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Calls to known constructors
                if node.func.id in ["list", "dict", "set", "tuple", "str", "int", "float", "bool"]:
                    return node.func.id
                elif node.func.id == "open":
                    return "TextIO"
            return None
        elif isinstance(node, ast.BinOp):
            # Attempt to infer type from binary operations
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                left_type = self._infer_type_from_node(node.left)
                right_type = self._infer_type_from_node(node.right)
                
                # String concatenation
                if left_type == "str" or right_type == "str":
                    return "str"
                # Numeric operations
                elif left_type in ["int", "float"] or right_type in ["int", "float"]:
                    # Division typically results in float
                    if isinstance(node.op, ast.Div):
                        return "float"
                    # If any operand is float, result is float
                    elif left_type == "float" or right_type == "float":
                        return "float"
                    else:
                        return "int"
            return None
        elif isinstance(node, ast.UnaryOp):
            # Unary operations often preserve type
            operand_type = self._infer_type_from_node(node.operand)
            return operand_type
        
        # Fallback: can't determine type
        return None


def infer_parameter_type(usages: Set[str]) -> Optional[str]:
    """Infer parameter type from its usage patterns.
    
    Args:
        usages: Set of usage contexts for the parameter
        
    Returns:
        Inferred type annotation or None if can't be determined
    """
    if not usages:
        return None
    
    if "callable" in usages:
        return "Callable"
    elif "subscriptable" in usages:
        if "numeric" in usages:
            return "List[float]"  # List of numbers is a common case
        return "List[Any]"
    elif "object" in usages:
        return "object"  # Generic object type
    elif "numeric" in usages:
        return "float"  # Default to float for numeric operations
    elif "string" in usages:
        return "str"
    elif "comparable" in usages:
        return "Any"  # Can't determine specific comparable type
    
    return None


def infer_return_type(return_values: Set[str]) -> str:
    """Infer function return type from collected return statements.
    
    Args:
        return_values: Set of possible return value types
        
    Returns:
        Inferred return type annotation
    """
    if not return_values:
        return "None"
    
    # If None is a potential return value
    if "None" in return_values:
        other_types = return_values - {"None"}
        if len(other_types) == 1:
            return f"Optional[{next(iter(other_types))}]"
        elif other_types:
            return "Optional[Any]"
        else:
            return "None"
    
    # Single clear return type
    if len(return_values) == 1:
        return next(iter(return_values))
    
    # Mixed numeric types
    if return_values.issubset({"int", "float"}):
        return "float"
    
    # Multiple distinct types
    return "Any"


def analyze_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Analyze a Python file to collect function information.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary of function data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse with AST for high-level analysis
        tree = ast.parse(content, filename=file_path)
        collector = FunctionTypeCollector(file_path)
        
        # Add parent pointers for AST nodes
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
                
        collector.visit(tree)
        
        function_data = {}
        
        # Process collected data
        for func_name, data in collector.functions_defined.items():
            # Skip functions that already have full type annotations
            if data["has_return_annotation"] and all(
                param_data["has_annotation"] 
                for param_name, param_data in data["params"].items()
                if param_name not in ["self", "cls"]
            ):
                continue
            
            function_data[func_name] = {
                "param_types": {},
                "line": data["line"]
            }
            
            # Infer parameter types
            if func_name in collector.parameter_usages:
                for param_name, usages in collector.parameter_usages[func_name].items():
                    # Skip self/cls and already annotated parameters
                    if param_name in ["self", "cls"] or data["params"][param_name]["has_annotation"]:
                        continue
                    
                    inferred_type = infer_parameter_type(usages)
                    if inferred_type:
                        function_data[func_name]["param_types"][param_name] = inferred_type
            
            # Infer return type if not already annotated
            if not data["has_return_annotation"] and func_name in collector.return_values:
                return_type = infer_return_type(collector.return_values[func_name])
                function_data[func_name]["return_type"] = return_type
        
        return function_data
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {}


def add_type_annotations_with_regex(file_path: str, function_data: Dict[str, Dict[str, Any]]) -> int:
    """Add type annotations to functions in a file using regex-based replacements.
    
    Args:
        file_path: Path to the Python file
        function_data: Dictionary of function data with inferred types
        
    Returns:
        Number of annotations added
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if any changes need to be made
        if not function_data:
            return 0
        
        # Count annotations added
        annotations_added = 0
        
        # Make a copy of the content to modify
        modified_content = content
        
        # Process functions in reverse order of line number to avoid offset issues
        sorted_functions = sorted(
            function_data.items(), 
            key=lambda x: x[1]["line"], 
            reverse=True
        )
        
        for func_name, data in sorted_functions:
            # Define regex pattern to match the function definition
            # This handles both simple and complex function definitions
            pattern = r'(def\s+' + re.escape(func_name) + r'\s*\()(.*?)(\)\s*:)'
            
            matches = list(re.finditer(pattern, modified_content, re.DOTALL))
            if not matches:
                continue
                
            # Take the last match in case there are multiple functions with the same name
            match = matches[-1]
            
            # Extract parameter section
            params_text = match.group(2)
            
            # Add parameter annotations
            modified_params = params_text
            for param_name, param_type in data.get("param_types", {}).items():
                # Only annotate parameters that don't already have annotations
                param_pattern = r'(\b' + re.escape(param_name) + r'\b)(?!\s*:)'
                if re.search(param_pattern, modified_params):
                    modified_params = re.sub(
                        param_pattern,
                        r'\1: ' + param_type,
                        modified_params
                    )
                    annotations_added += 1
            
            # Add return annotation if needed
            return_annotation = ""
            if "return_type" in data:
                return_annotation = f" -> {data['return_type']}"
                annotations_added += 1
            
            # Ensure any whitespace between ) and : is preserved
            closing_paren_to_colon = match.group(3)
            # Split at the colon to preserve any whitespace
            parts = closing_paren_to_colon.split(':')
            if len(parts) == 2:
                # Add the return annotation before the colon
                replacement = f"{match.group(1)}{modified_params}{parts[0]}{return_annotation}:{parts[1]}"
            else:
                # Fallback if we can't parse it properly
                replacement = f"{match.group(1)}{modified_params}{closing_paren_to_colon.rstrip(':')}{return_annotation}:"
            
            # Replace the function definition in the content
            modified_content = (
                modified_content[:match.start()] + 
                replacement + 
                modified_content[match.end():]
            )
        
        # Write modified content back to file if changes were made
        if annotations_added > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        
        return annotations_added
    
    except Exception as e:
        print(f"Error modifying {file_path}: {str(e)}")
        return 0


def fix_typing_issues(file_path: str, verbose: bool = False) -> int:
    """Fix typing issues in a Python file.
    
    Args:
        file_path: Path to the Python file
        verbose: Whether to print verbose output
        
    Returns:
        Number of annotations added
    """
    if verbose:
        print(f"Analyzing {file_path}...")
        
    function_data = analyze_file(file_path)
    
    if verbose:
        print(f"Found {len(function_data)} functions to add annotations to.")
        
    annotations_added = add_type_annotations_with_regex(file_path, function_data)
    
    if verbose:
        print(f"Added {annotations_added} type annotations to {file_path}")
        
    return annotations_added


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
    parser = argparse.ArgumentParser(description="Automatically add type annotations to Python code")
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
    
    total_annotations = 0
    total_files_fixed = 0
    
    for file_path in files:
        annotations_added = fix_typing_issues(file_path, verbose)
        if annotations_added > 0:
            total_annotations += annotations_added
            total_files_fixed += 1
            
    print(f"Added {total_annotations} type annotations to {total_files_fixed} files")


if __name__ == "__main__":
    main() 