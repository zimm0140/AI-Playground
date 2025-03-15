#!/usr/bin/env python3
"""
Simulation Module Type Fixer

This specialized script fixes typing issues in the simulate_workflow_execution.py module,
which has complex typing patterns and requires careful handling of optional values,
indexing operations, and attribute access on potentially None values.
"""

import re
import sys
import ast
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path


class SimulationTypeFixer:
    """Fixes specific typing issues in the simulation module."""
    
    def __init__(self, file_path: str):
        """Initialize with the target file path."""
        self.file_path = file_path
        self.content = ""
        self.fixed_content = ""
        self.changes_made = 0
    
    def load_file(self) -> bool:
        """Load the file content."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def save_file(self) -> bool:
        """Save the fixed content back to the file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.fixed_content)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def fix_optional_subscripting(self) -> int:
        """Fix issues with subscripting Optional values."""
        pattern = r'(\w+)(?:\s*\[\s*[\w\.]+\s*\]|\.\w+)\s+(?:where|and|if)\s+\1\s+is\s+None'
        
        def replacement(match):
            var_name = match.group(1)
            return f"{var_name} is not None and {var_name}"
        
        new_content = re.sub(pattern, replacement, self.content)
        changes = sum(1 for a, b in zip(self.content.splitlines(), new_content.splitlines()) if a != b)
        self.content = new_content
        self.changes_made += changes
        return changes
    
    def fix_collection_append(self) -> int:
        """Fix issues with append operations on Collection types."""
        # Find variables typed as Collection or Sequence but using append
        pattern = r'(\w+)\s*:\s*(?:Collection|Sequence)\[([\w\[\], ]+)\]'
        append_pattern = r'(\w+)\.append\('
        
        # Find all variables with Collection or Sequence type
        collection_vars = re.findall(pattern, self.content)
        changes = 0
        
        for var_name, inner_type in collection_vars:
            # Check if this variable uses append method
            if re.search(rf'\b{var_name}\.append\(', self.content):
                # Replace Collection/Sequence with List
                self.content = re.sub(
                    rf'{var_name}\s*:\s*(?:Collection|Sequence)\[{re.escape(inner_type)}\]', 
                    f'{var_name}: List[{inner_type}]', 
                    self.content
                )
                changes += 1
        
        # Ensure List is imported
        if changes > 0 and 'from typing import List' not in self.content:
            import_match = re.search(r'from typing import (.*?)(?:\n|$)', self.content)
            if import_match:
                imports = import_match.group(1)
                if 'List' not in imports:
                    new_imports = f"{imports}, List"
                    self.content = self.content.replace(imports, new_imports)
            else:
                # Add List to imports
                self.content = re.sub(
                    r'from typing import (.*?)(?:\n|$)',
                    r'from typing import \1, List\n',
                    self.content
                )
        
        self.changes_made += changes
        return changes
    
    def add_null_checks(self) -> int:
        """Add explicit None checks before indexing or attribute access."""
        # Parse the content to find index/attribute access on variables
        tree = ast.parse(self.content)
        
        class NullAccessVisitor(ast.NodeVisitor):
            def __init__(self):
                self.accesses = []
                
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name):
                    # Record variables being subscripted
                    self.accesses.append((node.value.id, node.lineno, 'subscript'))
                self.generic_visit(node)
                
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    # Record variables being accessed for attributes
                    self.accesses.append((node.value.id, node.lineno, 'attribute'))
                self.generic_visit(node)
        
        visitor = NullAccessVisitor()
        visitor.visit(tree)
        
        # Group by line number to avoid duplicates
        access_by_line = {}
        for var, line, access_type in visitor.accesses:
            if line not in access_by_line:
                access_by_line[line] = set()
            access_by_line[line].add((var, access_type))
        
        # Now modify the content line by line
        lines = self.content.splitlines()
        changes = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            if line_num in access_by_line:
                # Check if line already has a None check
                if "is not None" in line or "if None in" in line:
                    continue
                    
                # Check if this line is in an if condition that might already check for None
                if line.strip().startswith(('if ', 'elif ')):
                    continue
                
                # Add None checks for the variables on this line
                for var, access_type in access_by_line[line_num]:
                    # Skip common variables that we know are not None
                    if var in ('self', 'cls', 'os', 'sys', 'datetime', 're'):
                        continue
                        
                    # Look for pattern: variable[...] or variable.attr
                    pattern = rf'\b{var}\s*(?:\[|\.)' 
                    if re.search(pattern, line):
                        # Check if line already has a condition with this variable
                        if re.search(rf'\b{var}\s+(?:is|==|!=|in)', line):
                            continue
                            
                        # Add explicit None check by modifying the line
                        mod_line = line.rstrip()
                        indentation = len(line) - len(line.lstrip())
                        indent = line[:indentation]
                        
                        # Add if statement with null check
                        new_line = f"{indent}if {var} is not None:\n{indent}    {line.lstrip()}"
                        lines[i] = new_line
                        changes += 1
                        break  # Only add one check per line to avoid nested issues
        
        if changes > 0:
            self.content = '\n'.join(lines)
        
        self.changes_made += changes
        return changes
    
    def fix_type_annotations(self) -> int:
        """Fix incorrect type annotations."""
        changes = 0
        
        # Fix 'dict' instead of 'Dict'
        dict_pattern = r'(\w+)\s*:\s*dict\[(.*?)\]'
        dict_repl = r'\1: Dict[\2]'
        new_content = re.sub(dict_pattern, dict_repl, self.content)
        dict_changes = sum(1 for a, b in zip(self.content.splitlines(), new_content.splitlines()) if a != b)
        changes += dict_changes
        self.content = new_content
        
        # Fix 'list' instead of 'List'
        list_pattern = r'(\w+)\s*:\s*list\[(.*?)\]'
        list_repl = r'\1: List[\2]'
        new_content = re.sub(list_pattern, list_repl, self.content)
        list_changes = sum(1 for a, b in zip(self.content.splitlines(), new_content.splitlines()) if a != b)
        changes += list_changes
        self.content = new_content
        
        # Fix missing Optional for None defaults
        optional_pattern = r'(\w+)\s*:\s*([^=\n]+?)\s*=\s*None'
        
        def optional_repl(match):
            var_name = match.group(1)
            type_name = match.group(2).strip()
            if type_name.startswith('Optional['):
                return match.group(0)  # Already using Optional
            return f"{var_name}: Optional[{type_name}] = None"
        
        new_content = re.sub(optional_pattern, optional_repl, self.content)
        optional_changes = sum(1 for a, b in zip(self.content.splitlines(), new_content.splitlines()) if a != b)
        changes += optional_changes
        self.content = new_content
        
        # Ensure we import all needed types
        if changes > 0:
            import_types = []
            if dict_changes > 0:
                import_types.append('Dict')
            if list_changes > 0:
                import_types.append('List')
            if optional_changes > 0:
                import_types.append('Optional')
                
            if import_types:
                # Check if we have typing imports
                typing_import = re.search(r'from typing import (.*?)$', self.content, re.MULTILINE)
                if typing_import:
                    # Add to existing import
                    imports = typing_import.group(1)
                    for import_type in import_types:
                        if import_type not in imports:
                            # Append to existing imports
                            new_imports = imports
                            if not new_imports.endswith(','):
                                new_imports += ','
                            new_imports += f" {import_type}"
                            self.content = self.content.replace(imports, new_imports)
                else:
                    # Add new import
                    import_line = f"from typing import {', '.join(import_types)}\n"
                    # Try to add after first imports
                    first_import = re.search(r'^import .*?$', self.content, re.MULTILINE)
                    if first_import:
                        # Add after existing import
                        pos = first_import.end()
                        self.content = self.content[:pos] + '\n' + import_line + self.content[pos:]
                    else:
                        # Add at beginning of file, after any docstring
                        docstring_match = re.search(r'^""".*?"""$', self.content, re.MULTILINE | re.DOTALL)
                        if docstring_match:
                            pos = docstring_match.end()
                            self.content = self.content[:pos] + '\n\n' + import_line + self.content[pos:]
                        else:
                            # Add at the very beginning
                            self.content = import_line + self.content
        
        self.changes_made += changes
        return changes
    
    def fix_path_endswith(self) -> int:
        """Fix Path.endswith() calls to use Path.suffix instead."""
        pattern = r'(\w+)\.endswith\([\'\"](.*?)[\'\"]'
        
        def repl(match):
            path_var = match.group(1)
            extension = match.group(2)
            if not extension.startswith('.'):
                extension = '.' + extension
            return f"{path_var}.suffix == '{extension}'"
        
        new_content = re.sub(pattern, repl, self.content)
        changes = sum(1 for a, b in zip(self.content.splitlines(), new_content.splitlines()) if a != b)
        self.content = new_content
        self.changes_made += changes
        return changes
    
    def fix_index_errors(self) -> int:
        """Fix invalid index operations on str objects."""
        # This is trickier - we need to find string indexing with string keys
        # First parse the code to find subscript operations
        tree = ast.parse(self.content)
        
        class StringIndexVisitor(ast.NodeVisitor):
            def __init__(self):
                self.string_index_lines = set()
                
            def visit_Subscript(self, node):
                # Check if we're indexing with a string into a string
                if (hasattr(node, 'slice') and 
                    isinstance(node.slice, ast.Constant) and 
                    isinstance(node.slice.value, str)):
                    self.string_index_lines.add(node.lineno)
                self.generic_visit(node)
        
        visitor = StringIndexVisitor()
        visitor.visit(tree)
        
        # Now fix the string indexing operations
        lines = self.content.splitlines()
        changes = 0
        
        for line_num in visitor.string_index_lines:
            i = line_num - 1
            if i < len(lines):
                line = lines[i]
                # Replace string indexing with get() method
                pattern = r'(\w+)\[([\'"].*?[\'"]\])'
                if re.search(pattern, line):
                    new_line = re.sub(pattern, r'\1.get(\2, "")', line)
                    lines[i] = new_line
                    changes += 1
        
        if changes > 0:
            self.content = '\n'.join(lines)
        
        self.changes_made += changes
        return changes
    
    def fix_all_issues(self) -> int:
        """Fix all typing issues in the file."""
        if not self.load_file():
            return 0
            
        print(f"Fixing typing issues in {self.file_path}...")
        
        # Apply all fixes
        self.fix_optional_subscripting()
        self.fix_collection_append()
        self.fix_type_annotations()
        self.fix_path_endswith()
        self.fix_index_errors()
        self.add_null_checks()
        
        # Save the fixed content
        self.fixed_content = self.content
        if self.changes_made > 0:
            if self.save_file():
                print(f"Fixed {self.changes_made} typing issues in {self.file_path}")
            else:
                print(f"Failed to save fixed file {self.file_path}")
        else:
            print(f"No typing issues to fix in {self.file_path}")
            
        return self.changes_made


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_simulation_typing.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    fixer = SimulationTypeFixer(file_path)
    fixed_count = fixer.fix_all_issues()
    
    sys.exit(0 if fixed_count >= 0 else 1)


if __name__ == "__main__":
    main() 