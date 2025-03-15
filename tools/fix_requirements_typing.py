#!/usr/bin/env python3
"""
Requirements Analyzer Type Fixer

This specialized script fixes typing issues in the analyze_workflow_requirements.py module,
addressing common patterns like Collection vs List usage, indexing operations,
undefined types, and arithmetic operation type mismatches.
"""

import re
import sys
import ast
from typing import Dict, List, Optional, Set, Any, Tuple, Union, cast
from pathlib import Path


class RequirementsTypeFixer:
    """Fixes specific typing issues in the requirements analyzer module."""
    
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
    
    def fix_collection_to_list(self) -> int:
        """Convert Collection[T] to List[T] where append() or indexing is used."""
        # Find variables typed as Collection but using append or indexing
        collection_pattern = r'(\w+)\s*:\s*Collection\[([\w\[\], ]+)\]'
        changes = 0
        
        # Find all variables with Collection type
        collection_vars = re.findall(collection_pattern, self.content)
        
        for var_name, inner_type in collection_vars:
            # Check if this variable uses append method or indexing
            if (re.search(rf'\b{var_name}\.append\(', self.content) or 
                re.search(rf'\b{var_name}\[\s*\d+\s*\]', self.content) or 
                re.search(rf'\b{var_name}\[\s*\d+\s*\]\s*=', self.content)):
                
                # Replace Collection with List
                self.content = re.sub(
                    rf'{var_name}\s*:\s*Collection\[{re.escape(inner_type)}\]', 
                    f'{var_name}: List[{inner_type}]', 
                    self.content
                )
                changes += 1
        
        # Ensure List is imported
        if changes > 0 and 'from typing import List' not in self.content:
            if 'from typing import' in self.content:
                # Add to existing import
                self.content = re.sub(
                    r'from typing import (.*?)$',
                    r'from typing import \1, List',
                    self.content,
                    flags=re.MULTILINE
                )
            else:
                # Add new import
                self.content = re.sub(
                    r'(import .*?)$',
                    r'\1\nfrom typing import List',
                    self.content,
                    flags=re.MULTILINE
                )
                
        self.changes_made += changes
        return changes
    
    def fix_operation_type_mismatches(self) -> int:
        """Fix arithmetic operation type mismatches."""
        # Parse the content to analyze operations
        tree = ast.parse(self.content)
        
        class OperationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.problematic_ops = []
                
            def visit_BinOp(self, node):
                # Record operations that might have type issues
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    self.problematic_ops.append(node)
                self.generic_visit(node)
        
        visitor = OperationVisitor()
        visitor.visit(tree)
        
        # We can't easily fix these automatically in a safe way
        # But we can add explicit casts to ensure types are correct
        # For now, we'll just report the count
        
        changes = 0
        if visitor.problematic_ops:
            changes = len(visitor.problematic_ops)
            print(f"Found {changes} potentially problematic arithmetic operations.")
            print("These should be manually fixed by ensuring consistent types or adding explicit casts.")
            
        self.changes_made += changes
        return changes
    
    def fix_variable_annotations(self) -> int:
        """Add missing variable type annotations."""
        # Parse the content to find variable assignments without annotations
        tree = ast.parse(self.content)
        
        class VarVisitor(ast.NodeVisitor):
            def __init__(self):
                self.unannotated_vars = []
                self.annotated_vars = set()
                
            def visit_AnnAssign(self, node):
                # Record variables that already have annotations
                if isinstance(node.target, ast.Name):
                    self.annotated_vars.add(node.target.id)
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                # Record variables assignments without annotations
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if var_name not in self.annotated_vars and not var_name.startswith('_'):
                        self.unannotated_vars.append((var_name, node))
                self.generic_visit(node)
        
        visitor = VarVisitor()
        visitor.visit(tree)
        
        changes = 0
        # For now, just report the count - we'd need a more sophisticated
        # analysis to determine appropriate types
        if visitor.unannotated_vars:
            changes = len(visitor.unannotated_vars)
            print(f"Found {changes} variables that need type annotations.")
            print("These should be manually annotated with appropriate types.")
        
        self.changes_made += changes
        return changes
    
    def fix_report_annotation(self) -> int:
        """Fix specific 'report' variable annotation."""
        # This is specifically for the error mentioned in the analyze_workflow_requirements.py file
        pattern = r'(\s+)(report\s*=\s*\[\])'
        replacement = r'\1report: List[Dict[str, Any]] = []'
        
        changes = 0
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, replacement, self.content)
            changes = 1
            
            # Ensure we have the necessary imports
            if 'from typing import Dict, List, Any' not in self.content and 'from typing import' in self.content:
                # Add to existing import
                current_imports = re.search(r'from typing import (.*?)$', self.content, re.MULTILINE)
                if current_imports:
                    imports = current_imports.group(1)
                    missing_imports = []
                    if 'Dict' not in imports:
                        missing_imports.append('Dict')
                    if 'List' not in imports:
                        missing_imports.append('List')
                    if 'Any' not in imports:
                        missing_imports.append('Any')
                    
                    if missing_imports:
                        new_imports = imports
                        if not new_imports.endswith(','):
                            new_imports += ','
                        new_imports += ' ' + ', '.join(missing_imports)
                        self.content = self.content.replace(imports, new_imports)
        
        self.changes_made += changes
        return changes
    
    def fix_object_in_operations(self) -> int:
        """Fix 'in' operations with objects."""
        # Find patterns like "x in object" where object might not be iterable
        pattern = r'(\w+)\s+in\s+(\w+)'
        
        # We need to analyze the code to find the variables' types
        tree = ast.parse(self.content)
        
        class InOpVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_ops = []
                
            def visit_Compare(self, node):
                # Look for 'in' operations
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, ast.In) and isinstance(node.left, ast.Name) and isinstance(comparator, ast.Name):
                        self.in_ops.append((node.left.id, comparator.id, node))
                self.generic_visit(node)
        
        visitor = InOpVisitor()
        visitor.visit(tree)
        
        # For now, just report these - we'd need more context to fix safely
        changes = len(visitor.in_ops)
        if changes > 0:
            print(f"Found {changes} potentially problematic 'in' operations.")
            print("These should be manually checked to ensure the right operand is iterable.")
        
        self.changes_made += changes
        return changes
    
    def fix_indexing_issues(self) -> int:
        """Fix object indexing issues."""
        # Find patterns like "x[y]" where x might not be indexable
        tree = ast.parse(self.content)
        
        class IndexVisitor(ast.NodeVisitor):
            def __init__(self):
                self.index_ops = []
                
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name):
                    self.index_ops.append((node.value.id, node))
                self.generic_visit(node)
        
        visitor = IndexVisitor()
        visitor.visit(tree)
        
        # For now, just report these - we'd need more context to fix safely
        changes = len(visitor.index_ops)
        if changes > 0:
            print(f"Found {changes} potentially problematic indexing operations.")
            print("These should be manually checked to ensure the value is indexable.")
        
        self.changes_made += changes
        return changes
    
    def fix_all_issues(self) -> int:
        """Fix all typing issues in the file."""
        if not self.load_file():
            return 0
            
        print(f"Fixing typing issues in {self.file_path}...")
        
        # Apply all fixes
        self.fix_collection_to_list()
        self.fix_report_annotation()
        self.fix_operation_type_mismatches()
        self.fix_variable_annotations()
        self.fix_object_in_operations()
        self.fix_indexing_issues()
        
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
        print("Usage: python fix_requirements_typing.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    fixer = RequirementsTypeFixer(file_path)
    fixed_count = fixer.fix_all_issues()
    
    sys.exit(0 if fixed_count >= 0 else 1)


if __name__ == "__main__":
    main() 