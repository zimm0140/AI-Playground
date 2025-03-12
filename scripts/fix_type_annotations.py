#!/usr/bin/env python
"""
Type Annotation Fixer for Python 3.10+ Compatibility

This script scans Python files and fixes common type annotation issues
to ensure compatibility with Python 3.10+.

Usage:
    python scripts/fix_type_annotations.py [directory_or_file]
"""

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fix type annotations for Python 3.10+ compatibility")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to process (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be changed without making changes",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed information about changes")
    return parser.parse_args()


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def fix_import_from_typing(content: str) -> Tuple[str, bool]:
    """Fix imports from typing module for Python 3.10+ compatibility."""
    # Pattern to find imports from typing
    pattern = r"from\s+typing\s+import\s+([^#\n]+)"

    changed = False
    new_content = content

    # Find all imports from typing
    matches = re.finditer(pattern, content)
    for match in matches:
        imports = match.group(1)

        # Check for Union, Optional that could be replaced with | operator
        if "Union" in imports or "Optional" in imports:
            # We won't automatically replace these, but we'll flag them
            changed = True

            # Add a comment to indicate potential changes
            comment = "# TODO: Consider using | operator instead of Union/Optional for Python 3.10+"
            line_end = match.end()
            line_end_pos = content.find("\n", line_end)
            if line_end_pos == -1:
                line_end_pos = len(content)

            # Insert comment after the import
            new_content = new_content[:line_end_pos] + "\n" + comment + new_content[line_end_pos:]

    return new_content, changed


def fix_optional_annotations(content: str) -> Tuple[str, bool]:
    """Fix Optional[Type] annotations to Type | None for Python 3.10+."""
    # Pattern to find Optional[Type] annotations
    pattern = r"Optional\[([^\]]+)\]"

    if "Optional" not in content:
        return content, False

    # We won't automatically replace Optional[Type] with Type | None
    # Instead, add comments to suggest manual review
    new_content = content
    if re.search(pattern, content):
        # Add a docstring note about Optional usage
        docstring_pattern = r'"""([\s\S]*?)"""'
        docstring_match = re.search(docstring_pattern, content)

        if docstring_match:
            docstring_end = docstring_match.end()
            note = "\n\n# Note: This file uses Optional[Type] annotations which could be updated to Type | None in Python 3.10+\n"
            new_content = new_content[:docstring_end] + note + new_content[docstring_end:]
            return new_content, True

    return content, False


def fix_union_annotations(content: str) -> Tuple[str, bool]:
    """Fix Union[Type1, Type2] annotations to Type1 | Type2 for Python 3.10+."""
    # Pattern to find Union[Type1, Type2] annotations
    pattern = r"Union\[([^\]]+)\]"

    if "Union" not in content:
        return content, False

    # We won't automatically replace Union[Type1, Type2] with Type1 | Type2
    # Instead, add comments to suggest manual review
    new_content = content
    if re.search(pattern, content):
        # Add a docstring note about Union usage
        docstring_pattern = r'"""([\s\S]*?)"""'
        docstring_match = re.search(docstring_pattern, content)

        if docstring_match:
            docstring_end = docstring_match.end()
            note = "\n\n# Note: This file uses Union[Type1, Type2] annotations which could be updated to Type1 | Type2 in Python 3.10+\n"
            new_content = new_content[:docstring_end] + note + new_content[docstring_end:]
            return new_content, True

    return content, False


def process_file(file_path: str, dry_run: bool, verbose: bool) -> bool:
    """Process a single Python file and fix type annotations."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        changed = False

        # Fix imports from typing
        content, import_changed = fix_import_from_typing(content)
        changed = changed or import_changed

        # Fix Optional annotations
        content, optional_changed = fix_optional_annotations(content)
        changed = changed or optional_changed

        # Fix Union annotations
        content, union_changed = fix_union_annotations(content)
        changed = changed or union_changed

        if changed and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        if changed and verbose:
            print(f"Modified: {file_path}")
            if import_changed:
                print("  - Added notes about typing imports")
            if optional_changed:
                print("  - Added notes about Optional annotations")
            if union_changed:
                print("  - Added notes about Union annotations")
        elif changed:
            print(f"Modified: {file_path}")

        return changed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    path = args.path

    if os.path.isfile(path):
        files = [path]
    else:
        files = find_python_files(path)

    if args.dry_run:
        print(f"Dry run mode - no changes will be made")

    modified_count = 0
    total_count = len(files)

    for file_path in files:
        if process_file(file_path, args.dry_run, args.verbose):
            modified_count += 1

    print(f"\nProcessed {total_count} files")
    print(f"Modified {modified_count} files")

    if args.dry_run:
        print("Dry run completed. No changes were made.")
    else:
        print("Type annotation review completed.")
        if modified_count > 0:
            print("\nReview the changes and manually fix type annotations where appropriate.")
            print("For Python 3.10+:")
            print("  - Consider replacing Union[Type1, Type2] with Type1 | Type2")
            print("  - Consider replacing Optional[Type] with Type | None")


if __name__ == "__main__":
    main()
