#!/usr/bin/env python
"""Fix issues with Path.open() calls missing the mode parameter."""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """Fix Path.open() calls in a file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Fix missing mode parameter in Path.open()
    content = re.sub(r"Path\(([^)]+)\)\.open\(\s*,", r"Path(\1).open(", content)
    content = re.sub(r"Path\(([^)]+)\)\.open\(\s*,\s*\"", r"Path(\1).open(\"", content)

    # Fix incorrect variable names in tools/scripts/uvfast.py
    content = re.sub(r"Path\(config_pat\)\.open\(h\)", r"Path(config_path).open()", content)

    # Fix other specific patterns as needed
    content = re.sub(r"Path\(schema_fil\)\.open\(e\)", r"Path(schema_file).open()", content)
    content = re.sub(r"Path\(workflow_fil\)\.open\(e\)", r"Path(workflow_file).open()", content)
    content = re.sub(r"Path\(self\.config_fil\)\.open\(e\)", r"Path(self.config_file).open()", content)

    # Write changes if needed
    if content != original_content:
        try:
            file_path.write_text(content, encoding="utf-8")
            print(f"✅ Fixed Path.open() issues in {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    else:
        return False


def main():
    """Main function."""
    # Get python files in the tools directory
    tools_dir = Path("tools")
    fixed_count = 0

    # Search for Python files in the tools directory and subdirectories
    for py_file in tools_dir.glob("**/*.py"):
        if fix_file(py_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")


if __name__ == "__main__":
    main()
