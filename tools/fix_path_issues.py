#!/usr/bin/env python
"""Fix common path-related issues detected by Ruff."""

import re
import sys
from pathlib import Path


def fix_file(file_path: str) -> bool:
    """Fix path-related issues in a file."""
    # Skip if file doesn't exist
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return False

    # Read file content
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        original_content = content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Apply fixes
    # PTH123: Replace open() with Path.open()
    content = re.sub(r"with\s+open\s*\(\s*([^,\)]+)(.+)\)\s+as\s+([^:]+):", r"with Path(\1).open(\2) as \3:", content)

    # PTH110: Replace os.path.exists() with Path.exists()
    content = re.sub(r"os\.path\.exists\s*\(\s*([^)]+)\s*\)", r"Path(\1).exists()", content)

    # PTH112: Replace os.path.isdir() with Path.is_dir()
    content = re.sub(r"os\.path\.isdir\s*\(\s*([^)]+)\s*\)", r"Path(\1).is_dir()", content)

    # PTH113: Replace os.path.isfile() with Path.is_file()
    content = re.sub(r"os\.path\.isfile\s*\(\s*([^)]+)\s*\)", r"Path(\1).is_file()", content)

    # PTH118: Replace os.path.join() with Path
    content = re.sub(r"os\.path\.join\s*\(\s*([^,)]+)\s*,\s*([^,)]+)\s*\)", r"Path(\1) / \2", content)

    # PTH119: Replace os.path.basename() with Path.name
    content = re.sub(r"os\.path\.basename\s*\(\s*([^)]+)\s*\)", r"Path(\1).name", content)

    # PTH207: Replace glob.glob with Path.glob or Path.rglob
    content = re.sub(r"glob\.glob\s*\(\s*([^,)]+)\s*,\s*recursive=True\s*\)", r"Path().rglob(\1)", content)
    content = re.sub(r"glob\.glob\s*\(\s*([^,)]+)\s*\)", r"Path().glob(\1)", content)

    # Special fix for Path objects in f-strings
    content = content.replace('f"', 'f"')

    # Add pathlib import if missing
    if "from pathlib import Path" not in content and "import pathlib" not in content:
        # Find the import section
        import_match = re.search(r"^import.*$|^from.*import", content, re.MULTILINE)
        if import_match:
            # Insert after the last import
            last_import = re.findall(r"^(?:import|from).*$", content, re.MULTILINE)[-1]
            content = content.replace(last_import, last_import + "\nfrom pathlib import Path")
        else:
            # Insert at the beginning, after any module docstring
            docstring_match = re.match(r'^""".*?"""', content, re.DOTALL)
            if docstring_match:
                docstring_end = docstring_match.end()
                content = content[:docstring_end] + "\nfrom pathlib import Path\n" + content[docstring_end:]
            else:
                content = "from pathlib import Path\n\n" + content

    # Write changes if needed
    if content != original_content:
        try:
            Path(file_path).write_text(content, encoding="utf-8")
            print(f"✅ Fixed path issues in {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    else:
        print(f"No changes needed for {file_path}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default to fixing all Python files in the tools directory
        files = list(Path("tools").rglob("*.py"))

    fixed_count = 0
    for file_path in files:
        if fix_file(str(file_path)):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")


if __name__ == "__main__":
    main()
