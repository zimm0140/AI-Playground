#!/usr/bin/env python3
"""
Script to automatically fix common markdown linting issues.

This script addresses:
- MD013: Line length
- MD025: Multiple top-level headings
- Various whitespace issues
"""

import subprocess
import sys
from pathlib import Path


def fix_markdown_line_length(file_path, max_line_length=180):
    """Fix line length issues in markdown files by wrapping text."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        new_lines = []

        # Keep track of code blocks as we don't want to modify those
        in_code_block = False

        for line in lines:
            # Toggle code block status
            if line.startswith("```"):
                in_code_block = not in_code_block
                new_lines.append(line)
                continue

            # Don't modify lines in code blocks, headings, or lists
            if (
                in_code_block
                or line.startswith("#")
                or line.startswith("-")
                or line.startswith("*")
                or len(line) <= max_line_length
                or not line.strip()
            ):
                new_lines.append(line)
                continue

            # Basic text wrapping for long lines
            words = line.split()
            new_line = words[0]
            for word in words[1:]:
                if len(new_line + " " + word) <= max_line_length:
                    new_line += " " + word
                else:
                    new_lines.append(new_line)
                    new_line = word
            new_lines.append(new_line)

        new_content = "\n".join(new_lines)

        # Ensure file ends with a single newline
        new_content = new_content.rstrip("\n") + "\n"

        # Write changes if needed
        if new_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_multiple_h1(file_path):
    """Fix multiple top-level headings (h1) in markdown files."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        new_lines = []

        found_h1 = False

        for line in lines:
            if line.strip().startswith("# ") and found_h1:
                # Convert subsequent h1 to h2
                new_lines.append(line.replace("# ", "## "))
            else:
                if line.strip().startswith("# "):
                    found_h1 = True
                new_lines.append(line)

        new_content = "\n".join(new_lines)

        # Write changes if needed
        if new_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def run_mdformat(file_path):
    """Run mdformat on the given file to fix formatting issues."""
    try:
        subprocess.run(
            [sys.executable, "-m", "mdformat", file_path],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_markdown_files():
    """Find all Markdown files in the repository."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    md_files = list(repo_root.glob("**/*.md"))

    # Filter out virtual environments or other non-project paths
    excluded_patterns = [".venv", "venv", ".git", "__pycache__", "node_modules", "build", "dist"]

    return [str(path) for path in md_files if not any(pattern in str(path) for pattern in excluded_patterns)]


def main():
    """Find and fix linting issues in Markdown files."""
    md_files = find_markdown_files()
    print(f"Found {len(md_files)} Markdown files to process")

    fixed_count = 0
    for file_path in md_files:
        # Apply fixes
        line_length_fixed = fix_markdown_line_length(file_path)
        h1_fixed = fix_multiple_h1(file_path)

        # Try to run mdformat if available
        mdformat_fixed = run_mdformat(file_path)

        if line_length_fixed or h1_fixed or mdformat_fixed:
            fixed_count += 1
            print(f"✅ Fixed issues in {file_path}")
        else:
            print(f"✓ No fixable issues in {file_path}")

    print(f"\n✅ Fixed issues in {fixed_count} files")
    print("\nNote: Some markdown issues may require manual review.")
    print("Consider installing markdownlint-cli for more comprehensive checks.")


if __name__ == "__main__":
    main()
