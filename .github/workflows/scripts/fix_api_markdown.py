#!/usr/bin/env python3
"""
Script to fix formatting issues in the API reference markdown file.
This addresses:
- Code block language specifier issues
- Indentation and line breaks in complex JSON examples
- Proper backtick escaping
"""

import re
from pathlib import Path


def fix_api_markdown():
    """Fix formatting issues in the API reference markdown file."""
    api_md_path = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "reference" / "api.md"

    if not api_md_path.exists():
        print(f"API markdown file not found at {api_md_path}")
        return False

    with open(api_md_path, encoding="utf-8") as f:
        content = f.read()

    # Fix 1: Add language specifiers to code blocks
    content = re.sub(r"```\s*\n", "```text\n", content)

    # Fix 2: Fix broken JSON formatting in code blocks
    # This regex finds JSON code blocks that have broken formatting
    json_code_block_pattern = r"```json\n(.*?)```"

    def fix_json_block(match):
        json_content = match.group(1)
        # Fix common issues like missing line breaks and indentation
        json_content = re.sub(r'{\s*"', '{\n  "', json_content)
        json_content = re.sub(r',\s*"', ',\n  "', json_content)
        json_content = re.sub(r"}\s*,\s*{", "},\n{", json_content)
        json_content = re.sub(r"}\s*$", "\n}", json_content)
        return f"```json\n{json_content}```"

    content = re.sub(json_code_block_pattern, fix_json_block, content, flags=re.DOTALL)

    # Fix 3: Fix backtick escaping and code block nesting
    # Find nested code blocks and ensure proper formatting
    nested_code_pattern = r"```text\n```(\w+)\n"
    content = re.sub(nested_code_pattern, r"```\1\n", content)

    # Write back the fixed content
    with open(api_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed API markdown file at {api_md_path}")
    return True


if __name__ == "__main__":
    fix_api_markdown()