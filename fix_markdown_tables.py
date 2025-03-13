#!/usr/bin/env python3
"""
Script to fix markdown table formatting in workflows-document.md.
It ensures all tables have proper leading and trailing pipe characters
and fixes any column count issues.
"""

import sys


def fix_markdown_tables(content):
    """Fix markdown tables to use consistent pipe formatting."""
    lines = content.split("\n")
    in_table = False
    expected_cols = 0
    fixed_lines = []

    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "|" in line[1:]:
            # This appears to be a table row
            if not in_table:
                in_table = True
                # Count columns in header
                expected_cols = line.count("|") - 1
                if not line.strip().endswith("|"):
                    # Add trailing pipe if missing
                    line = line.rstrip() + " |"
            else:
                # Inside table row
                # Make sure row has leading pipe
                if not line.strip().startswith("|"):
                    line = "| " + line.lstrip()

                # Make sure row has trailing pipe
                if not line.strip().endswith("|"):
                    line = line.rstrip() + " |"

                # Check column count
                cols = line.count("|") - 1
                if cols < expected_cols:
                    # Fill missing columns
                    line = line.rstrip() + " | " * (expected_cols - cols)
        elif in_table and line.strip() and not line.strip().startswith("|"):
            # Line is not empty but doesn't start with pipe, probably not in table
            in_table = False

        # Line 56 has length issue - break into multiple lines
        if i == 55 and len(line) > 180:  # 0-indexed, so line 56 is index 55
            # Break the long line into multiple lines
            parts = line.split(". ")
            if len(parts) > 1:
                new_lines = []
                current_line = parts[0] + "."
                for part in parts[1:]:
                    if len(current_line + " " + part) > 100:
                        new_lines.append(current_line)
                        current_line = "  " + part  # Indent continuation
                    else:
                        current_line += " " + part
                new_lines.append(current_line)
                fixed_lines.extend(new_lines)
                continue

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def main():
    input_file = "docs/workflows/workflows-document.md"

    try:
        with open(input_file, encoding="utf-8") as f:
            content = f.read()

        fixed_content = fix_markdown_tables(content)

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        print(f"Successfully fixed markdown tables in {input_file}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
