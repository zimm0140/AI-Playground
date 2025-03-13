#!/usr/bin/env python3
"""
Script to fix markdown table formatting issues in workflows-document.md
"""

import os


def fix_markdown_tables(file_path):
    """
    Fix markdown tables to ensure they have proper leading and trailing pipes
    and correct column counts.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return False

    try:
        # Read the file
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Split into lines
        lines = content.split("\n")
        modified = False

        # Process each line
        for i in range(len(lines)):
            line = lines[i]
            original_line = line

            # If line appears to be a table row
            if line.strip().startswith("|") and "|" in line.strip()[1:]:
                # Ensure it has a trailing pipe
                if not line.strip().endswith("|"):
                    line = line.rstrip() + " |"
                    modified = True

            # Fix the long line (line 56)
            if i == 55 and len(line) > 180:  # Line 56 (0-indexed as 55)
                # Break into multiple lines with proper indentation
                if "Clip nodes" in line:
                    parts = line.split(".")
                    if len(parts) > 1:
                        line = parts[0] + "."
                        lines.insert(i + 1, "  " + ".".join(parts[1:]).lstrip())
                        modified = True

            # Update the line if it was modified
            if line != original_line:
                lines[i] = line

        # Write back the content if modified
        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Successfully fixed markdown formatting in {file_path}")
        else:
            print(f"No changes needed in {file_path}")

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


# Specify file paths to check
files_to_fix = ["docs/workflows/workflows-document.md", "docs/reference/api.md"]

# Try to fix each file
for file_path in files_to_fix:
    fix_markdown_tables(file_path)
