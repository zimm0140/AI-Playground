#!/bin/bash

# Script to fix markdown files in CI
echo "Fixing markdown files..."

# Run the python script to fix common markdown issues
python tools/scripts/fix_markdown_files.py --dir docs/ --exclude node_modules --recursive

# Run markdownlint to verify fixes
echo "Verifying markdown files..."
npx markdownlint-cli2 "docs/**/*.md" --config .markdownlint.yaml

# Check for any remaining errors
if [ $? -ne 0 ]; then
  echo "Markdown linting failed! Please run the fix_markdown_files.py script locally."
  exit 1
else
  echo "All markdown files pass linting! ✅"
  exit 0
fi 