// This script would use prettier to format the README.md file
// However, since we can't run npm commands directly, we'll manually apply
// the most common Prettier formatting rules for Markdown files

const fs = require('fs');
const path = require('path');

// Path to the README file
const readmePath = path.join('WebUI', 'external', 'workflows', 'README.md');

// Read the file content
const content = fs.readFileSync(readmePath, 'utf8');

// Fix common issues:
// 1. Remove trailing whitespace on each line
// 2. Ensure single newline at EOF
// 3. Consistent line endings (LF)
let fixed = content
  .split(/\r?\n/)
  .map(line => line.trimRight())
  .join('\n');

// Ensure exactly one newline at the end of file (no trailing whitespace)
fixed = fixed.trimRight() + '\n';

// Write the fixed content back to the file
fs.writeFileSync(readmePath, fixed, 'utf8');

console.log(`Fixed ${readmePath}`); 