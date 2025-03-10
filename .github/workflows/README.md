# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the AI-Playground project.

## Common Issues and Solutions

### NPM Directory ENOENT Error

If you encounter `ENOENT: no such file or directory, lstat 'C:\Users\...\AppData\Roaming\npm'` errors in CI runs,
use the `setup-npm-dir.js` script in the `.github/actions` directory to ensure the npm directory exists before
running any npm/npx commands.

Example usage in a workflow:

```yaml
steps:
  - name: Checkout code
    uses: actions/checkout@v3

  - name: Set up Node.js
    uses: actions/setup-node@v3
    with:
      node-version: '18'

  - name: Ensure npm directory exists
    run: node .github/actions/setup-npm-dir.js

  - name: Run Prettier
    run: cd WebUI && npx prettier --check external/components/flux_sampler.json external/components/README.md external/components/text_encoder_t5_clip.json external/workflows/Colorize.json
```

### Python Linting Issues

When adding or modifying Python code, ensure:

1. All imports are at the top of the file (to avoid E402 errors)
1. Run linting checks locally before pushing:

```bash
pip install ruff
ruff check ./service
```
