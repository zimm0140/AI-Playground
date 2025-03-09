---
name: Ruff Linting Error
about: Report a linting error that needs resolution
title: 'Ruff Linting Error: [brief description]'
labels: 'linting, bug'
assignees: ''
---

## Ruff Error Description

**Error code(s):** <!-- Example: F401, E302, etc. -->

**Files affected:**
<!-- List file paths where the error occurs -->

## Error Details

<!-- Paste the exact error message if available -->
```
Paste error message here
```

## Steps to Reproduce

1. Run Ruff on the affected files: `ruff check ./service --select=E,F --ignore=E501`
2. See error in output

## Proposed Solution

<!-- If you have an idea how to fix it, describe it here -->

## Screenshots

<!-- If applicable, add screenshots to help explain the problem -->

## Additional Context

<!-- Add any other context about the problem here -->

## Checklist before submitting

- [ ] I've run the local fix script first: `.github/workflows/scripts/fix_ruff_issues_local.py`
- [ ] I've checked for similar issues
- [ ] I've included all relevant details 