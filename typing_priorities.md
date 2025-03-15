# Type Fixing Prioritization Report

## Most Common Issue Types

| Issue Type | Count | Description |
|------------|-------|-------------|
| missing_return_type | 119 | Function is missing a return type annotation |
| missing_param_type | 80 | Function parameter is missing a type annotation |

## Prioritized Files to Fix

| File | Functions | Type Coverage (%) | Missing Types | LOC | Priority | Description |
|------|-----------|-------------------|---------------|-----|----------|-------------|
| .github/workflows/scripts\comment_on_workflow_pr.py | 32 | 97.8 | 1 | 640 | 50.62 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\simulate_workflow_execution.py | 26 | 36.5 | 18 | 861 | 37.70 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\generate_workflow_versions.py | 12 | 52.5 | 3 | 160 | 30.38 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\hardware_compatibility_advisor.py | 12 | 88.3 | 2 | 712 | 22.94 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\hardware_compatibility_autofix.py | 11 | 87.3 | 2 | 660 | 21.17 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\hardware_compatibility_tester.py | 10 | 86.0 | 2 | 607 | 19.38 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\fix_markdown_all.py | 11 | 63.6 | 2 | 328 | 19.15 | Good typing foundation, needs completion |
| .github/workflows/scripts\validate_components.py | 11 | 93.6 | 1 | 317 | 18.98 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\fix_markdown_issues.py | 16 | 95.6 | 1 | 543 | 18.40 | Nearly complete typing, minor fixes needed |
| .github/workflows/scripts\ci_setup.py | 17 | 30.0 | 21 | 683 | 18.28 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\track_workflow_versions.py | 24 | 56.7 | 6 | 740 | 15.05 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\generate_badges.py | 14 | 30.0 | 17 | 395 | 14.92 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\analyze_workflow_requirements.py | 32 | 30.0 | 89 | 719 | 14.78 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\collect_ci_metrics.py | 12 | 30.0 | 15 | 803 | 14.17 | Moderate typing coverage, many functions need annotations |
| .github/workflows/scripts\generate_workflow_dashboard.py | 19 | 66.8 | 19 | 644 | 12.43 | Good typing foundation, needs completion |