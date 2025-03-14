# Complex Functions Refactoring Report

This report identifies functions with high cyclomatic complexity.
These functions are primary candidates for refactoring to improve maintainability.

## Priority Refactoring List

### 1.  (Complexity: 50)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\generate_workflow_dashboard.py`
- Line: 345
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 2.  (Complexity: 41)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\comment_on_workflow_pr.py`
- Line: 304
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 3.  (Complexity: 40)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\generate_workflow_docs.py`
- Line: 46
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 4.  (Complexity: 37)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\analyze_workflow_requirements.py`
- Line: 132
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 5.  (Complexity: 34)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\fix_ci_issues.py`
- Line: 14
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 6.  (Complexity: 34)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\validate_components.py`
- Line: 19
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 7.  (Complexity: 27)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\ensure_unique_artifacts.py`
- Line: 15
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 8.  (Complexity: 26)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\generate_workflow_dashboard.py`
- Line: 151
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 9.  (Complexity: 21)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\check_requirements_consistency.py`
- Line: 69
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 10.  (Complexity: 21)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\generate_api_summary.py`
- Line: 15
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 11.  (Complexity: 21)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\test_workflow_execution.py`
- Line: 243
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 12.  (Complexity: 21)
- File: `C:\Code\ML\AI-Playground\service\aipg_utils.py`
- Line: 303
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 13.  (Complexity: 20)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\analyze_workflow_requirements.py`
- Line: 435
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 14.  (Complexity: 20)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\generate_workflow_dashboard.py`
- Line: 668
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 15.  (Complexity: 19)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\validate_comfyui_workflows.py`
- Line: 104
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 16.  (Complexity: 18)
- File: `C:\Code\ML\AI-Playground\hardware_detection\core.py`
- Line: 157
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 17.  (Complexity: 17)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\fix_markdown_all.py`
- Line: 72
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 18.  (Complexity: 17)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\hardware_compatibility_advisor.py`
- Line: 117
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 19.  (Complexity: 17)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\hardware_compatibility_tester.py`
- Line: 278
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

### 20.  (Complexity: 17)
- File: `C:\Code\ML\AI-Playground\.github\workflows\scripts\simulate_workflow_execution.py`
- Line: 449
- Refactoring suggestions:
  - Break into smaller functions
  - Simplify conditional logic
  - Use helper functions for repeated code

## Next Steps
1. Start with the top 5 most complex functions
2. Create unit tests before refactoring
3. Refactor one function at a time
4. Re-run complexity analysis after each refactoring