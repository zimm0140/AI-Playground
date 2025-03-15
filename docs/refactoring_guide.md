# Refactoring Guide: Reducing Code Complexity

This guide documents the refactoring patterns and techniques we've established during our technical debt reduction process.

## Table of Contents

1. [Introduction](#introduction)

2. [Identifying Complex Functions](#identifying-complex-functions)

3. [Refactoring Patterns](#refactoring-patterns)

   - [Extract Method](#extract-method)

   - [Single Responsibility Principle](#single-responsibility-principle)

   - [Early Returns](#early-returns)

   - [Simplify Conditional Logic](#simplify-conditional-logic)

   - [State Management](#state-management)

4. [Before and After Examples](#before-and-after-examples)

5. [Testing Refactored Code](#testing-refactored-code)

6. [Best Practices](#best-practices)

## Introduction

High cyclomatic complexity is a measure of the number of linearly independent paths through a program's source code. Functions with high complexity are harder to understand, test, and maintain. This guide outlines strategies for reducing complexity based on our technical debt reduction project.

## Identifying Complex Functions

We use `ruff` with the `C901` rule to identify complex functions:

```bash

# Find all complex functions in the codebase

python -m ruff check --select C901 .

# Check a specific file

python -m ruff check --select C901 path/to/file.py
```

## Refactoring Patterns

### Extract Method

The most common and effective refactoring pattern is to extract related blocks of code into separate methods.

**When to use:**

- A function has distinct sections that perform different tasks
- A block of code can be isolated and has a clear purpose
- Code reuse is possible

**Example:**

```python

# Before: Single complex function

def process_data(data):
    # Data validation (20 lines)

    # Data transformation (30 lines)

    # Result calculation (25 lines)

    # Output formatting (15 lines)

    return result

# After: Main function orchestrates smaller helper functions

def process_data(data):
    validated_data = _validate_data(data)
    transformed_data = _transform_data(validated_data)
    results = _calculate_results(transformed_data)
    return _format_output(results)

def _validate_data(data):
    # 20 lines of validation logic

    return validated_data

def _transform_data(data):
    # 30 lines of transformation logic

    return transformed_data

def _calculate_results(data):
    # 25 lines of calculation logic

    return results

def _format_output(results):
    # 15 lines of formatting logic

    return formatted_results
```

### Single Responsibility Principle

Each function should have a single responsibility - one reason to change.

**When to use:**

- A function is doing multiple unrelated things
- A function has "and" in its name or description
- Changes to one part of the function might affect other parts

**Example:**

```python

# Before: Function with multiple responsibilities

def validate_and_process_user(user_data):
    # Validation logic

    # Processing logic

    # Database update logic

    return result

# After: Separate functions for each responsibility

def validate_user(user_data):
    # Validation logic

    return is_valid, validation_errors

def process_user(validated_user_data):
    # Processing logic

    return processed_data

def update_user_database(processed_data):
    # Database update logic

    return success
```

### Early Returns

Using early returns can reduce nesting levels and make code easier to follow.

**When to use:**

- Deep nesting of conditionals
- Guard clauses can be used to handle edge cases

**Example:**

```python

# Before: Nested conditionals

def process_request(request):
    if request is not None:
        if request.has_data():
            data = request.get_data()
            if validate_data(data):
                # Process the data

                return result
            else:
                return error("Invalid data")
        else:
            return error("No data in request")
    else:
        return error("Request is None")

# After: Early returns

def process_request(request):
    if request is None:
        return error("Request is None")
    
    if not request.has_data():
        return error("No data in request")
    
    data = request.get_data()
    if not validate_data(data):
        return error("Invalid data")
    
    # Process the data

    return result
```

### Simplify Conditional Logic

Complex conditional logic can be simplified through various techniques.

**When to use:**

- Multiple nested conditions
- Complex boolean expressions
- Duplicated conditions in different branches

**Techniques:**

- Replace nested conditions with guard clauses
- Extract complex conditions into well-named functions
- Use dictionary lookups instead of lengthy if-elif chains
- Apply the Strategy pattern for varying behaviors

**Example:**

```python

# Before: Complex conditional logic

def calculate_discount(customer, order, season):
    discount = 0
    if customer.is_premium():
        if order.total > 1000:
            if season == "summer":
                discount = 0.15
            elif season == "winter":
                discount = 0.10
            else:
                discount = 0.12
        else:
            if season == "summer":
                discount = 0.10
            elif season == "winter":
                discount = 0.08
            else:
                discount = 0.09
    else:
        if order.total > 1000:
            discount = 0.07
        else:
            discount = 0.05
    return discount

# After: Simplified with lookup tables

def calculate_discount(customer, order, season):
    # Define discount lookup tables

    premium_discounts = {
        "high_value": {"summer": 0.15, "winter": 0.10, "default": 0.12},
        "regular": {"summer": 0.10, "winter": 0.08, "default": 0.09}
    }
    standard_discounts = {"high_value": 0.07, "regular": 0.05}
    
    # Determine customer and order categories

    customer_type = "premium" if customer.is_premium() else "standard"
    order_category = "high_value" if order.total > 1000 else "regular"
    
    # Get discount based on categories

    if customer_type == "premium":
        season_key = season if season in ["summer", "winter"] else "default"
        return premium_discounts[order_category][season_key]
    else:
        return standard_discounts[order_category]
```

### State Management

Explicitly manage state to reduce complexity when dealing with state changes.

**When to use:**

- Functions that track and modify state
- Code with multiple flags or state variables

**Example:**

```python

# Before: Implicit state management

def process_document(doc):
    in_header = True
    in_table = False
    table_header_seen = False
    result = []
    
    for line in doc.split('\n'):
        if line.startswith('#'):

            if in_table:
                in_table = False
                table_header_seen = False
            in_header = True
            result.append(format_header(line))
        elif line.startswith('|') and line.endswith('|'):
            in_header = False
            if not in_table:
                in_table = True
                if not table_header_seen:
                    table_header_seen = True
                    result.append(format_table_header(line))
                else:
                    result.append(format_table_row(line))
            else:
                result.append(format_table_row(line))
        else:
            if in_table:
                in_table = False
                table_header_seen = False
            in_header = False
            result.append(format_paragraph(line))
    
    return '\n'.join(result)

# After: State object with clear transitions

class DocumentState:
    def __init__(self):
        self.in_header = False
        self.in_table = False
        self.table_header_seen = False
    
    def process_header(self):
        self.in_header = True
        self.in_table = False
        self.table_header_seen = False
    
    def process_table_line(self):
        self.in_header = False
        # Handle table state transitions

        if not self.in_table:
            self.in_table = True
            return not self.table_header_seen  # Is this a new table header?

        return False
    
    def process_paragraph(self):
        self.in_header = False
        self.in_table = False
        self.table_header_seen = False

def process_document(doc):
    state = DocumentState()
    result = []
    
    for line in doc.split('\n'):
        if line.startswith('#'):

            state.process_header()
            result.append(format_header(line))
        elif line.startswith('|') and line.endswith('|'):
            is_table_header = state.process_table_line()
            if is_table_header:
                state.table_header_seen = True
                result.append(format_table_header(line))
            else:
                result.append(format_table_row(line))
        else:
            state.process_paragraph()
            result.append(format_paragraph(line))
    
    return '\n'.join(result)
```

## Before and After Examples

### Example 1: Generate Workflow Documentation

**Before:** A single 150+ line function that handled all aspects of documentation generation.

**After:** Small helper functions for each section of the documentation:

```python
def generate_workflow_doc(workflow, workflow_file):
    """Generate markdown documentation for a workflow."""
    sections = []
    
    # Add each section of the documentation

    sections.extend(_generate_header_section(workflow))
    sections.extend(_generate_description_section(workflow))
    sections.extend(_generate_examples_section(workflow))
    sections.extend(_generate_resource_requirements_section(workflow))
    sections.extend(_generate_components_section(workflow))
    sections.extend(_generate_system_requirements_section(workflow))
    sections.extend(_generate_technical_requirements_section(workflow))
    sections.extend(_generate_default_settings_section(workflow))
    sections.extend(_generate_inputs_section(workflow))
    sections.extend(_generate_outputs_section(workflow))
    sections.extend(_generate_change_log_section(workflow))
    sections.extend(_generate_footer_section())
    
    return "\n".join(sections)
```

### Example 2: Patch Files for CI

**Before:** A single function with multiple responsibilities and deep nesting.

**After:** Small helper functions for each patching operation:

```python
def patch_files():
    """Apply patches to make code work in CI environment"""
    print("Applying CI compatibility patches...")

    # Fix invalid escape sequences in paint_biz.py

    _patch_paint_biz()
    
    # Patch web_api.py to handle imports safely

    _patch_web_api()
    
    # Aggressively fix indentation in test_api.py

    _patch_test_api()
    
    # Fix xpu_hijacks.py for more resilient ipex usage

    _patch_xpu_hijacks()
    
    # Create dummy test_api.py if all else fails

    _ensure_valid_test_api()

    print("CI compatibility patches applied")
```

## Testing Refactored Code

When refactoring complex functions:

1. Ensure you maintain the same behavior
2. Write tests before refactoring when possible
3. Run existing tests after refactoring
4. Run linters to verify complexity reduction:
   ```bash
   python -m ruff check --select C901 path/to/file.py
   ```

## Best Practices

1. **Incremental Refactoring**: Refactor in small, testable steps
2. **Clear Naming**: Use descriptive names for extracted methods
3. **Consistent Style**: Follow existing code conventions
4. **Documentation**: Update docstrings to reflect new structure
5. **Testing**: Ensure tests pass after refactoring
6. **Review**: Have others review your refactored code
7. **Commit Messages**: Clearly describe refactoring in commit messages

Remember, the goal of refactoring is not just to reduce complexity metrics, but to improve code readability, maintainability, and testability.