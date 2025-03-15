# Docstring Standard

This document defines our project's standard for docstrings to ensure consistency across the codebase.

## Python Docstring Format

We use the [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) format for all Python code.

### Module Docstrings

Every module should have a docstring at the top describing its purpose and usage:

```python
"""
Module description.

This module provides functionality for X.
Longer description and usage examples can go here.
"""
```

### Class Docstrings

Classes should have docstrings describing their purpose, attributes, and usage:

```python
class MyClass:
    """Short description of the class.
    
    Longer description explaining the class purpose and behavior.
    
    Attributes:
        attr1 (type): Description of attr1.
        attr2 (type): Description of attr2.
    """
```

### Function/Method Docstrings

Functions and methods should use the following format:

```python
def my_function(param1, param2, param3=None):
    """Short description of function purpose.
    
    Longer description that explains what the function does,
    special cases, and any other important information.
    
    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.
        param3 (type, optional): Description of param3. Defaults to None.
    
    Returns:
        type: Description of return value.
        
    Raises:
        ExceptionType: Description of when this exception is raised.
    """
```

### Properties

Properties should be documented using the same format as methods:

```python
@property
def my_property(self):
    """Short description of the property.
    
    Returns:
        type: Description of return value.
    """
```

## Type Annotations

In addition to docstring type descriptions, we use Python type annotations:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Example function with type annotations.
    
    Args:
        param1: A string parameter.
        param2: An integer parameter. Defaults to 0.
        
    Returns:
        Success status of the operation.
    """
```

Note that when using type annotations, you don't need to repeat the type in the docstring Args section.

## Implementation Guidelines

1. **Be Consistent:** Follow the same format for all docstrings.
2. **Be Descriptive:** Provide clear descriptions that explain "why" not just "what."
3. **Include Examples:** When appropriate, add examples of how to use complex functions.
4. **Update When Changing Code:** Always update the docstring when changing function parameters or behavior.

## Tools

We use the following tools to enforce our docstring standards:

- **Ruff** with the `D` prefix rules to check docstring formats
- **mypy** to verify type annotations

## Example

```python
def calculate_discount(customer_id: str, order_total: float, apply_special_offer: bool = False) -> float:
    """Calculate the discount for a customer order.
    
    Determines the appropriate discount based on customer loyalty status,
    order total amount, and any applicable special offers.
    
    Args:
        customer_id: Unique identifier for the customer.
        order_total: Total amount of the order before discount.
        apply_special_offer: Whether to apply seasonal special offers. Defaults to False.
        
    Returns:
        The calculated discount amount.
        
    Raises:
        ValueError: If order_total is negative.
        
    Example:
        >>> calculate_discount("CUST123", 150.0, True)
        15.0
    """
```