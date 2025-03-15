# Type Safety Guide

This guide outlines the best practices and standards for maintaining type safety in the AI Playground codebase. Proper type annotations make code more readable, self-documenting, and catch errors at development time rather than runtime.

## Why Type Annotations?

Type annotations provide several benefits:

1. **Catch errors early**: Type checkers like mypy find type errors before your code runs
2. **Improved IDE support**: Better code completion, refactoring, and navigation
3. **Self-documenting code**: Clear indication of expected inputs and outputs
4. **Simplified refactoring**: More confidence when changing code
5. **Enhanced readability**: Easier to understand code's intent

## Basic Type Annotations

### Function Annotations

All functions should have parameter and return type annotations:

```python
def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
```

### Variable Annotations

Variables should be annotated, especially at module level or in class definitions:

```python
from typing import Dict, List

# Module-level variables
ALLOWED_EXTENSIONS: List[str] = ["jpg", "png", "gif"]
CONFIG: Dict[str, str] = {"api_key": "default_key", "timeout": "30"}

# Within functions - type inference often makes this unnecessary
def process_data() -> Dict[str, int]:
    results: Dict[str, int] = {}  # Explicit annotation
    count = 0  # Type inferred as int, no annotation needed
    # ...
    return results
```

## Common Typing Patterns

### Collections

Use the appropriate collection type from the `typing` module:

```python
from typing import Dict, List, Set, Tuple

# List of strings
names: List[str] = ["Alice", "Bob", "Charlie"]

# Dictionary mapping strings to integers
scores: Dict[str, int] = {"Alice": 95, "Bob": 87, "Charlie": 92}

# Set of integers
unique_ids: Set[int] = {1001, 1002, 1003}

# Tuple with specific types for each position
point: Tuple[float, float] = (23.5, 42.1)

# Tuple with variable length but same type
coordinates: Tuple[float, ...] = (23.5, 42.1, 5.7)
```

### Optional Values

Use `Optional` for values that might be `None`:

```python
from typing import Optional

def find_user(user_id: int) -> Optional[User]:
    """Find a user by ID, returns None if not found."""
    # ...
    if user_exists:
        return user
    return None
```

### Union Types

Use `Union` for values that could be one of several types:

```python
from typing import Union

def process_identifier(identifier: Union[int, str]) -> None:
    """Process an identifier which could be an integer or string."""
    if isinstance(identifier, int):
        # Handle integer case
        pass
    else:
        # Handle string case
        pass
```

### Type Aliases

Create type aliases for complex or commonly used types:

```python
from typing import Dict, List, NewType, Tuple, TypeAlias

# Simple type alias
UserId = int

# More complex alias
UserRecord: TypeAlias = Dict[str, Union[str, int, bool]]

# Type for enhanced type safety
AuthToken = NewType('AuthToken', str)

# Usage
def get_user(user_id: UserId) -> UserRecord:
    # ...
    pass

def authenticate(token: AuthToken) -> bool:
    # This function will only accept AuthToken, not any string
    # ...
    pass
```

## Advanced Type Annotations

### Callable Types

For functions that accept other functions as arguments:

```python
from typing import Callable

def apply_function(func: Callable[[int], str], value: int) -> str:
    """Apply a function that converts an int to a string."""
    return func(value)
```

### Type Variables and Generics

For functions that work with any type while preserving type information:

```python
from typing import Generic, List, TypeVar

T = TypeVar('T')  # Define a type variable

def first_element(items: List[T]) -> T:
    """Return the first element of a list with preserved type."""
    return items[0]

# The return type will match the input list's element type
name: str = first_element(["Alice", "Bob"])  # Type is str
number: int = first_element([1, 2, 3])  # Type is int
```

### Protocol Classes

For structural typing (duck typing with type checks):

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...

def render(item: Drawable) -> None:
    """Render any object that has a draw method."""
    item.draw()

# Any class with a draw method will satisfy the Drawable protocol
class Circle:
    def draw(self) -> None:
        print("Drawing a circle")

render(Circle())  # This works, even though Circle doesn't inherit from Drawable
```

## Working with Third-Party Code

### Type Stubs

For libraries without type annotations, use stub files or install type packages:

```bash
# Install type stubs for popular libraries
pip install types-requests
```

### Type Ignores

Use type ignores sparingly and with comments explaining why:

```python
# Third-party library returns dynamic type
result = external_lib.complex_function()  # type: ignore  # Returns dynamic JSON structure
```

## Type Checking

### Running mypy

We use mypy for type checking. Run it locally before committing:

```bash
# Check a specific file
python -m mypy --config-file mypy.ini path/to/file.py

# Check using our gradual adoption tool
python -m tools.run_type_checks --targets path/to/file.py
```

### Configuration

Our mypy configuration is in `mypy.ini`. Key settings include:

- `disallow_untyped_defs`: Disallows functions without type annotations
- `disallow_incomplete_defs`: Disallows functions with some parameters not annotated
- `check_untyped_defs`: Type checks the body of functions without annotations
- `disallow_any_generics`: Disallows usage of Any in generic types (e.g., List[Any])

## Fixing Common Type Issues

### Missing Return Types

Always add return types to functions, including those that return None:

```python
# Bad
def update_user(user_id, data):
    db.update(user_id, data)
    
# Good
def update_user(user_id: int, data: Dict[str, Any]) -> None:
    db.update(user_id, data)
```

### Optional Handling

Always check if Optional values are None before using them:

```python
# Bad
def process_data(data: Optional[Dict[str, Any]]) -> str:
    return data["name"]  # Might be None!

# Good
def process_data(data: Optional[Dict[str, Any]]) -> str:
    if data is None:
        return "No data"
    return data["name"]
```

### Any Usage

Avoid using `Any` unless absolutely necessary:

```python
# Bad
def process_data(data: Any) -> Any:
    return data.transform()

# Better
from typing import TypeVar, Protocol

T = TypeVar('T')

class Transformable(Protocol):
    def transform(self) -> T:
        ...

def process_data(data: Transformable[T]) -> T:
    return data.transform()
```

## Tools

We've created several tools to help with type safety:

1. `tools/fix_typing_issues.py`: Automatically fixes common typing issues
   ```bash
   python -m tools.fix_typing_issues --path path/to/file.py
   ```

2. `tools/gradual_type_adoption.py`: Analyzes files for typing readiness
   ```bash
   python -m tools.gradual_type_adoption --directory path/to/dir --output report.md
   ```

3. `tools/run_type_checks.py`: Runs standardized type checks on specified files
   ```bash
   python -m tools.run_type_checks --targets file1.py file2.py
   ```

## Best Practices

1. **Start with correct types from the beginning** - It's easier than adding them later
2. **Use specific types rather than Any** - Be as specific as possible
3. **Create custom types for domain concepts** - Improves code readability
4. **Use Literal for constrained string/int values** - Prevents invalid values
5. **Document complex type decisions** - Add comments explaining non-obvious type choices
6. **Keep type aliases in a common module** - For reuse across the codebase
7. **Test type correctness** - Include type-incorrect examples in tests

## References

- [Python Type Checking Guide](https://realpython.com/python-type-checking/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [PEP 585 – Type Hinting Generics In Standard Collections](https://peps.python.org/pep-0585/)
- [PEP 593 – Flexible function and variable annotations](https://peps.python.org/pep-0593/)