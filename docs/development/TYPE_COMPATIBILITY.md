
# Type Compatibility Guide

This guide helps address common type compatibility issues when migrating to Python 3.10+.

## Common Type Compatibility Issues

### 1. Union Types

Python 3.10 introduced the pipe (`|`) operator as a more concise way to define union types.

#### Before (Python 3.9 and earlier)

\`\`\`text\`python
from typing import Union

def process_data(data: Union[dict, list]) -> Union\[str, None\]:

```text`text

...

```text

```text

#### After (Python 3.10+)

```python

def process_data(data: dict | list) -> str | None:


```text

...

```text

```text

#### Migration Approach

For compatibility with both Python 3.9 and Python 3.10+, continue using `Union` from the typing module:

```python

from typing import Union

def process_data(data: Union[dict, list]) -> Union[str, None]:

```text

...

```text

```text

### 2. Optional Types

`Optional[T]` is equivalent to `Union[T, None]` or `T | None` in Python 3.10+.

#### Before

```python

from typing import Optional

def get_user(user_id: Optional[int] = None) -> Optional[dict]:

```text

...

```text

```text

#### After (Python 3.10+)

```python

def get_user(user_id: int | None = None) -> dict | None:


```text

...

```text

```text

#### Migration Approach

For backward compatibility, continue using `Optional` from the typing module:

```python

from typing import Optional

def get_user(user_id: Optional[int] = None) -> Optional[dict]:

```text

...

```text

```text

### 3. Type Aliases

Python 3.10 introduces the ability to use `TypeAlias` for explicit type aliases.

#### Before

```python

from typing import Dict, List, Union

JSONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```text

#### After (Python 3.10+)

```python

from typing import TypeAlias

JSONValue: TypeAlias = str | int | float | bool | None | dict[str, 'JSONValue'] | list['JSONValue']


```text

#### Migration Approach

For backward compatibility:

```python

from typing import Dict, List, Union

## For Python 3.9 and earlier

JSONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```text

### 4. Generic Types

Python 3.9 allows using built-in collection types as generic types, but Python 3.8 requires importing from typing.

#### Before (Python 3.8)

```python

from typing import Dict, List

def process_data(data: Dict[str, List[int]]) -> None:

```text

...

```text

```text

#### After (Python 3.9+)

```python

def process_data(data: dict[str, list[int]]) -> None:

```text

...

```text

```text

#### Migration Approach

For backward compatibility:

```python

from typing import Dict, List

def process_data(data: Dict[str, List[int]]) -> None:

```text

...

```text

```text

## Using the Type Annotation Fix Tool

We've provided a tool to help identify type annotation issues:

```bash

## Scan the entire project

python scripts/fix_type_annotations.py .

## Scan a specific file

python scripts/fix_type_annotations.py path/to/file.py

## Run in dry-run mode (don't make changes)

python scripts/fix_type_annotations.py --dry-run .

## Show detailed information about changes

python scripts/fix_type_annotations.py --verbose .

```text

## Common Runtime Type Errors

### 1. None Handling

When dealing with potentially `None` values, always add explicit checks:

```python

## Problematic

def process_path(path: Optional[str]) -> str:

```text

return os.path.join(path, "subdir")  # TypeError if path is None

```text

## Fixed

def process_path(path: Optional[str]) -> str:

```text

if path is None:

```text

path = ""

```text

return os.path.join(path, "subdir")

```text

```text

### 2. Collection Type Checking

When checking collection types:

```python

## Problematic

def process_data(data: Union[dict, list]) -> None:

```text

if isinstance(data, dict):

```text

## dict processing

```text

```text

```text

elif isinstance(data, list):

```text

## list processing

```text

```text

```text

else:

```text

raise ValueError(f"Expected dict or list, got {type(data)}")

```text

```text

## Better

def process_data(data: Union[dict, list]) -> None:

```text

if isinstance(data, dict):

```text

## dict processing

```text

```text

```text

elif isinstance(data, list):

```text

## list processing

```text

```text

```text

else:

```text

acceptable_types = (dict, list)
raise ValueError(f"Expected one of {acceptable_types}, got {type(data)}")

```text

```text

```text

## IDE Support

Modern IDEs like VS Code with Pylance, PyCharm, or tools like mypy can help identify type annotation issues. Ensure your IDE is configured to check types with Python 3.10+
compatibility.

```text`

```text`
