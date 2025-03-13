
# Type Compatibility Guide {#type-compatibility-guide}

This guide helps address common type compatibility issues when migrating to Python 3.10+.

## Common Type Compatibility Issues {#common-type-compatibility-issues}

### 1. Union Types {#union-types}

Python 3.10 introduced the pipe (`|`) operator as a more concise way to define union types.

#### Before (Python 3.9 and earlier) {#before-python-39-and-earlier}

\`\`\`text\`python
from typing import Union

def process_data(data: Union[dict, list]) -> Union\[str, None\]:

```text`text

...

```text

```text

#### After (Python 3.10+) {#after-python-310}

```python

def process_d
ata(data: dict | list) -> str | None:


```text
..
.

```text

```text

#### Migration Approach {#migration-approach}

For
 com
patibility wi
th both Python 3.9 and Python 3.10+, continue using `Union` from the typing module:

```python

from typing import Union

def process_data(data: Union[
dict, list]) -> Union[str, None]:

```text
...

```text

```text

### 2. Optional Types {#optional-types}

`Opti
onal
[T]` is equiv
alent to `Union[T, None]` or `T | None` in Python 3.10+.

#### Before {#before}

```python

from typing import Optional

def get_user(user_id:
Optional[int] = None) -> Optional[dict]:

```text
...

```text

```text

#### After (Python 3.10+) {#after-python-3

10}

```python

d
ef get_user(user_id: int | None = None) -> dic
t | None:


```text
...

```text

```text

#### Migration Approach {#migration-ap

proa
ch}

For back
ward compatibility, continue using `Optional` from the typing module:

```python

from typing import Optional

def get_user(us
er_id: Optional[int] = None) -> Optional[dict]:

```text
...

```text

```text

### 3. Type Aliases {#type-aliases}

Py
thon 3.10 int
roduces the ability to use `TypeAlias` for explicit type aliases.

#### Before {#before}

```python

from typing import Dict, List, Union

JS
ONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```text

#### After (Python 3.10+) {#after-python-310}

```text
python

from typing import TypeAlias

JSONValu
e: TypeAlias = str | int | float | bool | None | dict[str, 'JSONValue'] | list['JSONValue']


```text

#### Migration Approach {#migration-approach}

F
or backward compatibility:

```python

from typing import Dict, List, Union

## For Python 3.9 and earlier {#for-python-39-and-earlier}

JSONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```text

### 4. Generic Types {#generic-types}

Python
3.9 allows using built-in collection types as generic types, but Python 3.8 requires importing from typing.

#### Before (Python 3.8) {#before-python-38}

```python

from typing import Dict, List

def
 process_data(data: Dict[str, List[int]]) -> None:

```text
...

```text

```text

#### After (Python 3.9+)

{#aft
er-python-39}

```python

def process_data(data: dict[str
, list[int]]) -> None:

```text
...

```text

```text

#### Migration Approac

h {#
migration-app
roach}

For backward compatibility:

```python

from typing import Dict, Li
st

def process_data(data: Dict[str, List[int]]) -> None:

```text
...

```text

```text

## Using the Type

Anno
tation Fix To
ol {#using-the-type-annotation-fix-tool}

We've provided a tool to help identify type annotation issues:

```bash

## Scan the entire proje

ct {#scan-the-entire-project}

python scripts/fix_type_annotations.py .

## Scan a specific file {#scan-a-specific-file}

python scripts/fix_type_annotations.py path/to/file.py

## Run in dry-run mode (don't make changes) {#run-in-dry-run-mode-dont-make-changes}

python scripts/fix_type_annotations.py --dry-run .

## Show detailed information about changes {#show-detailed-information-about-changes}

python scripts/fix_type_annotations.py --verbose .

```text

## Common Runtime Type Errors {#c

ommon-runtime-type-errors}

### 1. None Handling {#none-handling}

When dealing with potentially `None` values, always add explicit checks:

```python

## Problematic {#pro

blematic}

def process_path(path: Optional[str]) -> str:

```text
return os.path.join(path, "subd
ir")  # TypeError if path is None

```text

## Fixed {#fixed}

def pr
ocess_path(path: Optional[str]) -> str:

```text
if path is None:

```text
path =
""

```text
return os
.path.join(path, "
subdir")

```text

```text

### 2. Colle

ction Type Ch
ecking {#collection-type-checking}

When checking collection types:

```python

## Problemat

ic {#problematic}

def process_data(data: Union[dict, list]) -> None:

```text
if isinstance(data, dic
t):

```text

## dict processin

g {#dict-processing}

```text

```text

```text
elif isinsta

nce(dat
a, list):

```text

## list proce

ssing {#list-processing}

```text

```text

```text
else:

`

``
rais
e Valu
eError(f"Expected dict or list, got {type(data)}")

```text

```text

## Bett

er {#better}

def process_data(data: Union[dict, list]) -> None:

```text
if isinstan
ce(data, dict):

```text

## di

ct processing {#dict-processing}

```text

```text

```text
elif is
instance(data, list):

```text
#

# list processing {#list-processing}

```text

```text

```text
els
e:

``
`
acceptable_types = (dict, list)
raise ValueError(f"Expected one of {acceptable_types}, got {type(data)}")

```text

```text

```text

## IDE Support {#ide-support}

Modern IDEs like VS Code with Pylance, PyCharm, or tools like mypy can help identify type annotation issues. Ensure your IDE is configured to check types with Python 3.10+
compatibility.

```text`

```text`
