
# Type Compatibility Guide {#type-compatibility-guide}

This guide helps address common type compatibility issues when migrating to Python 3.10+.

## Common Type Compatibility Issues {#common-type-compatibility-issues}

### 1. Union Types {#union-types}

Python 3.10 introduced the pipe (`|`) operator as a more concise way to define union types.

#### Before (Python 3.9 and earlier) {#before-python-39-and-earlier}

\`\`\`text\`python
from typing import Union

def process_data(data: Union[dict, list]) -> Union\[str, None\]:

````

...

```

```

#### After (Python 3.10+) {#after-python-310}

```python

def process_d

ata(data: dict | list) -> str | None:

```

.
.
.

```

```

#### Migration Approach {#migration-approach}

Fo
r
 co
m
patibility
wi
th both Python 3.9 and Python 3.10+, continue using `Union` from the typing module:

```python

from typing import Union

def process_data(data: Union[

dict, list]) -> Union[str, None]:

```

...

```

```

### 2. Optional Types {#optional-types}

`Opti

ona
l
[T]` is equ
iv
alent to `Union[T, None]` or `T | None` in Python 3.10+.

#### Before {#before}

```python

from typing import Optional

def get_user(user_id:

Optional[int] = None) -> Optional[dict]:

```

...

```

```

#### After (Python 3.10+) {#after-python-3

 {#a

fter-python-3
10-after-python-3}

10}

```python

d
ef get_user(user_id: int | None = None) -> di
c
t | None:

```

...

```

```

#### Migration Approach {#migration-ap

 {#m

igration-appr
oach-migration-ap}

proa
ch}

For back
ward compatibility, continue using `Optional` from the typing module:

```python

from typing import Optional

def get_user(u
s
er_id: Optional[int] = None) -> Optional[dict]:

```

...

```

```

### 3. Type Aliases {#type-aliases

}

P
y
thon 3.10 i
nt
roduces the ability to use `TypeAlias` for explicit type aliases.

#### Before {#before}

```python

from typing import Dict, List, Union

J
S
ONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```

#### After (Python 3.10+) {#after-python-310}

``
`
python

from typing import TypeAlias

JSONVa
lu
e: TypeAlias = str | int | float | bool | None | dict[str, 'JSONValue'] | list['JSONValue']

```

#### Migration Approach {#migration-approach}

F
or backward compatibility:

```python

from typing import Dict, List, Unio
n

## For Python 3.9 and earlier {#for-python-39-and-earlier}

JSONValue = Union[str, int, float, bool, None, Dict[str, 'JSONValue'], List['JSONValue']]

```

### 4. Generic Types {#generic-types}

Python

3.9 allows using built-in collection types as generic types, but Python 3.8 requires importing from typing.

#### Before (Python 3.8) {#before-python-38}

```python

from typing import Dict, List

de
f
 process_data(data: Dict[str, List[int]]) -> None:

```

...

```

```

#### After (Python 3.9+)

 {#a

fter-python-3
9}

{#aft

er-python-39}

```python

def process_data(data: dict[s
tr
, list[int]]) -> None:

```

...

```

```

#### Migration Appro

ac {
#migration-ap

proac}

h {#
migration-app
roach}

For backward compatibility:

```python

from typing import Dict,
Li
st

def process_data(data: Dict[str, List[int]]) -> None:

```

...

```

```

## Using the Typ

e {#
using-the-typ
e}

Anno
tation Fix To
ol {#using-the-type-annotation-fix-tool}

We've provided a tool to help identify type annotation issues:

```bash

## Scan the entire pro

je {#scan-the-entire-proje}

ct {#scan-the-entire-project}

python scripts/fix_type_annotations.py .

## Scan a specific file {#scan-a-specific-file}

python scripts/fix_type_annotations.py path/to/file.py

## Run in dry-run mode (don't make changes) {#run-in-dry-run-mode-dont-make-changes}

python scripts/fix_type_annotations.py --dry-run .

## Show detailed information about changes {#show-detailed-information-about-changes}

python scripts/fix_type_annotations.py --verbose .

```

## Common Runtime Type Errors {

#c {#common-runtime-type-errors-c}

ommon-runtime-type-errors}

### 1. None Handling {#none-handling}

When dealing with potentially `None` values, always add explicit checks:

```python

## Problematic {#p

ro {#problematic-pro}

blematic}

def process_path(path: Optional[str]) -> str:

```

return os.path.join(path, "su
bd
ir")  # TypeError if path is None

```

## Fixed {#fixed}

def
pr
ocess_path(path: Optional[str]) -> str:

```

if path is None:

```

path
=
""

```

return
os
.path.join(path
, "
subdir")

```

```

### 2. Col

le {#colle}

ction Type Ch
ecking {#collection-type-checking}

When checking collection types:

```python

## Problem

at {#problemat}

ic {#problematic}

def process_data(data: Union[dict, list]) -> None:

```

if isinstance(data, d
ic
t):

```

## dict process

in {#dict-processin}

g {#dict-processing}

```

```

```

elif isins

ta

nce
(dat
a, list):

```

## list pro

ce {#list-proce}

ssing {#list-processing}

```

```

```

else:

`

``

rais
e Valu
eError(f"Expected dict or list, got {type(data)}")

```

```

## Bett {#b

ett}

er {#better}

def process_data(data: Union[dict, list]) -> None:

```

if isi
nstan
ce(data, dict):

```

## di {#d

i}

ct processing {#dict-processing}

```

```

```

el

if is
instance(data, list):

```

#

#
list processing {#list-processing}

```t
ext

```

``

`
els
e:

``
`
acceptable_types = (dict, list)
raise ValueError(f"Expected one of {acceptable_types}, got {type(data)}")

```

```

```

## IDE Support {#ide-support}

Modern IDEs like VS Code with Pylance, PyCharm, or tools like mypy can help identify type annotation issues. Ensure your IDE is configured to check types with Python 3.10+
compatibility.

````

````
