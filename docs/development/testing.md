
# Testing Guide {#testing-guide}

This guide explains how to run and write tests for AI-Playground, helping developers ensure their changes maintain code quality and functionality.

## Testing Philosophy {#testing-philosophy}

Tests in AI-Playground should:

1. *_Verify behavior, not implementation__: Focus on testing what the code does, not how it does it

1. **Be independent**: Tests should not depend on other tests

1. **Be deterministic**: Tests should pass or fail consistently

1. **Be fast**: Tests should run quickly to encourage frequent testing

1. **Cover edge cases**: Test normal operation and exceptional conditions

## Test Structure {#test-structure}

The tests are organized as follows:

- `tests/conftest.py`: Contains pytest fixtures shared across tests

- `tests/test_hardware_detection.py`: Tests for the hardware detection module

- `tests/test_uvfast.py`: Tests for the main uvfast.py module

- `tests/test_setup_hardware_env.py`: Tests for the setup_hardware_env.py module

- `tests/test_coverage.py`: Tests to assess and monitor test coverage

- `tests/utils/test_workflow_parser.py`: Tests for the workflow parser utilities

## Running Tests {#running-tests}

### Running All Tests {#running-all-tests}

To run all tests:

\`\`\`text\`bash
pytest

```text`text

### Running Tests with Detailed Output {#running-tests-with-detailed-output}

To run tests with detailed output:

```bash

pytest -v

```text

### Running a Specific Test File {#ru

nning-a-specific-test-fi
le}

To run tests from a specific file:

```bash

pytest tests/test_hardware_detection.py

```text

### R

unning a Specific Test Function {#running-a-specific-t
est-function}

To run a specific test function:

```bash

pytest tests/test_hardware_detection.py::TestHard
wareDetection::test_detect_arc_gpu

```text

### Running Tests with Coverage Report {#running-tests-wi

th-coverage-report}

To run tests with a coverage report:

```bash

pytest --cov=. tests/

```text
For a more detailed
coverage report:

```bash

pytest -
-cov=. --cov-report=html tests/

```text
This will generate an HTML coverage report in the `ht
mlcov` directory.

## Writing Tests {#writing-tests}

### Test File Naming {#test-file-naming}

- Test files should be named `test__.py`

- Test functions should be named `test__`

- Test classes should be named `Test_`

### Test Function Structure {#test-function-structure}

A good test function should:

1. Set up the test environment (Arrange)

1. Execute the code being tested (Act)

1. Verify the expected outcomes (Assert)

Example:

```python

def test_detect_arc_gpu():

```text

## Arran

ge {#arrange}

```text

```text
mock_gpu_info =
 ["Intel(R) Arc(TM) A77
0 Graphics"]

```text

```text

## Act {#act}

```text

```text
with pa
tch("hardware
_detection.get
_gpu_info",
return_value=mock_gpu_info):

```text
result = hardware_detection.detect_hardwa
re_type()

```text

```text

```text

## Assert {#assert}

```text

```text
assert r
esult == "acm"

```t
ext

```text
##

# Using Fixtures {#usin

g-fixtures}

Fixtures are a powerful way to reuse test setup code:

```python

@pytest.fixture
def sample
_config():

```text
"""Fixture providing a sample config
uration."""
return {

```text
"hardware_types": ["base", "acm
"],
"default_hardware": "base",

```text
}

```text
def test_hardware_detection
(s
ample_config):

```text

## Use the sample_config fixture

 {#use-the-sample_config-fixture}

```text

```text
with patch("hardwa
re_detection
.load_config", return_value=sample_config):

```text

## Test code here {#test

-code-here}

```text

```text

```text

```text
pass

```te

xt

```text

```text

### Mocking {#m

ocking}

Use mocking to isolate the code being tested:

```python

@patch("har
dware_detection.get_gpu_info")
def test_gpu_detection(mock_get_gpu_info):

```text

## Configure the mock

 {#configure-the-mock}

```text

```text
mock_ge
t_gpu_info.r
eturn_value = ["Intel(R) Arc(TM) A770 Graphics"]

```text

```text

## T

est code that
 uses get_gpu_info {#test-code-that-uses-get_gpu_info}

```text

```text
res
ult = hardwa
re_detection.detect_hardware_type()

```text

```text

## Verify res

ults {#verify-results}

```text

```text
assert resu
lt == "acm"

```text

`
``

### Testi

ng Hardware-Specific Code {#testing-hardware-specific-code}

When testing hardware-specific functionality:

1. *_Mock hardware detection__: Don't rely on actual hardware for tests

1. **Test all hardware paths**: Ensure each hardware configuration works

1. **Test fallback behavior**: Verify code works with unsupported hardware

1. **Parameterize tests**: Use parameterized tests for different hardware

Example:

```pytho
n

@pytest.mark.parametrize("gpu_info,expected_type", [

```text
(["Inte
l(R) Arc(TM) A770 Graphics"], "acm"),
(["Intel(R) Battlemage(TM) B770 Graphics"], "bmg"),
(["NVIDIA GeForce RTX 3080"], "base"),

```text
])

def test_multiple_hardware_types(gpu_info, expected_type):

```text
with
patch("hardware_detection.get_gpu_info", return_value=gpu_info):

```text
result = hardware_detection.detect_hardware_type()
assert result == expected_type

```text

``

`

```text
#

# Testing Best Practices {#testing-best-practices}

1. **Test the public API**: Focus on testing public interfaces, not implementation details

1. **Keep tests simple**: Tests should be easy to understand

1. **Test one thing per test**: Each test should verify a single behavior

1. **Use meaningful test names**: Names should indicate what's being tested

1. **Don't test external dependencies**: Mock external dependencies

1. **Clean up after tests**: Tests should clean up any resources they create

1. **Don't modify production code for testing**: Use mocks and dependency injection instead

## Test Coverage Assessment {#test-coverage-assessment}

The `test_coverage.py` file provides utilities to assess test coverage without additional tools:

```bash

pytest tests/test_coverage.py -v

```text
This will show which core modules have tests and which functions might lack coverage, helping you identify areas that need more testing.

## Testing Strategies for Different Types of Code {#testing-strategies-for-different-types-of-code}

### Unit Testing {#unit-testing}

- Test individual functions and classes in isolation

- Mock dependencies

- Focus on code behavior, not implementation details

### Integration Testing {#integration-testing}

- Test how components work together

- Focus on interfaces between components

- Minimize mocking when testing integration points

### Hardware-Aware Testing {#hardware-aware-testing}

- Use parameterized tests for different hardware configurations

- Mock hardware detection to test all supported hardware

- Include tests that verify hardware-specific optimization code paths

## Additional Resources {#additional-resources}

- [pytest Documentation](https://docs.pytest.org/)

- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)

---
**Previous**: [Code Quality Standards](code-quality.md) | **Next**: [Linting](linting.md) | __See also_*: [Contributing Guide](contributing.md)


```text`

```text`
