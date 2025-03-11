# Testing Guide

This guide explains how to run and write tests for AI-Playground, helping developers ensure their changes maintain code quality and functionality.

## Testing Philosophy

Tests in AI-Playground should:

1. **Verify behavior, not implementation**: Focus on testing what the code does, not how it does it
1. **Be independent**: Tests should not depend on other tests
1. **Be deterministic**: Tests should pass or fail consistently
1. **Be fast**: Tests should run quickly to encourage frequent testing
1. **Cover edge cases**: Test normal operation and exceptional conditions

## Test Structure

The tests are organized as follows:

- `tests/conftest.py`: Contains pytest fixtures shared across tests
- `tests/test_hardware_detection.py`: Tests for the hardware detection module
- `tests/test_uvfast.py`: Tests for the main uvfast.py module
- `tests/test_setup_hardware_env.py`: Tests for the setup_hardware_env.py module
- `tests/test_coverage.py`: Tests to assess and monitor test coverage
- `tests/utils/test_workflow_parser.py`: Tests for the workflow parser utilities

## Running Tests

### Running All Tests

To run all tests:

```bash
pytest

```text

### Running Tests with Detailed Output

To run tests with detailed output:

```bash
pytest -v

```text

### Running a Specific Test File

To run tests from a specific file:

```bash
pytest tests/test_hardware_detection.py

```text

### Running a Specific Test Function

To run a specific test function:

```bash
pytest tests/test_hardware_detection.py::TestHardwareDetection::test_detect_arc_gpu

```text

### Running Tests with Coverage Report

To run tests with a coverage report:

```bash
pytest --cov=. tests/

```text

For a more detailed coverage report:

```bash
pytest --cov=. --cov-report=html tests/

```text

This will generate an HTML coverage report in the `htmlcov` directory.

## Writing Tests

### Test File Naming

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Test Function Structure

A good test function should:

1. Set up the test environment (Arrange)
1. Execute the code being tested (Act)
1. Verify the expected outcomes (Assert)

Example:

```python
def test_detect_arc_gpu():

```text

# Arrange

```text

```text

mock_gpu_info = ["Intel(R) Arc(TM) A770 Graphics"]

```text

```text

# Act

```text

```text

with patch("hardware_detection.get_gpu_info", return_value=mock_gpu_info):

```text

result = hardware_detection.detect_hardware_type()

```text

```text

```text

# Assert

```text

```text

assert result == "acm"

```text

```text

### Using Fixtures

Fixtures are a powerful way to reuse test setup code:

```python
@pytest.fixture
def sample_config():

```text

"""Fixture providing a sample configuration."""
return {

```text

"hardware_types": ["base", "acm"],
"default_hardware": "base",

```text
}

```text

def test_hardware_detection(sample_config):

```text

# Use the sample_config fixture

```text

```text

with patch("hardware_detection.load_config", return_value=sample_config):

```text

# Test code here

```text

```text

```text

```text

pass

```text

```text

```text

### Mocking

Use mocking to isolate the code being tested:

```python
@patch("hardware_detection.get_gpu_info")
def test_gpu_detection(mock_get_gpu_info):

```text

# Configure the mock

```text

```text

mock_get_gpu_info.return_value = ["Intel(R) Arc(TM) A770 Graphics"]

```text

```text

# Test code that uses get_gpu_info

```text

```text

result = hardware_detection.detect_hardware_type()

```text

```text

# Verify results

```text

```text

assert result == "acm"

```text

```text

### Testing Hardware-Specific Code

When testing hardware-specific functionality:

1. **Mock hardware detection**: Don't rely on actual hardware for tests
1. **Test all hardware paths**: Ensure each hardware configuration works
1. **Test fallback behavior**: Verify code works with unsupported hardware
1. **Parameterize tests**: Use parameterized tests for different hardware

Example:

```python
@pytest.mark.parametrize("gpu_info,expected_type", [

```text

(["Intel(R) Arc(TM) A770 Graphics"], "acm"),
(["Intel(R) Battlemage(TM) B770 Graphics"], "bmg"),
(["NVIDIA GeForce RTX 3080"], "base"),

```text
])
def test_multiple_hardware_types(gpu_info, expected_type):

```text

with patch("hardware_detection.get_gpu_info", return_value=gpu_info):

```text

result = hardware_detection.detect_hardware_type()
assert result == expected_type

```text

```text

```text

## Testing Best Practices

1. **Test the public API**: Focus on testing public interfaces, not implementation details
1. **Keep tests simple**: Tests should be easy to understand
1. **Test one thing per test**: Each test should verify a single behavior
1. **Use meaningful test names**: Names should indicate what's being tested
1. **Don't test external dependencies**: Mock external dependencies
1. **Clean up after tests**: Tests should clean up any resources they create
1. **Don't modify production code for testing**: Use mocks and dependency injection instead

## Test Coverage Assessment

The `test_coverage.py` file provides utilities to assess test coverage without additional tools:

```bash
pytest tests/test_coverage.py -v

```text

This will show which core modules have tests and which functions might lack coverage, helping you identify areas that need more testing.

## Testing Strategies for Different Types of Code

### Unit Testing

- Test individual functions and classes in isolation
- Mock dependencies
- Focus on code behavior, not implementation details

### Integration Testing

- Test how components work together
- Focus on interfaces between components
- Minimize mocking when testing integration points

### Hardware-Aware Testing

- Use parameterized tests for different hardware configurations
- Mock hardware detection to test all supported hardware
- Include tests that verify hardware-specific optimization code paths

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)

---
**Previous**: [Code Quality Standards](code-quality.md) | **Next**: [Linting](linting.md) | **See also**: [Contributing Guide](contributing.md)
