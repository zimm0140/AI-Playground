# Testing Guide

This guide explains how to run and write tests for AI-Playground, helping developers ensure their changes maintain code quality and functionality.

## Testing Philosophy

Tests in AI-Playground should:

1. **Verify behavior, not implementation**: Focus on testing what the code does, not how it does it
2. **Be independent**: Tests should not depend on other tests
3. **Be deterministic**: Tests should pass or fail consistently
4. **Be fast**: Tests should run quickly to encourage frequent testing
5. **Cover edge cases**: Test normal operation and exceptional conditions

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
```

### Running Tests with Detailed Output

To run tests with detailed output:

```bash
pytest -v
```

### Running a Specific Test File

To run tests from a specific file:

```bash
pytest tests/test_hardware_detection.py
```

### Running a Specific Test Function

To run a specific test function:

```bash
pytest tests/test_hardware_detection.py::TestHardwareDetection::test_detect_arc_gpu
```

### Running Tests with Coverage Report

To run tests with a coverage report:

```bash
pytest --cov=. tests/
```

For a more detailed coverage report:

```bash
pytest --cov=. --cov-report=html tests/
```

This will generate an HTML coverage report in the `htmlcov` directory.

## Writing Tests

### Test File Naming

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Test Function Structure

A good test function should:

1. Set up the test environment (Arrange)
2. Execute the code being tested (Act)
3. Verify the expected outcomes (Assert)

Example:

```python
def test_detect_arc_gpu():
    # Arrange
    mock_gpu_info = ["Intel(R) Arc(TM) A770 Graphics"]
    
    # Act
    with patch("hardware_detection.get_gpu_info", return_value=mock_gpu_info):
        result = hardware_detection.detect_hardware_type()
    
    # Assert
    assert result == "acm"
```

### Using Fixtures

Fixtures are a powerful way to reuse test setup code:

```python
@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration."""
    return {
        "hardware_types": ["base", "acm"],
        "default_hardware": "base",
    }

def test_hardware_detection(sample_config):
    # Use the sample_config fixture
    with patch("hardware_detection.load_config", return_value=sample_config):
        # Test code here
        pass
```

### Mocking

Use mocking to isolate the code being tested:

```python
@patch("hardware_detection.get_gpu_info")
def test_gpu_detection(mock_get_gpu_info):
    # Configure the mock
    mock_get_gpu_info.return_value = ["Intel(R) Arc(TM) A770 Graphics"]
    
    # Test code that uses get_gpu_info
    result = hardware_detection.detect_hardware_type()
    
    # Verify results
    assert result == "acm"
```

### Testing Hardware-Specific Code

When testing hardware-specific functionality:

1. **Mock hardware detection**: Don't rely on actual hardware for tests
2. **Test all hardware paths**: Ensure each hardware configuration works
3. **Test fallback behavior**: Verify code works with unsupported hardware
4. **Parameterize tests**: Use parameterized tests for different hardware

Example:

```python
@pytest.mark.parametrize("gpu_info,expected_type", [
    (["Intel(R) Arc(TM) A770 Graphics"], "acm"),
    (["Intel(R) Battlemage(TM) B770 Graphics"], "bmg"),
    (["NVIDIA GeForce RTX 3080"], "base"),
])
def test_multiple_hardware_types(gpu_info, expected_type):
    with patch("hardware_detection.get_gpu_info", return_value=gpu_info):
        result = hardware_detection.detect_hardware_type()
        assert result == expected_type
```

## Testing Best Practices

1. **Test the public API**: Focus on testing public interfaces, not implementation details
2. **Keep tests simple**: Tests should be easy to understand
3. **Test one thing per test**: Each test should verify a single behavior
4. **Use meaningful test names**: Names should indicate what's being tested
5. **Don't test external dependencies**: Mock external dependencies
6. **Clean up after tests**: Tests should clean up any resources they create
7. **Don't modify production code for testing**: Use mocks and dependency injection instead

## Test Coverage Assessment

The `test_coverage.py` file provides utilities to assess test coverage without additional tools:

```bash
pytest tests/test_coverage.py -v
```

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
