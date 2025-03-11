# Tests for AI-Playground

This directory contains tests for the AI-Playground project. The tests are designed to verify the functionality of the core modules without modifying the functional code.

## Test Structure

The tests are organized as follows:

- `conftest.py`: Contains pytest fixtures that are shared across tests
- `test_hardware_detection.py`: Tests for the hardware detection module
- `test_uvfast.py`: Tests for the main uvfast.py module
- `test_setup_hardware_env.py`: Tests for the setup_hardware_env.py module
- `test_coverage.py`: Tests to assess and monitor test coverage
- `utils/test_workflow_parser.py`: Tests for the workflow parser utilities

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

## Test Coverage Assessment

The `test_coverage.py` file provides utilities to assess test coverage without requiring additional tools. It checks whether core modules have corresponding test modules and whether the functions in core modules have corresponding test functions.

To run the coverage assessment:

```bash
pytest tests/test_coverage.py -v

```text

The output will show which core modules have tests and which functions might lack test coverage.

## Adding New Tests

When adding tests:

1. Create a file named `test_<module_name>.py` for the module you want to test
1. Use appropriate fixtures from `conftest.py` to set up test conditions
1. Follow the unittest or pytest patterns as shown in existing tests
1. Focus on testing functionality without modifying the implementation
1. Use the mock library to isolate tests from external dependencies

## Test Guidelines

- Tests should not modify the functionality of the code they are testing
- Tests should focus on verifying behavior, not implementation details
- Tests should be independent of each other
- Use mocks to isolate tests from external dependencies
- Aim for high test coverage, especially for critical paths
- Update tests when the corresponding code changes
