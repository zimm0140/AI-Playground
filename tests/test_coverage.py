#!/usr/bin/env python3
"""Test coverage assessment and monitoring.

This module provides utilities to assess and monitor test coverage for the project.
It doesn't modify any functional code but helps ensure that tests adequately cover
the codebase.
"""

import importlib
import inspect
import os
import sys
import unittest
from types import ModuleType


class TestCoverage(unittest.TestCase):
    """Tests to assess and monitor test coverage."""

    def setUp(self):
        """Set up test environment."""
        # Add parent directory to path so we can import from the root
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if script_dir not in sys.path:
            sys.path.append(script_dir)

        # Core modules that should have tests
        self.core_modules = [
            "hardware_detection",
            "uvfast",
            "setup_hardware_env",
        ]

        # Test modules that should exist for each core module
        self.expected_test_modules = [
            "tests.test_hardware_detection",
            "tests.test_uvfast",
            "tests.test_setup_hardware_env",
        ]

    def test_core_modules_have_tests(self):
        """Test that core modules have corresponding test modules."""
        missing_tests = []

        for core_module in self.core_modules:
            test_module_name = f"tests.test_{core_module}"
            try:
                importlib.import_module(test_module_name)
            except ImportError:
                missing_tests.append(core_module)

        self.assertEqual(missing_tests, [], f"These core modules are missing tests: {missing_tests}")

    def test_core_functions_have_tests(self):
        """Test that core functions in modules have corresponding test functions."""
        # This is a check that helps ensure test coverage without using coverage tools
        modules_missing_tests = {}

        for core_module_name in self.core_modules:
            try:
                # Import the core module
                core_module = importlib.import_module(core_module_name)
                # Import the test module
                test_module_name = f"tests.test_{core_module_name}"
                try:
                    test_module = importlib.import_module(test_module_name)
                    # Get core module functions and test module functions
                    core_functions = self._get_module_functions(core_module)
                    test_functions = self._get_all_test_functions(test_module)

                    # Check for untested functions
                    untested_functions = []
                    for func_name in core_functions:
                        # Skip private functions (starting with underscore)
                        if func_name.startswith("_"):
                            continue

                        # Check if a test exists for this function
                        if not self._has_test_for_function(func_name, test_functions):
                            untested_functions.append(func_name)

                    if untested_functions:
                        modules_missing_tests[core_module_name] = untested_functions
                except ImportError:
                    # If no test module exists, all functions are untested
                    modules_missing_tests[core_module_name] = core_functions
            except ImportError:
                self.fail(f"Could not import core module {core_module_name}")

        # This is an informational test - it won't fail as long as modules import successfully
        if modules_missing_tests:
            print("\nFunctions that may need test coverage:")
            for module, funcs in modules_missing_tests.items():
                print(f"\n{module}:")
                for func in funcs:
                    print(f"  - {func}")

    def _get_module_functions(self, module: ModuleType) -> list[str]:
        """Get all functions defined in a module."""
        return [
            name
            for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__
        ]

    def _get_all_test_functions(self, test_module: ModuleType) -> list[str]:
        """Get all test functions from a test module."""
        test_functions = []

        # Get functions directly in the module
        direct_functions = [
            name
            for name, obj in inspect.getmembers(test_module)
            if inspect.isfunction(obj) and name.startswith("test_")
        ]
        test_functions.extend(direct_functions)

        # Get test methods from test classes
        for name, obj in inspect.getmembers(test_module):
            if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
                class_test_methods = [
                    method_name for method_name, _ in inspect.getmembers(obj) if method_name.startswith("test_")
                ]
                test_functions.extend(class_test_methods)

        return test_functions

    def _has_test_for_function(self, func_name: str, test_functions: list[str]) -> bool:
        """Check if there's a test for a specific function."""
        target_test_name = f"test_{func_name}"

        # Direct match
        if target_test_name in test_functions:
            return True

        # Check for more specific test functions that include the function name
        for test_func in test_functions:
            if test_func.startswith(target_test_name + "_"):
                return True

        return False


if __name__ == "__main__":
    unittest.main()
