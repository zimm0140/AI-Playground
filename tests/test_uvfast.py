#!/usr/bin/env python3
"""Unit tests for uvfast.py module."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvfast


class TestUVFast(unittest.TestCase):
    """Test cases for UVFast class."""

    def setUp(self):
        """Set up test environment."""
        self.uv_fast = uvfast.UVFast()

        # Sample config for testing
        self.sample_config = {
            "project_name": "test-project",
            "python_version": "3.10",
            "hardware_types": ["base", "acm"],
            "default_hardware": "base",
            "requirements": {
                "base": "requirements.txt",
                "dev": "requirements-dev.txt",
                "hardware": {
                    "base": "requirements-hardware-base.txt",
                    "acm": "requirements-hardware-acm.txt",
                },
            },
            "lockfiles": {
                "base": "requirements.lock",
                "dev": "requirements-dev.lock",
                "hardware": {
                    "base": "requirements-hardware-base.lock",
                    "acm": "requirements-hardware-acm.lock",
                },
            },
        }

    @patch("builtins.open", new_callable=mock_open, read_data='{"project_name": "test-project"}')
    @patch("pathlib.Path.exists")
    def test_load_config(self, mock_exists, mock_file):
        """Test loading configuration from file."""
        # Mock config file exists
        mock_exists.return_value = True

        # Call _load_config through a dummy private method accessor
        config = self.uv_fast._load_config()

        # Verify the config contains expected values
        self.assertEqual(config["project_name"], "test-project")
        self.assertIn("hardware_types", config)

    @patch("pathlib.Path.exists")
    def test_load_config_default(self, mock_exists):
        """Test falling back to default config when file doesn't exist."""
        # Mock config file doesn't exist
        mock_exists.return_value = False

        # Call _load_config
        config = self.uv_fast._load_config()

        # Verify it falls back to default values
        self.assertEqual(config["project_name"], "ai-playground")
        self.assertEqual(config["python_version"], "3.10")
        self.assertIn("base", config["hardware_types"])

    @patch("uvfast.UVFast._load_config")
    @patch("uvfast.hardware_detection.detect_hardware_type")
    def test_simple_hardware_detection(self, mock_detect_hardware, mock_load_config):
        """Test simple hardware detection with hardware_detection module."""
        # Mock hardware detection
        mock_detect_hardware.return_value = "acm"

        # Mock config
        mock_load_config.return_value = self.sample_config

        # Test hardware detection
        hardware_type = self.uv_fast._simple_hardware_detection()
        self.assertEqual(hardware_type, "acm")

    @patch("uvfast.UVFast._load_config")
    @patch("uvfast.hardware_detection", None)  # Simulate hardware_detection not available
    def test_simple_hardware_detection_fallback(self, mock_load_config):
        """Test simple hardware detection falling back without hardware_detection module."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Test hardware detection fallback
        hardware_type = self.uv_fast._simple_hardware_detection()
        self.assertEqual(hardware_type, "base")  # Should use default hardware

    @patch("uvfast.UVFast._load_config")
    def test_get_venv_path(self, mock_load_config):
        """Test getting virtual environment path."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Test venv path
        venv_path = self.uv_fast._get_venv_path()
        self.assertIsInstance(venv_path, Path)
        self.assertTrue(str(venv_path).endswith(".venv"))

    @patch("uvfast.UVFast._load_config")
    @patch("uvfast.UVFast._get_venv_path")
    def test_get_python_executable(self, mock_venv_path, mock_load_config):
        """Test getting Python executable path."""
        # Mock config and venv path
        mock_load_config.return_value = self.sample_config
        test_venv_path = Path(".venv")
        mock_venv_path.return_value = test_venv_path

        # Test Python executable path
        python_path = self.uv_fast._get_python_executable()
        self.assertIsInstance(python_path, Path)

        # Check that it's formed correctly based on the platform
        if sys.platform == "win32":
            self.assertTrue(str(python_path).endswith(os.path.join("Scripts", "python.exe")))
        else:
            self.assertTrue(str(python_path).endswith(os.path.join("bin", "python")))

    @patch("uvfast.UVFast._load_config")
    def test_get_requirements_files(self, mock_load_config):
        """Test getting requirements files for different hardware types."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Test base hardware requirements
        req_files = self.uv_fast._get_requirements_files("base")
        self.assertIn("requirements-hardware-base.txt", req_files)

        # Test specialized hardware requirements
        req_files = self.uv_fast._get_requirements_files("acm")
        self.assertIn("requirements-hardware-acm.txt", req_files)

        # Test with dev flag
        req_files = self.uv_fast._get_requirements_files("base", dev=True)
        self.assertIn("requirements-hardware-base.txt", req_files)
        self.assertIn("requirements-dev.txt", req_files)


if __name__ == "__main__":
    unittest.main()
