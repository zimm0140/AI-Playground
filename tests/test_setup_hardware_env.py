#!/usr/bin/env python3
"""Unit tests for setup_hardware_env.py module."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import setup_hardware_env


class TestSetupHardwareEnv(unittest.TestCase):
    """Test cases for setup_hardware_env module."""

    def test_parse_args_defaults(self):
        """Test argument parsing with default values."""
        with patch("sys.argv", ["setup_hardware_env.py"]):
            args = setup_hardware_env.parse_args()
            self.assertIsNone(args.hardware)
            self.assertFalse(args.dev)
            self.assertFalse(args.clean)
            self.assertEqual(args.venv_dir, ".venv")

    def test_parse_args_with_values(self):
        """Test argument parsing with provided values."""
        with patch(
            "sys.argv",
            [
                "setup_hardware_env.py",
                "--hardware",
                "acm",
                "--dev",
                "--clean",
                "--venv-dir",
                "custom_venv",
            ],
        ):
            args = setup_hardware_env.parse_args()
            self.assertEqual(args.hardware, "acm")
            self.assertTrue(args.dev)
            self.assertTrue(args.clean)
            self.assertEqual(args.venv_dir, "custom_venv")

    @patch("shutil.which")
    def test_is_uv_available(self, mock_which):
        """Test detection of uv availability."""
        # Test when uv is available
        mock_which.return_value = "/usr/bin/uv"
        self.assertTrue(setup_hardware_env.is_uv_available())

        # Test when uv is not available
        mock_which.return_value = None
        self.assertFalse(setup_hardware_env.is_uv_available())

    @patch("shutil.which")
    def test_is_venv_available(self, mock_which):
        """Test detection of venv module availability."""
        # Test when venv is available
        mock_which.side_effect = lambda cmd: "/usr/bin/python" if cmd == "python" else None

        # Mock the subprocess call to check for venv module
        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            self.assertTrue(setup_hardware_env.is_venv_available())

        # Test when venv is not available
        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_run.return_value = mock_process

            self.assertFalse(setup_hardware_env.is_venv_available())

    @patch("setup_hardware_env.is_venv_available")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("shutil.rmtree")
    def test_create_venv(self, mock_rmtree, mock_exists, mock_run, mock_is_venv):
        """Test virtual environment creation."""
        # Mock venv availability
        mock_is_venv.return_value = True

        # Test creating a new venv
        mock_exists.return_value = False
        mock_run.return_value.returncode = 0

        result = setup_hardware_env.create_venv(".venv")
        self.assertTrue(result)
        mock_rmtree.assert_not_called()  # Should not try to remove non-existent venv

        # Test clean install with existing venv
        mock_exists.return_value = True
        result = setup_hardware_env.create_venv(".venv", clean=True)
        mock_rmtree.assert_called_once()  # Should remove existing venv
        self.assertTrue(result)

        # Test failure case
        mock_run.return_value.returncode = 1
        result = setup_hardware_env.create_venv(".venv")
        self.assertFalse(result)

    def test_get_python_executable(self):
        """Test getting the Python executable path."""
        # Test on Windows
        with patch("platform.system", return_value="Windows"):
            path = setup_hardware_env.get_python_executable(".venv")
            self.assertEqual(path, Path(".venv/Scripts/python.exe"))

        # Test on non-Windows
        with patch("platform.system", return_value="Linux"):
            path = setup_hardware_env.get_python_executable(".venv")
            self.assertEqual(path, Path(".venv/bin/python"))

    @patch("subprocess.run")
    @patch("hardware_detection.get_requirements_file")
    def test_install_requirements(self, mock_get_req, mock_run):
        """Test requirements installation."""
        # Mock the requirements file path
        mock_get_req.return_value = "requirements-hardware-acm.txt"
        mock_run.return_value.returncode = 0

        # Test standard pip installation
        python_exec = Path(".venv/bin/python")
        result = setup_hardware_env.install_requirements(python_exec, "acm", dev=False, use_uv=False)
        self.assertTrue(result)

        # Test with dev dependencies
        mock_get_req.return_value = ["requirements-hardware-acm.txt", "requirements-dev.txt"]
        result = setup_hardware_env.install_requirements(python_exec, "acm", dev=True, use_uv=False)
        self.assertTrue(result)

        # Test with uv
        result = setup_hardware_env.install_requirements(python_exec, "acm", dev=True, use_uv=True)
        self.assertTrue(result)

        # Test failure case
        mock_run.return_value.returncode = 1
        result = setup_hardware_env.install_requirements(python_exec, "acm")
        self.assertFalse(result)

    @patch("hardware_detection.detect_hardware_type")
    def test_check_hardware_availability(self, mock_detect):
        """Test hardware availability checking."""
        # Test when requested hardware matches detected hardware
        mock_detect.return_value = "acm"
        result = setup_hardware_env.check_hardware_availability("acm")
        self.assertTrue(result)

        # Test when requested hardware doesn't match detected hardware
        mock_detect.return_value = "base"
        result = setup_hardware_env.check_hardware_availability("acm")
        self.assertFalse(result)

        # Test with auto-detection (None as hardware type)
        mock_detect.return_value = "mtl"
        result = setup_hardware_env.check_hardware_availability(None)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
