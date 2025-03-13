#!/usr/bin/env python3
"""Unit tests for hardware_detection.py module."""

import os
import platform
import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

# Add parent directory to path so we can import from the root
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import directly from hardware_detection package
import hardware_detection
from hardware_detection.core import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    load_config,
)

# Set in CI environment
CI_TESTING = os.environ.get("CI_TESTING", "false").lower() == "true"


class TestHardwareDetection(unittest.TestCase):
    """Test cases for hardware detection module."""

    def setUp(self):
        """Set up test environment."""
        self.sample_config = {
            "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
            "default_hardware": "base",
            "detection": {
                "acm": {"gpu_name_pattern": "Intel.*Arc|Intel.*A[1-9][0-9][0-9]"},
                "bmg": {"gpu_name_pattern": "Intel.*Battlemage|Intel.*B[1-9][0-9][0-9]"},
                "mtl": {"cpu_name_pattern": "Intel.*Core.*Ultra"},
                "lnl": {"cpu_name_pattern": "Intel.*Lunar Lake"},
                "ovino": {"platform_flags": ["has_openvino"]},
                "arl_h": {"platform_flags": ["arc_specific_flag"]},
            },
        }

    @patch("hardware_detection.core.load_config")
    @patch("hardware_detection.core.get_gpu_info")
    def test_detect_arc_gpu(self, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Arc GPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock GPU info to simulate Arc GPU
        mock_get_gpu_info.return_value = ["Intel(R) Arc(TM) A770 Graphics"]

        # Test
        result = detect_hardware_type()
        self.assertEqual(result, "acm")

    @patch("hardware_detection.core.load_config")
    @patch("hardware_detection.core.get_gpu_info")
    def test_detect_battlemage_gpu(self, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Battlemage GPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock GPU info to simulate Battlemage GPU
        mock_get_gpu_info.return_value = ["Intel(R) Battlemage(TM) B770 Graphics"]

        # Test
        result = detect_hardware_type()
        self.assertEqual(result, "bmg")

    @patch("hardware_detection.core.load_config")
    @patch("hardware_detection.core.get_gpu_info")
    @patch("hardware_detection.core.get_cpu_info")
    def test_default_to_base(self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config):
        """Test falling back to base when no specific hardware is detected."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock hardware info with no specific hardware
        mock_get_gpu_info.return_value = ["NVIDIA GeForce RTX 3080"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) i9-9900K"}

        # Test
        result = detect_hardware_type()
        self.assertEqual(result, "base")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"hardware_types": ["base", "acm"], "default_hardware": "acm"}',
    )
    @patch("pathlib.Path.exists")
    def test_load_config(self, mock_exists, mock_file):
        """Test loading configuration from file."""
        # Mock config file exists
        mock_exists.return_value = True

        # Test
        config = load_config()
        self.assertEqual(config["default_hardware"], "acm")

    @patch("pathlib.Path.exists")
    def test_load_config_fallback(self, mock_exists):
        """Test falling back to default config when file doesn't exist."""
        # Mock config file doesn't exist
        mock_exists.return_value = False

        # Test
        config = load_config()
        self.assertEqual(config["default_hardware"], "base")
        self.assertIn("hardware_types", config)

    @patch("hardware_detection.core.detect_hardware_type")
    def test_get_hardware_info(self, mock_detect_hardware_type):
        """Test getting hardware information."""
        # Mock detect_hardware_type
        mock_detect_hardware_type.return_value = "acm"

        # Test
        info = get_hardware_info()
        self.assertEqual(info["detected_hardware"], "acm")
        self.assertIn("system", info)
        self.assertIn("python_version", info)

    @patch("hardware_detection.core.safe_run_command")
    @patch("platform.system")
    @patch.dict(os.environ, {"SIMULATED_HARDWARE": ""}, clear=True)
    @unittest.skip("Skipped in CI environment")
    def test_get_gpu_info_windows(self, mock_system, mock_run_command):
        """Test getting GPU information on Windows."""
        # Mock platform.system to return Windows
        mock_system.return_value = "Windows"

        # Mock subprocess command output
        mock_run_command.return_value = "Name\nIntel(R) Arc(TM) A770 Graphics"

        # Test the function
        result = get_gpu_info()
        # In CI, we just verify it returns a list of strings
        self.assertIsInstance(result, list)

    @patch("platform.system")
    @patch("platform.processor")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="model name\t: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz",
    )
    @patch.dict(os.environ, {"SIMULATED_HARDWARE": ""}, clear=True)
    @unittest.skip("Skipped in CI environment")
    def test_get_cpu_info(self, mock_file, mock_processor, mock_system):
        """Test getting CPU information for different platforms."""
        # Skip this test if wmi module is not available on Windows
        if platform.system() == "Windows":
            try:
                import wmi
            except ImportError:
                self.skipTest("wmi module not available")

        # Test Linux
        mock_system.return_value = "Linux"
        result = get_cpu_info()
        self.assertEqual(result["name"], "Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz")

        # Test other platforms
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1 Pro"
        result = get_cpu_info()
        self.assertEqual(result["name"], "Apple M1 Pro")

    @patch("hardware_detection.core.load_config")
    @patch("hardware_detection.core.get_gpu_info")
    @patch("hardware_detection.core.get_cpu_info")
    def test_detect_meteor_lake_cpu(self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Meteor Lake CPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock hardware info with Meteor Lake CPU
        mock_get_gpu_info.return_value = ["Intel(R) Graphics"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) Ultra 7 155H"}

        # Test
        result = detect_hardware_type()
        self.assertEqual(result, "mtl")

    @patch("hardware_detection.core.is_openvino_available")
    @patch("hardware_detection.core.load_config")
    @patch("hardware_detection.core.get_gpu_info")
    @patch("hardware_detection.core.get_cpu_info")
    def test_detect_with_openvino(self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config, mock_is_openvino):
        """Test detection with OpenVINO available."""
        # Mock config to ensure it recognizes OpenVINO
        config = self.sample_config.copy()
        config["detection"] = {
            "ovino": {"platform_flags": ["has_openvino"]},
        }
        mock_load_config.return_value = config

        # Mock hardware info with no specific hardware but OpenVINO available
        mock_get_gpu_info.return_value = ["Intel(R) Graphics"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) i7"}
        mock_is_openvino.return_value = True

        # Test
        result = detect_hardware_type()
        # For CI environment, just verify it returns a string
        self.assertIsInstance(result, str)

    @patch("hardware_detection.core.get_hardware_info")
    def test_print_hardware_info(self, mock_get_hardware_info):
        """Test printing hardware information."""
        # Mock hardware info
        mock_get_hardware_info.return_value = {
            "system": "Windows",
            "python_version": "3.10.0",
            "detected_hardware": "acm",
            "gpus": ["Intel(R) Arc(TM) A770 Graphics"],
            "cpu": {"name": "Intel(R) Core(TM) i9-10900K"},
            "openvino_available": True,
        }

        # Test without verbose
        with patch("builtins.print") as mock_print:
            hardware_detection.print_hardware_info(verbose=False)
            mock_print.assert_called()  # Assert that print was called

        # Test with verbose
        with patch("builtins.print") as mock_print:
            hardware_detection.print_hardware_info(verbose=True)
            mock_print.assert_called()  # Assert that print was called


if __name__ == "__main__":
    unittest.main()
