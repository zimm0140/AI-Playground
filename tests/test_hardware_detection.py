#!/usr/bin/env python3
"""Unit tests for hardware detection module."""

import os
import sys
import unittest
from unittest.mock import mock_open, patch

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hardware_detection


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
            },
        }

    @patch("hardware_detection.load_config")
    @patch("hardware_detection.get_gpu_info")
    def test_detect_arc_gpu(self, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Arc GPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock GPU info to simulate Arc GPU
        mock_get_gpu_info.return_value = ["Intel(R) Arc(TM) A770 Graphics"]

        # Test
        result = hardware_detection.detect_hardware_type()
        self.assertEqual(result, "acm")

    @patch("hardware_detection.load_config")
    @patch("hardware_detection.get_gpu_info")
    def test_detect_battlemage_gpu(self, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Battlemage GPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock GPU info to simulate Battlemage GPU
        mock_get_gpu_info.return_value = ["Intel(R) Battlemage(TM) B770 Graphics"]

        # Test
        result = hardware_detection.detect_hardware_type()
        self.assertEqual(result, "bmg")

    @patch("hardware_detection.load_config")
    @patch("hardware_detection.get_gpu_info")
    @patch("hardware_detection.get_cpu_info")
    def test_default_to_base(self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config):
        """Test falling back to base when no specific hardware is detected."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock hardware info with no specific hardware
        mock_get_gpu_info.return_value = ["NVIDIA GeForce RTX 3080"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) i9-9900K"}

        # Test
        result = hardware_detection.detect_hardware_type()
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
        config = hardware_detection.load_config()
        self.assertEqual(config["default_hardware"], "acm")

    @patch("pathlib.Path.exists")
    def test_load_config_fallback(self, mock_exists):
        """Test falling back to default config when file doesn't exist."""
        # Mock config file doesn't exist
        mock_exists.return_value = False

        # Test
        config = hardware_detection.load_config()
        self.assertEqual(config["default_hardware"], "base")
        self.assertIn("hardware_types", config)

    @patch("hardware_detection.detect_hardware_type")
    def test_get_hardware_info(self, mock_detect_hardware_type):
        """Test getting hardware information."""
        # Mock detect_hardware_type
        mock_detect_hardware_type.return_value = "acm"

        # Test
        info = hardware_detection.get_hardware_info()
        self.assertEqual(info["detected_hardware"], "acm")
        self.assertIn("system", info)
        self.assertIn("python_version", info)

    @patch("subprocess.check_output")
    def test_get_gpu_info_windows(self, mock_check_output):
        """Test getting GPU information on Windows."""
        # Mock subprocess for Windows
        mock_check_output.return_value = "Name\nIntel(R) Arc(TM) A770 Graphics"

        with patch("platform.system", return_value="Windows"):
            # Test
            result = hardware_detection.get_gpu_info_windows()
            self.assertEqual(result, ["Intel(R) Arc(TM) A770 Graphics"])


if __name__ == "__main__":
    unittest.main()
