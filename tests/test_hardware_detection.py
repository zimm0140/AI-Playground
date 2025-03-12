#!/usr/bin/env python3
"""Unit tests for hardware detection module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Add parent directory to path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from tools.hardware first, then fall back to root import
try:
    # Check if tools directory exists and add it to path
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if os.path.exists(tools_dir):
        sys.path.append(tools_dir)
    from tools.hardware import hardware_detection
except ImportError:
    # Fall back to root import
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
                "mtl": {"cpu_name_pattern": "Intel.*Core.*Ultra"},
                "lnl": {"cpu_name_pattern": "Intel.*Lunar Lake"},
                "ovino": {"platform_flags": ["has_openvino"]},
                "arl_h": {"platform_flags": ["arc_specific_flag"]},
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

    @patch("subprocess.check_output")
    def test_get_gpu_info_linux(self, mock_check_output):
        """Test getting GPU information on Linux."""
        # Mock subprocess for Linux
        mock_check_output.return_value = "00:02.0 VGA compatible controller: Intel Corporation Device 56a0 (rev 0c) (prog-if 00 [VGA controller])"

        # Test
        result = hardware_detection.get_gpu_info_linux()
        self.assertEqual(
            result[0],
            "00:02.0 VGA compatible controller: Intel Corporation Device 56a0 (rev 0c) (prog-if 00 [VGA controller])",
        )

    @patch("subprocess.check_output")
    def test_get_gpu_info_macos(self, mock_check_output):
        """Test getting GPU information on macOS."""
        # Mock subprocess for macOS
        mock_check_output.return_value = "Graphics/Displays:\n\n      Chipset Model: Apple M1 Pro"

        # Test
        result = hardware_detection.get_gpu_info_macos()
        self.assertEqual(result, ["Apple M1 Pro"])

    @patch("platform.system")
    @patch("hardware_detection.get_gpu_info_windows")
    @patch("hardware_detection.get_gpu_info_linux")
    @patch("hardware_detection.get_gpu_info_macos")
    def test_get_gpu_info(self, mock_macos, mock_linux, mock_windows, mock_system):
        """Test getting GPU information for different platforms."""
        # Test Windows
        mock_system.return_value = "Windows"
        mock_windows.return_value = ["NVIDIA GeForce RTX 3080"]
        result = hardware_detection.get_gpu_info()
        self.assertEqual(result, ["NVIDIA GeForce RTX 3080"])

        # Test Linux
        mock_system.return_value = "Linux"
        mock_linux.return_value = ["Intel Corporation Device 56a0"]
        result = hardware_detection.get_gpu_info()
        self.assertEqual(result, ["Intel Corporation Device 56a0"])

        # Test macOS
        mock_system.return_value = "Darwin"
        mock_macos.return_value = ["Apple M1 Pro"]
        result = hardware_detection.get_gpu_info()
        self.assertEqual(result, ["Apple M1 Pro"])

        # Test unsupported platform
        mock_system.return_value = "Unknown"
        result = hardware_detection.get_gpu_info()
        self.assertEqual(result, []

    @patch("platform.system")
    @patch("platform.processor")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="model name\t: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz",
    )
    def test_get_cpu_info(self, mock_file, mock_processor, mock_system):
        """Test getting CPU information for different platforms."""
        # Test Windows
        mock_system.return_value = "Windows"
        with patch("wmi.WMI") as mock_wmi:
            mock_processor_obj = MagicMock()
            mock_processor_obj.Name = "Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz"
            mock_processor_obj.Manufacturer = "Intel Corporation"
            mock_wmi.return_value.Win32_Processor.return_value = [mock_processor_obj]

            result = hardware_detection.get_cpu_info()
            self.assertEqual(result["name"], "Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz")
            self.assertEqual(result["manufacturer"], "Intel Corporation")

        # Test Linux
        mock_system.return_value = "Linux"
        result = hardware_detection.get_cpu_info()
        self.assertEqual(result["name"], "Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz")

        # Test other platforms
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1 Pro"
        result = hardware_detection.get_cpu_info()
        self.assertEqual(result["name"], "Apple M1 Pro")

    @patch("hardware_detection.load_config")
    @patch("hardware_detection.get_gpu_info")
    @patch("hardware_detection.get_cpu_info")
    def test_detect_meteor_lake_cpu(self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config):
        """Test detection of Intel Meteor Lake CPU."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock hardware info with Meteor Lake CPU
        mock_get_gpu_info.return_value = ["Intel(R) Graphics"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) Ultra 7 155H"}

        # Test
        result = hardware_detection.detect_hardware_type()
        self.assertEqual(result, "mtl")

    @patch("hardware_detection.is_openvino_available")
    @patch("hardware_detection.load_config")
    @patch("hardware_detection.get_gpu_info")
    @patch("hardware_detection.get_cpu_info")
    def test_detect_with_openvino(
        self, mock_get_cpu_info, mock_get_gpu_info, mock_load_config, mock_is_openvino
    ):
        """Test detection with OpenVINO available."""
        # Mock config
        mock_load_config.return_value = self.sample_config

        # Mock hardware info with no specific hardware but OpenVINO available
        mock_get_gpu_info.return_value = ["Intel(R) Graphics"]
        mock_get_cpu_info.return_value = {"name": "Intel(R) Core(TM) i7"}
        mock_is_openvino.return_value = True

        # Test
        result = hardware_detection.detect_hardware_type()
        self.assertEqual(result, "ovino")

    @patch("hardware_detection.load_config")
    def test_get_requirements_file(self, mock_load_config):
        """Test getting the appropriate requirements file for hardware types."""
        # Mock config with requirements
        config = {
            "requirements": {
                "base": "requirements.txt",
                "dev": "requirements-dev.txt",
                "hardware": {
                    "base": "requirements-hardware-base.txt",
                    "acm": "requirements-hardware-acm.txt",
                },
            }
        }
        mock_load_config.return_value = config

        # Test base hardware requirements
        result = hardware_detection.get_requirements_file("base")
        self.assertEqual(result, "requirements-hardware-base.txt")

        # Test specialized hardware requirements
        result = hardware_detection.get_requirements_file("acm")
        self.assertEqual(result, "requirements-hardware-acm.txt")

        # Test with dev flag
        result = hardware_detection.get_requirements_file("base", dev=True)
        self.assertEqual(result, ["requirements-hardware-base.txt", "requirements-dev.txt"])

        # Test fallback to base for unknown hardware
        result = hardware_detection.get_requirements_file("unknown_hardware")
        self.assertEqual(result, "requirements.txt")

    @patch("hardware_detection.get_hardware_info")
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
            with patch("hardware_detection.load_config") as mock_load_config:
                mock_load_config.return_value = self.sample_config
                hardware_detection.print_hardware_info(verbose=True)
                # Check that print was called more times with verbose flag
                self.assertGreater(mock_print.call_count, 7)  # At least 7 calls


if __name__ == "__main__":
    unittest.main()
