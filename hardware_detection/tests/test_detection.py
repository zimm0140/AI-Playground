"""Tests for hardware detection module."""

import os
from unittest import mock

from hardware_detection import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
)


def test_version():
    """Test that the module has a version."""
    from hardware_detection import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_detect_hardware_type_base(mock_base_env):
    """Test hardware type detection with base environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "base", f"Expected base, got {hw_type}"


def test_detect_hardware_type_acm(mock_acm_env):
    """Test hardware type detection with Intel Arc environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "acm", f"Expected acm, got {hw_type}"


def test_detect_hardware_type_ovino(mock_ovino_env):
    """Test hardware type detection with OpenVINO environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "ovino", f"Expected ovino, got {hw_type}"


def test_get_gpu_info_base(mock_base_env):
    """Test GPU info detection with base environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "Generic GPU" in gpus[0]


def test_get_gpu_info_acm(mock_acm_env):
    """Test GPU info detection with Intel Arc environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "Arc" in gpus[0]


def test_get_gpu_info_ovino(mock_ovino_env):
    """Test GPU info detection with OpenVINO environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "UHD" in gpus[0]


def test_get_cpu_info_base(mock_base_env):
    """Test CPU info detection with base environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "Generic CPU" in cpu_info["name"]
    assert cpu_info["cores"] == 4


def test_get_cpu_info_acm(mock_acm_env):
    """Test CPU info detection with Intel Arc environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "i9" in cpu_info["name"]
    assert cpu_info["cores"] == 24


def test_get_cpu_info_ovino(mock_ovino_env):
    """Test CPU info detection with OpenVINO environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "i7" in cpu_info["name"]
    assert cpu_info["cores"] == 16


def test_get_hardware_info(mock_acm_env):
    """Test hardware info retrieval."""
    info = get_hardware_info()
    assert isinstance(info, dict)
    assert "system" in info
    assert "gpus" in info
    assert "cpu" in info
    assert "detected_hardware" in info
    assert info["detected_hardware"] == "acm"


def test_is_openvino_available():
    """Test OpenVINO availability detection."""
    # Test with environment variable
    with mock.patch.dict(os.environ, {"SIMULATED_HARDWARE": "ovino"}):
        assert is_openvino_available()

    # Test without OpenVINO installed
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            assert not is_openvino_available()


def test_platform_specific_detection():
    """Test platform-specific detection paths."""
    # Test Windows path
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch(
            "hardware_detection.core.safe_run_command", return_value="Name\nTest GPU"
        ):
            gpus = get_gpu_info()
            assert "Test GPU" in gpus

    # Test Linux path
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch(
            "hardware_detection.core.safe_run_command",
            return_value="00:00.0 VGA compatible controller: Test GPU",
        ):
            gpus = get_gpu_info()
            assert "Test GPU" in gpus[0]

    # Test macOS path
    with mock.patch("platform.system", return_value="Darwin"):
        with mock.patch(
            "hardware_detection.core.safe_run_command",
            return_value="Chipset Model: Test GPU",
        ):
            gpus = get_gpu_info()
            assert "Test GPU" in gpus


def test_custom_config(mock_config_file, monkeypatch):
    """Test loading and using a custom config file."""
    from hardware_detection.core import load_config

    # Monkeypatch the file path check to find our mock config
    def mock_exists(path):
        return str(path) == str(mock_config_file)

    monkeypatch.setattr(os.path, "exists", mock_exists)

    # Override open to use our mock config
    orig_open = open

    def mock_open(*args, **kwargs):
        if args and str(args[0]) == str(mock_config_file):
            return orig_open(mock_config_file, *args[1:], **kwargs)
        return orig_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    # Load the config
    config = load_config()

    # Verify the config
    assert "test_hw" in config["hardware_types"]
    assert config["detection"]["test_hw"]["gpu_name_pattern"] == "Test GPU"
