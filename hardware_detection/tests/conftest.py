"""Pytest configuration for hardware detection tests."""

import os

import pytest


@pytest.fixture
def mock_dir(tmp_path):
    """Create a temporary directory for mock files."""
    mock_dir = tmp_path / "mock"
    mock_dir.mkdir()
    return mock_dir


@pytest.fixture
def mock_base_env(mock_dir):
    """Set up a mock base environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Generic GPU\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Generic\n")
        f.write("name: Generic CPU\n")
        f.write("cores: 4\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "base"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)


@pytest.fixture
def mock_acm_env(mock_dir):
    """Set up a mock Intel Arc environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) Arc(TM) A770 Graphics\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i9-13900K\n")
        f.write("cores: 24\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "acm"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)


@pytest.fixture
def mock_ovino_env(mock_dir):
    """Set up a mock OpenVINO environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) UHD Graphics 770\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i7-1370P\n")
        f.write("cores: 16\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "ovino"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock uvfast.json config file."""
    config_path = tmp_path / "uvfast.json"

    config_content = {
        "hardware_types": ["base", "acm", "ovino", "test_hw"],
        "default_hardware": "base",
        "detection": {
            "acm": {
                "gpu_name_pattern": "Intel.*Arc|Arc.*Graphics",
                "cpu_name_pattern": "Intel.*i9",
            },
            "ovino": {"cpu_name_pattern": "Intel.*i7", "package_check": "openvino"},
            "test_hw": {"gpu_name_pattern": "Test GPU", "cpu_name_pattern": "Test CPU"},
        },
    }

    import json

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_content, f, indent=2)

    return config_path


@pytest.fixture
def mock_base_env(monkeypatch):
    """Mock environment for base hardware setup."""
    monkeypatch.setenv("SIMULATED_HARDWARE", "base")


@pytest.fixture
def mock_acm_env(monkeypatch):
    """Mock environment for Intel Arc hardware setup."""
    monkeypatch.setenv("SIMULATED_HARDWARE", "acm")


@pytest.fixture
def mock_ovino_env(monkeypatch):
    """Mock environment for OpenVINO hardware setup."""
    monkeypatch.setenv("SIMULATED_HARDWARE", "ovino")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment before each test."""
    monkeypatch.delenv("SIMULATED_HARDWARE", raising=False)
