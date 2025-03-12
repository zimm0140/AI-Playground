"""Mock Intel GPU package for testing."""

__version__ = "1.0.0"


def get_device_info():
    """Return mock device information."""
    return {
        "name": "Intel(R) Arc(TM) A770 Graphics",
        "vendor": "Intel",
        "memory": 16384,  # MB
        "compute_units": 32,
    }


def is_available():
    """Check if Intel GPU is available (always returns True for mock)."""
    return True


def get_device_count():
    """Return mock device count."""
    return 1


class Device:
    """Mock Intel GPU Device class."""

    def __init__(self, device_id=0):
        """Initialize the mock Device."""
        self.id = device_id
        self.name = "Intel(R) Arc(TM) A770 Graphics"

    def get_info(self):
        """Get mock device info."""
        return get_device_info()

    def synchronize(self):
        """Mock synchronize method."""
        pass
