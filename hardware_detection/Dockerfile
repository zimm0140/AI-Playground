FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files first to leverage caching
COPY setup.py /app/
COPY pyproject.toml /app/ 
COPY README.md /app/
COPY hardware_detection /app/hardware_detection/

# Install Python dependencies and test tools
RUN pip install --no-cache-dir pytest pytest-cov mypy typing-extensions

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV CI_TESTING=true
ENV SIMULATED_HARDWARE=base

# Create minimal test file to verify import works
RUN mkdir -p /app/hardware_detection/tests
RUN echo 'def test_import():\n\
    try:\n\
    import hardware_detection\n\
    from hardware_detection.core import get_hardware_info\n\
    print("Successfully imported hardware_detection")\n\
    assert hardware_detection is not None\n\
    # Try to get hardware info to make sure basic functionality works\n\
    info = get_hardware_info()\n\
    assert isinstance(info, dict)\n\
    assert "detected_hardware" in info\n\
    print(f"Hardware info retrieved: {info[\"detected_hardware\"]}")\n\
    return True\n\
    except ImportError as e:\n\
    print(f"Import error: {e}")\n\
    raise\n\
    ' > /app/hardware_detection/tests/test_import.py

# Verify installation and run basic test
RUN python -c "import hardware_detection; print(f'Hardware detection version: {hardware_detection.__version__}')" && \
    python -m pytest -xvs /app/hardware_detection/tests/test_import.py

# Default command for when the container is run
CMD ["python", "-m", "pytest", "-xvs", "hardware_detection/tests/"] 