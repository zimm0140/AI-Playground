FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files first to leverage caching
COPY setup.py /app/
COPY pyproject.toml /app/ 
COPY README.md /app/
COPY hardware_detection /app/hardware_detection/

# Create a test setup.py file within the hardware_detection directory
RUN echo '#!/usr/bin/env python3\n\
    from setuptools import setup, find_packages\n\
    \n\
    setup(\n\
    name="hardware_detection",\n\
    version="0.1.0",\n\
    packages=find_packages(),\n\
    install_requires=["typing-extensions>=4.0.0"],\n\
    )' > /app/hardware_detection/setup.py

# Ensure we have a test directory with a working test
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
    except ImportError as e:\n\
    print(f"Import error: {e}")\n\
    raise\n\
    ' > /app/hardware_detection/tests/test_import.py

# Install Python dependencies 
RUN pip install --no-cache-dir pytest pytest-cov

# First install the main package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV CI_TESTING=true
ENV SIMULATED_HARDWARE=base

# Run tests to verify everything works
RUN python -m pytest -xvs /app/hardware_detection/tests/test_import.py

# Default command for when the container is run
CMD ["python", "-m", "pytest", "-xvs", "hardware_detection/tests/"] 