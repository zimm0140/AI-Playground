FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project directory
COPY . /app/

# Create necessary directory structure if it doesn't exist
RUN mkdir -p /app/hardware_detection/tests

# Create minimal test file if none exists
RUN if [ ! -d "/app/hardware_detection/tests" ]; then \
    mkdir -p /app/hardware_detection/tests && \
    echo 'def test_import():\n    import hardware_detection\n    assert hardware_detection' > /app/hardware_detection/tests/test_import.py; \
    fi

# Install Python dependencies
RUN pip install --no-cache-dir pytest pytest-cov

# Set environment variables
ENV PYTHONPATH=/app
ENV CI_TESTING=true
ENV SIMULATED_HARDWARE=base

# Default command
CMD ["python", "-m", "pytest", "-v"] 