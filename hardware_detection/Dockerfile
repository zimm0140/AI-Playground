FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project directory
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir pytest pytest-cov

# Set environment variables
ENV PYTHONPATH=/app
ENV CI_TESTING=true
ENV SIMULATED_HARDWARE=base

# Default command
CMD ["python", "-m", "pytest", "-v", "hardware_detection/tests/"] 