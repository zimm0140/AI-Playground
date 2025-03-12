FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY hardware_detection/ /app/hardware_detection/
COPY pyproject.toml README.md /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV CI_TESTING=true
ENV SIMULATED_HARDWARE=base

# Default command
CMD ["pytest", "-v", "hardware_detection/tests/"] 