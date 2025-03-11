# syntax=docker/dockerfile:1

# Base Python image with common dependencies
FROM python:3.10-slim as base

WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_INSTALL_DIR="/tmp/uv" \
    PATH="$PATH:/tmp/uv/bin"

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency installation
RUN curl -sSf https://astral.sh/uv/install.sh | sh

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./
COPY requirements-hardware-*.txt ./

# Development image with all dependencies
FROM base as development

# Install all dependencies including development requirements
RUN uv pip install -r requirements.txt -r requirements-dev.txt

# Copy the application code
COPY . .

# Set up pre-commit
RUN pre-commit install

# Production image with minimal dependencies
FROM base as production

# Install just the runtime dependencies
RUN uv pip install -r requirements.txt

# Copy only the necessary files
COPY service/ ./service/
COPY scripts/ ./scripts/
COPY README.md ./

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Entry point - can be overridden
ENTRYPOINT ["python", "-m", "service"]

# OpenVINO image with specific dependencies
FROM production as openvino

# Install OpenVINO dependencies
COPY requirements-hardware-ovino.txt ./
RUN uv pip install -r requirements-hardware-ovino.txt

# Arc GPU image with specific dependencies
FROM production as arcgpu

# Install Intel Arc GPU dependencies
COPY service/requirements-acm.txt ./
RUN uv pip install -r service/requirements-acm.txt

# Default configuration
FROM production

EXPOSE 8000

CMD ["python", "-m", "service", "--host", "0.0.0.0", "--port", "8000"] 