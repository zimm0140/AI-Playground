FROM python:3.13-slim AS builder

# Install uv
RUN pip install uv --no-cache-dir

# Set working directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY requirements.lock .
COPY requirements-dev.lock .

# Create and activate virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN uv pip sync requirements.lock

# Development stage (comment out for production)
FROM builder as development
RUN uv pip sync requirements-dev.lock
COPY . .
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

# Production stage
FROM python:3.13-slim AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"] 