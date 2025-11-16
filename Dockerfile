# Backend Dockerfile for Plexe
# Multi-stage build optimized for large dependencies like torch

FROM python:3.11-slim as base

WORKDIR /app

# Set environment variables to improve pip/poetry performance
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies (for build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============== Dependencies Stage ==============
FROM base as deps

# Install poetry
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir poetry

# Copy only dependency files (for better layer caching)
COPY pyproject.toml poetry.lock* README.md ./

# Install dependencies only (no package installation yet)
# Retry on timeout with exponential backoff
RUN poetry install --no-interaction --no-ansi --no-root 2>&1 || \
    (echo "First attempt failed, retrying..." && sleep 5 && poetry install --no-interaction --no-ansi --no-root) || \
    (echo "Second attempt failed, retrying once more..." && sleep 10 && poetry install --no-interaction --no-ansi --no-root)

# ============== Application Stage ==============
FROM deps as app

# Copy application code (after deps are cached)
COPY . .

# Install the package itself
RUN poetry install --no-interaction --no-ansi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=30s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)"

# Run the server with auto-reload for development
CMD ["python", "-m", "uvicorn", "plexe.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]