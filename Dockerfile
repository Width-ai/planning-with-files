# ---------------------------------------------------------------------------
# Dockerfile – Planning With Files API
# ---------------------------------------------------------------------------
# Base: python:3.12-slim
# Installs dependencies via uv for speed, then runs uvicorn.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Copy dependency specification first (cache-friendly layer)
COPY pyproject.toml ./

# Install project dependencies using uv
RUN uv pip install --system --no-cache .

# Copy the API source code
COPY api/ ./api/

# Expose the API port
EXPOSE 8000

# Run the FastAPI application via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
