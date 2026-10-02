FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy files required to build the Python package
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies and the project
RUN uv sync --frozen --no-dev

# Copy application code
COPY api/ ./api/
COPY static/ ./static/
COPY models/ ./models/

# Make src package importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]