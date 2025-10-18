FROM python:3.9-slim

# Use bash shell for better compatibility with && and multiline RUN commands
SHELL ["/bin/bash", "-c"]

# Install system dependencies required by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files early to use Docker cache
COPY pyproject.toml poetry.lock ./

# Disable virtual environments and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi
# Copy source code
COPY . .

# Set PYTHONPATH to locate 'app'
ENV PYTHONPATH=/app/src

# Expose FastAPI default port
EXPOSE 8000

# Launch the FastAPI application
CMD ["poetry", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
