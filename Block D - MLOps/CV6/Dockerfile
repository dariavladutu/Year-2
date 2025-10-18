# ---------- Base image -------------------------------------------------
FROM python:3.9-slim

# ---------- System libs for Pillow, OpenCV, curl (for health-check) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Work dir ---------------------------------------------------
WORKDIR /app

# ---------- Python deps ------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy application code & assets ----------------------------
COPY src ./src
COPY frontend_simple ./frontend_simple
COPY models ./models

# ---------- Environment tweaks ----------------------------------------
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ---------- Expose & launch -------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
