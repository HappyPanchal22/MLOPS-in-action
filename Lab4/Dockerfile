# ── Base Image ────────────────────────────────────────────────
FROM python:3.9-slim

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="Bhumi"
LABEL description="S&P 500 Next-Day Direction Predictor — Dockerized LightGBM ML Pipeline"

# ── System Dependencies ───────────────────────────────────────
# libgomp1 is required by LightGBM for multi-threading support
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Source Code & Data ───────────────────────────────────
COPY src/ .

# ── Create models directory for saving artifacts ──────────────
RUN mkdir -p models

# ── Run the Full Pipeline ─────────────────────────────────────
CMD ["python", "pipeline.py"]