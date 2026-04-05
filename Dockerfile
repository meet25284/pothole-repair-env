# syntax=docker/dockerfile:1

############################################
# Base Image
############################################
FROM python:3.12-slim

############################################
# Environment Settings
############################################
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

############################################
# Install system dependencies
############################################
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

############################################
# Set working directory
############################################
WORKDIR /app

############################################
# Upgrade pip
############################################
RUN python -m pip install --upgrade pip setuptools wheel

############################################
# Copy requirements first (Docker caching)
############################################
COPY requirements.txt .

############################################
# Install Python dependencies
############################################
RUN pip install -r requirements.txt

############################################
# Create non-root user
############################################
RUN useradd -m appuser

############################################
# Copy application code
############################################
COPY --chown=appuser:appuser . .

############################################
# Switch to non-root user
############################################
USER appuser

############################################
# Expose port (HuggingFace/Gradio/Inference)
############################################
EXPOSE 7860

############################################
# Run application (HTTP on 7860 — required for Hugging Face Docker Spaces)
############################################
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

