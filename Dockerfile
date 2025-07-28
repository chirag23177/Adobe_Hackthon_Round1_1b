# Document Intelligence System - CPU-Only Dockerfile
# Platform: linux/amd64
# Base: Python 3.9 slim for minimal footprint

FROM --platform=linux/amd64 python:3.9-slim

# Set metadata
LABEL maintainer="Adobe Hackathon Team"
LABEL description="CPU-only Document Intelligence System for offline PDF analysis"
LABEL version="1.0"

# Set environment variables for CPU-only execution
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Set working directory
WORKDIR /app

# Create input and output directories
RUN mkdir -p /app/input /app/output /app/input/PDFs

# Install system dependencies (minimal for PDF processing)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy source code
COPY main.py /app/
COPY plan.md /app/
COPY approach_explanation.md /app/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check to verify the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch, sentence_transformers, fitz; print('Dependencies OK')" || exit 1

# Volume mounts for input/output
VOLUME ["/app/input", "/app/output"]

# Default command - run the document intelligence pipeline
CMD ["python3", "main.py"]

# Entry point for debugging (can be overridden)
ENTRYPOINT ["python3"]
