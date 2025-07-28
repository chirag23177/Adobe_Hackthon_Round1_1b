# Stage 1: Runtime Environment
# Use a specific python version and set platform for AMD64 compatibility
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies that PyMuPDF might need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files, including the pre-downloaded 'models' directory
COPY . .

# Create a non-root user for better security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser


# Set the command to run your main script when the container starts
CMD ["python", "main.py"]
