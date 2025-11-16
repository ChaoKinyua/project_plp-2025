# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Create required directories
RUN mkdir -p logs data/raw data/processed visualization/outputs models/checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose port for API server (use 8000 for Waitress)
EXPOSE 8000

# Default: run the API via Waitress (production)
# If you want to run the analysis pipeline instead, override the CMD at runtime.
CMD ["python", "run_app_8000.py"]

# For alternative runs:
# docker run -p 8000:8000 stock-analysis:latest python main.py  # run pipeline
