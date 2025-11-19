FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download ALFWorld data
ENV ALFWORLD_DATA=/app/alfworld_data
RUN alfworld-download

# Copy all source code
COPY *.py ./ 
COPY *.yaml ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create output directory
RUN mkdir -p /app/outputs

# Default command - run benchmark
CMD ["python", "main.py", "--num_trials", "2", "--num_envs", "9", "--run_name", "docker_run"]
