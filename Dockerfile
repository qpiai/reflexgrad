FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt constraints.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

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
