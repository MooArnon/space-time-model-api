# Use the Python 3.9 slim base image
FROM python:3.9-slim-buster AS build

# Install build dependencies and system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    libgomp1 \
    libhdf5-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy only the requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies (suppress root user warning)
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir --root-user-action=ignore -r /app/requirements.txt --timeout 600

# Copy the rest of the application
COPY . /app/

# Use a smaller image for the final stage
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy installed Python packages and application from build stage
COPY --from=build /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=build /app /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
