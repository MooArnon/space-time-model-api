# Use the Python 3.9 slim base image
FROM python:3.9-buster

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    libgomp1 \
    libhdf5-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt deep_requirements_uninstall.txt /

# Install numpy, scipy, and scikit-learn
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt --timeout 600 && \
    pip uninstall -y -r deep_requirements_uninstall.txt && \
    pip install tensorflow keras && \
    rm requirements.txt deep_requirements_uninstall.txt

# Set the working directory
WORKDIR /app

# ARG MODEL_TYPE
# COPY ${MODEL_TYPE}.pkl /app/model.pkl

# Copy your application code to the container
COPY framework /app/framework
COPY config /app/config
COPY utils /app/utils
COPY classifier_model_api.py /app/classifier_model_api.py
