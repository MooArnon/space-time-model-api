# Use Python 3.9 slim-buster as the base image
FROM python:3.9-slim-buster

# Copy requirements file first to leverage Docker's cache
COPY classifier_requirements.txt /requirements.txt

# Define build argument
ARG MODEL_TYPE

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r /requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

COPY ${MODEL_TYPE}.pkl /app/model.pkl
COPY framework /app/framework
COPY classifier_model_api.py /app/app.py 

# Define default command to run when the container starts
CMD ["python", "app.py"]