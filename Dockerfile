# Use the official Docker image with DinD
FROM docker:20.10-dind

# Install necessary tools
RUN apk --no-cache add python3 py3-pip curl bash && \
    pip3 install awscli

# Set the working directory
WORKDIR /app/

# Copy necessary files into the image
COPY . /app/

# Prebuild training image
#RUN docker build -t train-model -f train_model.Dockerfile .

# Make entrypoint script executable and add shebang
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && \
    sed -i '1i#!/bin/sh' /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
