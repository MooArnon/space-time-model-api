#!/bin/sh

echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" > .env
echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> .env
echo "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION" >> .env
echo "AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID" >> .env
echo "MODEL=$MODEL" >> .env

# Function to handle Docker daemon shutdown gracefully
cleanup() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Shutting down Docker daemon..."
    kill -s SIGTERM $DOCKER_PID
    wait $DOCKER_PID
}

# Start the Docker daemon in the background and capture the PID
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Docker daemon..."
dockerd &
DOCKER_PID=$!

# Wait for the Docker daemon to start
while ! docker info > /dev/null 2>&1; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting for Docker daemon to start..."
    sleep 1
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Docker daemon started successfully."

# Set DOCKER_HOST if not already set
export DOCKER_HOST=${DOCKER_HOST:-unix:///var/run/docker.sock}

# Trap signals to shut down the Docker daemon properly
trap cleanup SIGINT SIGTERM

# Attempt to build the Docker image
if ! docker build -t train-model -f train_model.Dockerfile .; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Docker build failed."
    exit 1
fi

# Pull model from S3
#aws s3 cp s3://space-time-model/classifier/btc/xgboost/xgboost.pkl xgboost.pkl

# Attempt to build the model image
#source deploy_model.sh

# Execute the command passed to the container
echo "$(date '+%Y-%m-%d %H:%M:%S') - Executing the command: $@"
exec "$@"
