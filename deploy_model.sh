#!/bin/bash

# Check if MODEL and DOCKER_FILE_NAME arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: No MODEL or DOCKER_FILE_NAME argument provided."
  echo "Usage: ./deploy_model.sh <MODEL> <DOCKER_FILE_NAME>"
  
fi

# Variables
MODEL=$1
DOCKER_FILE_NAME=$2
REPOSITORY_NAME="space-time/classifier-model"
REGION="ap-southeast-1"

# Fetch AWS account ID from environment if not already set
if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "Fetching AWS account ID..."
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  
  if [ $? -ne 0 ]; then
    echo "Error: Unable to fetch AWS account ID."
    
  fi
fi

LAMBDA_MODEL_NAME="${MODEL//_/-}"
LAMBDA_FUNCTION_NAME="arn:aws:lambda:ap-southeast-1:${AWS_ACCOUNT_ID}:function:predict-btc-$LAMBDA_MODEL_NAME"

# Check if fetching AWS account ID was successful
if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "Error: AWS_ACCOUNT_ID not found. Please set it or ensure AWS CLI is configured correctly."
  
fi

# Fetch the latest tag number from ECR
latest_tag=$(aws ecr describe-images --repository-name $REPOSITORY_NAME \
  --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]' --output text --region $REGION)

# Extract the numeric part of the latest tag if it exists
latest_tag_number=$(echo "$latest_tag" | grep -o '[0-9]\+')

if [ -n "$latest_tag_number" ]; then
  next_tag_number=$((latest_tag_number + 1))
else
  next_tag_number=1
fi

# Build the new tag (MODEL-NEXT_TAG_NUMBER)
FULL_TAG="$MODEL-$next_tag_number"
echo "Latest tag: $latest_tag, Next tag: $FULL_TAG"

# Build the Docker image with the new tag
echo "Building Docker image for tag: $FULL_TAG"
docker build -t $REPOSITORY_NAME:$FULL_TAG -f $DOCKER_FILE_NAME --build-arg MODEL_TYPE=$MODEL .

if [ $? -ne 0 ]; then
  echo "Error: Docker image build failed."
  
fi

# Tag the Docker image for ECR
echo "Tagging Docker image: $REPOSITORY_NAME:$FULL_TAG"
docker tag $REPOSITORY_NAME:$FULL_TAG $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$FULL_TAG

if [ $? -ne 0 ]; then
  echo "Error: Docker image tagging failed."
  
fi

# Log in to ECR
echo "Logging in to ECR"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

if [ $? -ne 0 ]; then
  echo "Error: ECR login failed."
  
fi

# Push the image to ECR
echo "Pushing Docker image to ECR with tag: $FULL_TAG"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$FULL_TAG

if [ $? -ne 0 ]; then
  echo "Error: Docker image push failed."
  
fi

echo "Image successfully pushed with tag: $FULL_TAG"

# Update Lambda function with the new image
echo "Updating Lambda function to use the new image: $FULL_TAG"
aws lambda update-function-code \
  --function-name $LAMBDA_FUNCTION_NAME \
  --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$FULL_TAG \
  > /dev/null 2>&1

if [ $? -ne 0 ]; then
  echo "Error: Failed to update Lambda function."
  
fi

echo "Lambda function successfully updated with image: $FULL_TAG"
