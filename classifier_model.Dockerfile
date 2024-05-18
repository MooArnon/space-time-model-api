# Use a base image that includes Python and other dependencies
FROM python:3.9-alpine

USER root
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libpq-dev

# Set the working directory inside the container
WORKDIR /app

# Copy your model file and any other necessary files into the container
COPY xgboost.pkl /app/model.pkl
COPY framework /app/framework
COPY classifier_model_api.py /app/app.py
COPY classifier_requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 80

# Command to run your Flask app
CMD ["python", "app.py"]

# docker run --rm -p 80:80 model-xgboost
# sudo docker build -f classifier_model.Dockerfile -t model-xgboost .
