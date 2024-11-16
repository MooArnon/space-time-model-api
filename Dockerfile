# Use the Python 3.9 slim base image
FROM public.ecr.aws/lambda/python:3.9

# Install build dependencies
RUN yum update -y && yum install -y git libgomp

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install numpy, scipy, and scikit-learn
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt --timeout 600

# Copy your application code to the container
COPY framework ${LAMBDA_TASK_ROOT}/framework
COPY config ${LAMBDA_TASK_ROOT}/config
COPY utils ${LAMBDA_TASK_ROOT}/utils
COPY *.py ${LAMBDA_TASK_ROOT}

CMD ["classifier_model_api.handler"]
