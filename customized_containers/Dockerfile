FROM python:3.7
RUN apt-get update && pip install sagemaker boto3 numpy sagemaker-training

# Copies the training code inside the container
COPY cv.py /opt/ml/code/train.py
COPY scikit_learn_iris.py /opt/ml/code/scikit_learn_iris.py
#COPY lightgbm_training.py /opt/ml/code/lightgbm_training.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py

# This is required for CodeArtifact to take effect
ARG PIP_INDEX_URL

# Install custom packages specified in requirements.txt
COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt

# ENV PYTHONPATH "/usr/local/lib/python3.7/site-packages"
