#!/usr/bin/env python
import os
import boto3
import re
import json
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import argparse
import time
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)


def fit_model(instance_type,
              output_path,
              s3_train_base_dir,
              s3_test_base_dir,
              f,
              subnets,
              security_group_ids,
              image_uri,
              preprocessing_categorical_encoder_min_samples_leaf="100",
              preprocessing_categorical_encoder_smoothing="1.0",
              over_sampler_sampling_strategy="0.5",
              estimator_learning_rate=0.2,
              ):
    """Fits a model using the specified algorithm.

       Args:
            instance_type: instance to use for Sagemaker Training job
            output_path: S3 URI as the location for the trained model artifact
            s3_train_base_dir: S3 URI for train datasets
            s3_test_base_dir: S3 URI for test datasets
            f: index represents a fold number in the K fold cross validation
            c: regularization parameter for SVM
            gamma: kernel coefficiency value
            kernel: kernel type for SVM algorithm


       Returns:
            Sagemaker Estimator created with given input parameters.
    """
    print("SARAH: cv.py > fit_model()")
    print(f"SARAH: cv.py > fit_model() > type(subnets)={type(subnets)}")
    print(f"SARAH: cv.py > fit_model() > subnets={subnets}")
    print(f"SARAH: cv.py > fit_model() > type(security_group_ids)={type(security_group_ids)}")
    print(f"SARAH: cv.py > fit_model() > security_group_ids={security_group_ids}")
    print(f"SARAH: cv.py > fit_model() > f={f}")
    print(f"SARAH: cv.py > fit_model() > preprocessing_categorical_encoder_min_samples_leaf={preprocessing_categorical_encoder_min_samples_leaf}")
    print(f"SARAH: cv.py > fit_model() > preprocessing_categorical_encoder_smoothing={preprocessing_categorical_encoder_smoothing}")
    print(f"SARAH: cv.py > fit_model() > over_sampler_sampling_strategy={over_sampler_sampling_strategy}")
    print(f"SARAH: cv.py > fit_model() > estimator_learning_rate={estimator_learning_rate}")

    sklearn_framework_version = '0.23-1'
    script_path = 'scikit_learn_iris.py'

    print(f"SARAH: cv.py > fit_model() > script_path={script_path}")
    # print(f"SARAH: cv.py > fit_model() > image_uri={image_uri}")
    print(f"SARAH: cv.py > fit_model() > sklearn_estimator.fit inputs train ={s3_train_base_dir}/{f}")
    print(f"SARAH: cv.py > fit_model() > sklearn_estimator.fit inputs test ={s3_test_base_dir}/{f}")

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    sklearn_estimator = SKLearn(
        entry_point=script_path,
        # image_uri=image_uri,
        instance_type=instance_type,
        framework_version=sklearn_framework_version,
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        hyperparameters={'preprocessing_categorical_encoder_min_samples_leaf': preprocessing_categorical_encoder_min_samples_leaf,
                         'preprocessing_categorical_encoder_smoothing': preprocessing_categorical_encoder_smoothing,
                         'over_sampler_sampling_strategy': over_sampler_sampling_strategy,
                         'estimator_learning_rate': estimator_learning_rate,
                         },
        metric_definitions=[{"Name": "test:score", "Regex": "model test score:(.*?);"}],
        subnets=[subnets],
        security_group_ids=[security_group_ids],
    )
    sklearn_estimator.fit(inputs={'train': f'{s3_train_base_dir}/{f}',
                                  'test': f'{s3_test_base_dir}/{f}'
                                  }, wait=False)
    return sklearn_estimator


def monitor_training_jobs(training_jobs, sm_client):
    """Monitors the submit training jobs for completion.

      Args:
         training_jobs: array of submitted training jobs
         sm_client: boto3 sagemaker client

    """
    print("SARAH: cv.py > monitor_training_jobs")
    all_jobs_done = False
    while not all_jobs_done:
        completed_jobs = 0
        print(f"SARAH: cv.py > monitor_training_jobs completed_jobs={completed_jobs}")
        for job in training_jobs:
            job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
            job_status = job_detail['TrainingJobStatus']
            print(f"SARAH: cv.py > monitor_training_jobs job_status={job_status}")
            if job_status.lower() in ('completed', 'failed', 'stopped'):
                completed_jobs += 1
        if completed_jobs == len(training_jobs):
            all_jobs_done = True
        else:
            time.sleep(30)


def evaluation(training_jobs, sm_client):
    """Evaluates and calculate the performance for the cross validation training jobs.

       Args:
         training_jobs: array of submitted training jobs
         sm_client: boto3 sagemaker client

       Returns:
         Average score from the training jobs collection in the given input
    """
    print("SARAH: cv.py > evaluation")
    scores = []
    for job in training_jobs:
        job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
        print(f"SARAH: cv.py > evaluation() > job_detail = {job_detail}")
        metrics = job_detail['FinalMetricDataList']
        print(f"SARAH: cv.py > evaluation() > type(metrics)={type(metrics)}")
        print(f"SARAH: cv.py > evaluation() > metrics={metrics}")
        score = [x['Value'] for x in metrics if x['MetricName'] == 'test:score'][0]
        scores.append(score)

    np_scores = np.array(scores)

    # Calculate the score by taking the average score across the performance of the training job
    score_avg = np.average(np_scores)
    logging.info(f'average model test score:{score_avg};')
    return score_avg


def train():
    """
    Trains a Cross Validation Model with the given parameters.

    """
    print("SARAH: cv.py > train")
    parser = argparse.ArgumentParser()
    image_uri = "813736554012.dkr.ecr.eu-north-1.amazonaws.com/engineering-custom-images:crossvalidation"

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('--train_src', type=str)
    parser.add_argument('--test_src', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--instance_type', type=str, default="ml.c4.xlarge")
    parser.add_argument('--region', type=str, default="us-east-2")
    parser.add_argument('--subnets', type=str, default="subnet-0724be5e7071e7070")
    parser.add_argument('--security_group_ids', type=str, default="sg-041054ee4500f96f6")
    # parser.add_argument('--subnets', type=str)  # TODO: these shouldn't be hardcoded
    # parser.add_argument('--security_group_ids', type=str)
    # --security_group_ids sg-041054ee4500f96f6 --subnets subnet-0724be5e7071e7070
    parser.add_argument('--preprocessing_categorical_encoder_min_samples_leaf', type=int)
    parser.add_argument('--preprocessing_categorical_encoder_smoothing', type=float)
    parser.add_argument('--over_sampler_sampling_strategy', type=float)
    parser.add_argument('--estimator_learning_rate', type=float)

    args = parser.parse_args()
    print(f"SARAH: cv.py > train args ={args}")

    os.environ['AWS_DEFAULT_REGION'] = args.region
    sm_client = boto3.client("sagemaker")
    training_jobs = []
    # Fit k training jobs with the specified parameters.
    for f in range(args.k):
        sklearn_estimator = fit_model(instance_type=args.instance_type,
                                      output_path=args.output_path,
                                      s3_train_base_dir=args.train_src,
                                      s3_test_base_dir=args.test_src,
                                      f=f,
                                      subnets=args.subnets,
                                      security_group_ids=args.security_group_ids,
                                      image_uri=image_uri,
                                      preprocessing_categorical_encoder_min_samples_leaf=args.preprocessing_categorical_encoder_min_samples_leaf,
                                      preprocessing_categorical_encoder_smoothing=args.preprocessing_categorical_encoder_smoothing,
                                      over_sampler_sampling_strategy=args.over_sampler_sampling_strategy,
                                      estimator_learning_rate=args.estimator_learning_rate,
                                      )
        training_jobs.append(sklearn_estimator)
        time.sleep(5)  # sleeps to avoid Sagemaker Training Job API throttling
    print(f"training_jobs= {training_jobs}")
    monitor_training_jobs(training_jobs=training_jobs, sm_client=sm_client)
    score = evaluation(training_jobs=training_jobs, sm_client=sm_client)
    return score


if __name__ == '__main__':
    print("SARAH: cv.py > __main__")
    train()
    sys.exit(0)

