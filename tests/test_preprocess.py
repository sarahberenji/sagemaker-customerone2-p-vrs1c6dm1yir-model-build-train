import pytest

import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.network import NetworkConfig

@pytest.mark.xfail
def test_that_you_wrote_tests():
    assert False, "No tests written"


def test_pipelines_importable():
    from ml_pipelines.training import pipeline  # noqa: F401
    pipeline.get_pipeline(
        "eu-north-1",
        project_name="test",
        model_package_group_name="AbalonePackageGroup",
        pipeline_name="AbalonePipeline",
        base_job_prefix="Abalone",
    )


def mock_environment_data():
    env_data = {
        "DomainArn": "arn:aws:sagemaker:eu-north-1:000000000000:domain/d-jvv0klpjqwtj",
        "DomainId": "d-abcdefghijkl",
        "DomainName": "s-000000000000-dev-eu-north-1-sagemaker-domain",
        "HomeEfsFileSystemId": "fs-00000000000000000",
        "Status": "InService",
        "AuthMode": "IAM",
        "AppNetworkAccessType": "VpcOnly",
        "SubnetIds": [
            "subnet-00000000000000000",
            "subnet-00000000000000000"
        ],
        "Url": "https://d-abcdefghijkl.studio.eu-north-1.sagemaker.aws",
        "VpcId": "vpc-00000000000000000",
        "KmsKeyId": "0000000a-a000-40f5-975f-a00000000000",
        "ExecutionRole": "arn:aws:iam::000000000000:role/sm-mlops-env-EnvironmentIAM-SageMakerExecutionRole-0000000000000",
        "SecurityGroups": [
            "sg-00000000000000000"
        ],
        "EnvironmentName": "s-000000000000",
        "EnvironmentType": "dev",
        "DataBucketName": "s-000000000000-dev-eu-north-1-data",
        "ModelBucketName": "s-000000000000-dev-eu-north-1-models",
        "S3KmsKeyId": "arn:aws:kms:eu-north-1:000000000000:key/00000000-0000-4f03-95bd-000000000000",
        "EbsKmsKeyArn": "arn:aws:kms:eu-north-1:000000000000:key/00000000-0000-4849-8128-000000000000"
    }
    env_data["ProcessingRole"] = env_data["ExecutionRole"]
    env_data["TrainingRole"] = env_data["ExecutionRole"]
    return env_data


def test_pipelines_standard_model():
    from ml_pipelines.training import pipeline
    from sagemaker.local import LocalSession
    from ml_pipelines.training.models.standard_model import standard_model_pipeline


    region = "eu-north-1"
    project_name = "test"
    model_package_group_name = "AbalonePackageGroup"
    pipeline_name = "AbalonePipeline"
    default_bucket = "some-s3-bucket"
    base_job_prefix = "Abalone"
    env_data = mock_environment_data()
    sagemaker_session = sagemaker_local_session()
    base_dir = os.getcwd()
    pipeline = standard_model_pipeline(base_job_prefix, default_bucket, env_data, model_package_group_name,
                                       pipeline_name, region, sagemaker_session, base_dir)

def test_preprocess():
    from source_scripts.preprocessing.preprocess import preprocess_abalone
    import os
    import pandas as pd
    print(os.getcwd())
    filepath = "../temp/abalone.csv"
    generate_abalone_csv(filepath)
    train, validation, test= preprocess_abalone(filepath)
    pd.DataFrame(train).to_csv("../temp/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv("../temp//validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv("../temp/test.csv", header=False, index=False)


def generate_abalone_csv(filepath):
    from pandas import DataFrame
    from sklearn.datasets import make_classification
    import numpy as np
    X2, Y2 = make_classification(n_samples=1000, n_features=8, n_redundant=1, n_informative=6, class_sep=1.0,
                                 flip_y=0.03, random_state=123)
    df = DataFrame(
        dict(
            label=Y2,
            feature1=X2[:, 0],
            feature2=X2[:, 1],
            feature3=X2[:, 2],
            feature4=X2[:, 3],
            feature5=X2[:, 4],
            feature6=X2[:, 5],
            feature7=X2[:, 6],
            feature8=X2[:, 7]
        )
    )
    # Fix categories
    df['label'] = np.where((df['label'] == 1) & (df['feature1'] > 1), 'I',
                           np.where((df['label'] == 1), 'M', 'F')
                           )
    df.feature8 = df.feature8.round()

    df.to_csv(filepath, header=False, index=False)


def sagemaker_local_session():
    from sagemaker.local import LocalSession
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
    return sagemaker_session
