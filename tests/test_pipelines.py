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
    source_scripts_path = "s3://some-bucket/source_scripts"
    pipeline = standard_model_pipeline(base_job_prefix, default_bucket, env_data, model_package_group_name,
                                       pipeline_name, region, sagemaker_session, base_dir,source_scripts_path)
    from unittest.mock import Mock
    return Mock(return_value=0)

def test_pipelines_steps():
    from ml_pipelines.training import pipeline

    from ml_pipelines.training.models.standard_model import sagemaker_pipeline_parameters
    from ml_pipelines.training.models.standard_model import preprocessing
    from ml_pipelines.training.models.standard_model import training_tasks
    from ml_pipelines.training.models.standard_model import evaluation_tasks
    from ml_pipelines.training.models.standard_model import model_register_tasks

    region = "eu-north-1"
    project_name = "test"
    model_package_group_name = "AbalonePackageGroup"
    pipeline_name = "AbalonePipeline"
    default_bucket = "some-s3-bucket"
    base_job_prefix = "Abalone"
    base_dir = os.getcwd()
    env_data = mock_environment_data()
    print(env_data)
    input_data, model_approval_status, processing_instance_count, processing_instance_type, training_instance_type = sagemaker_pipeline_parameters(data_bucket=default_bucket)
    network_config = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=env_data["SecurityGroups"],
        subnets=env_data["SubnetIds"],
        encrypt_inter_container_traffic=True)
    print(network_config)
    sagemaker_session = sagemaker_local_session()
    #sagemaker_session=None

    #sklearn_processor = SKLearnProcessor(
    #    framework_version="0.23-1",
    #    instance_type=processing_instance_type,
    #    instance_count=processing_instance_count,
    #    base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
    #    sagemaker_session=None,
    #    role=env_data["ProcessingRole"],
    #    network_config=network_config,
    #    volume_kms_key=env_data["EbsKmsKeyArn"],
    #    output_kms_key=env_data["S3KmsKeyId"]
    #)

    #print(sklearn_processor)

    step_process = preprocessing(base_job_prefix,
                                 env_data, input_data, network_config, processing_instance_count,
                                 processing_instance_type, sagemaker_session, base_dir)
    print(step_process)

    # training step for generating model artifacts

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    step_train, xgb_train = training_tasks(base_job_prefix,
                                           env_data,
                                           image_uri,
                                           network_config,
                                           sagemaker_session,
                                           step_process,
                                           training_instance_type
                                           )

    # processing step for evaluation
    # evaluation_report, model_metrics, step_eval = evaluation_tasks(base_job_prefix,
    #                                                                env_data,
    #                                                                image_uri,
    #                                                                network_config,
    #                                                                processing_instance_type,
    #                                                                sagemaker_session,
    #                                                                step_process,
    #                                                                step_train , base_dir
    #                                                                )
    #
    # step_cond = model_register_tasks(evaluation_report,
    #                                  model_approval_status,
    #                                  model_metrics,
    #                                  model_package_group_name,
    #                                  network_config,
    #                                  step_eval,
    #                                  step_train,
    #                                  xgb_train)

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train,
               # step_eval, step_cond
               ],
        #steps=[step_process],
        sagemaker_session=sagemaker_session,
    )
    print(pipeline)


def sagemaker_local_session():
    from sagemaker.local import LocalSession
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
    return sagemaker_session
