import pytest

import os
import json
import boto3
import sagemaker
import sagemaker.session
from datetime import datetime
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

def mock_model_metadata():
    model_metadata = {'ModelPackageGroupName': 'batch-ml-p-asyqvjvmsjlu',
                      'ModelPackageVersion': 27,
                      'ModelPackageArn': 'arn:aws:sagemaker:eu-north-1:370702650160:model-package/batch-ml-p-asyqvjvmsjlu/27',
                      'CreationTime': "2022-02-02:00:00:00",
                      'InferenceSpecification': {'Containers': [{'Image': '662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3',
                                                                 'ImageDigest': 'sha256:04889b02181f14632e19ef6c2a7d74bfe699ff4c7f44669a78834bc90b77fe5a',
                                                                 'ModelDataUrl': 's3://s-370702650160-dev-eu-north-1-models/lifecycle/max/batch-ml/5ee7693/abalone/2022_04_06_12_38_45/output/pipelines-ei1m1ye9priq-TrainAbaloneModel-5ZCjepMJum/output/model.tar.gz'}],
                                                 'SupportedTransformInstanceTypes': ['ml.m5.large'],
                                                 'SupportedRealtimeInferenceInstanceTypes': ['ml.t2.medium', 'ml.m5.large'],
                                                 'SupportedContentTypes': ['text/csv'],
                                                 'SupportedResponseMIMETypes': ['text/csv']},
                      'ModelPackageStatus': 'Completed',
                      'ModelPackageStatusDetails': {'ValidationStatuses': [],
                                                    'ImageScanStatuses': []},
                      'CertifyForMarketplace': False,
                      'ModelApprovalStatus': 'PendingManualApproval',
                      'MetadataProperties': {'GeneratedBy': 'arn:aws:sagemaker:eu-north-1:370702650160:pipeline/batch-ml-p-asyqvjvmsjlu-training/execution/ei1m1ye9priq'},
                      'ModelMetrics': {'ModelQuality': {'Statistics': {'ContentType': 'application/json',
                                                                       'S3Uri': 's3://s-370702650160-dev-eu-north-1-data/lifecycle/max/batch-ml/5ee7693/2022_04_06_12_38_45/output/evaluation/evaluation.json'}},
                                       'Bias': {},
                                       'Explainability': {}},
                      'CustomerMetadataProperties': {'git_revision': '5ee7693',
                                                     'preprocess': 's3://s-370702650160-dev-eu-north-1-models/lifecycle/max/batch-ml/5ee7693/input/source_scripts/preprocessing/preprocess.py',
                                                     'postprocess': 's3://s-370702650160-dev-eu-north-1-models/lifecycle/max/batch-ml/5ee7693/input/source_scripts/postprocessing/postprocess.py'},
            
                      'ResponseMetadata': {'RequestId': '744cc000-4ddb-477a-a513-b8bdf9c19dbb',
                                           'HTTPStatusCode': 200,
                                           'HTTPHeaders': {'x-amzn-requestid': '744cc000-4ddb-477a-a513-b8bdf9c19dbb',
                                                           'content-type': 'application/x-amz-json-1.1',
                                                           'content-length': '1602',
                                                           'date': 'Wed, 06 Apr 2022 13:18:47 GMT'},
                                           'RetryAttempts': 0}}
    return model_metadata

def test_pipelines_standard_model():
    from ml_pipelines.training import pipeline
    from sagemaker.local import LocalSession
    from ml_pipelines.inference.standard_model.batch.standard_model import standard_model_pipeline


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
    revision="aaaaaaa"

    model_metadata = mock_model_metadata()
    pipeline = standard_model_pipeline(
        base_job_prefix=base_job_prefix,
        default_bucket=default_bucket,
        env_data=env_data,
        model_package_group_name=model_package_group_name,
        pipeline_name=pipeline_name,
        region=region,
        sagemaker_session=sagemaker_session,
        base_dir=base_dir,
        source_scripts_path=source_scripts_path,
        model_metadata=model_metadata,
        project=project_name,
        revision = revision
    )

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
