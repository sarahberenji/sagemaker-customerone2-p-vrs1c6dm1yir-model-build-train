"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import json

from ml_pipelines.inference.standard_model.batch.standard_model import standard_model_pipeline
from ml_pipelines.utils.environment import get_session, environment_data
from ml_pipelines.utils.model import get_latest_model_metadata

def get_pipeline(
        region,
        project_name=None,
        source_scripts_path="./",
        model_package_group_name="AbalonePackageGroup",
        pipeline_name="AbalonePipeline",
        base_job_prefix="Abalone",
        revision="no-revision-provided",

):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        @todo arg. definitions

    Returns:
        an instance of a pipeline
    """

    # get env data
    env_data = environment_data(project_name)
    print(f"Environment data:\n{json.dumps(env_data, indent=2)}")

    sagemaker_session, sagemaker_client = get_session(region, env_data["DataBucketName"])
    default_bucket = sagemaker_session.default_bucket()
    base_dir = os.getcwd()
    print(f"Creating the pipeline '{pipeline_name}':")
    print(f"Parameters:{region}\n{env_data['SecurityGroups']}\n{env_data['SubnetIds']}\n{env_data['ProcessingRole']}\n\
    {env_data['TrainingRole']}\n{env_data['DataBucketName']}\n{env_data['ModelBucketName']}\n{model_package_group_name}\n\
    {pipeline_name}\n{base_job_prefix}\n{env_data['TrustedDefaultKinesisAccount']}")
    model_metadata = get_latest_model_metadata(sagemaker_client,model_package_group_name)
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
        revision = revision)
    return pipeline