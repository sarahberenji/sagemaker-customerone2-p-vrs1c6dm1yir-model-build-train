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
from ml_pipelines.training.models.standard_model import standard_model_pipeline
from ml_pipelines.utils.environment import get_session, environment_data


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def get_pipeline(
        region,
        project_name=None,
        source_scripts_path="./",
        model_package_group_name="AbalonePackageGroup",
        pipeline_name="AbalonePipeline",
        base_job_prefix="Abalone",
        revision="no-revision-provided",):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        @todo arg. definitions

    Returns:
        an instance of a pipeline
    """

    # get env data
    env_data = environment_data(project_name)
    print(f"Sarah: Environment data:\n{json.dumps(env_data, indent=2)}")

    sagemaker_session, sagemaker_client = get_session(region, env_data["DataBucketName"])
    default_bucket = sagemaker_session.default_bucket()
    base_dir = os.getcwd()
    print(f"Sarah: Creating the pipeline '{pipeline_name}':")
    print(f"Parameters:{region}\n{env_data['SecurityGroups']}\n{env_data['SubnetIds']}\n{env_data['ProcessingRole']}\n\
    {env_data['TrainingRole']}\n{env_data['DataBucketName']}\n{env_data['ModelBucketName']}\n{model_package_group_name}\n\
    {pipeline_name}\n{base_job_prefix}")

    print("Sarah: pipeline.py > get_pipeline() > calling standard_model_pipeline()")
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
        project=project_name,
        revision=revision)
    return pipeline

# TODO: Sarah, What does this part check?
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    print("##### OS CODEBUILD ENV")
    print(os.environ["CODEBUILD_SRC_DIR"])
    list_files(os.environ["CODEBUILD_SRC_DIR"])
except:
    print("no codebuild env")
print(f"#### Current Working Dir is: {os.getcwd()}")
print(f"#### BASE_DIR is {BASE_DIR}")
list_files(BASE_DIR)
