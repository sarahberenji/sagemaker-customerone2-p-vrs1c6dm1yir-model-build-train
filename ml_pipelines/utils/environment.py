import boto3
import sagemaker


def get_environment(project_name, ssm_params):
    sm = boto3.client("sagemaker")
    ssm = boto3.client("ssm")

    r = sm.describe_domain(
            DomainId=sm.describe_project(
                ProjectName=project_name
                )["CreatedBy"]["DomainId"]
        )
    del r["ResponseMetadata"]
    del r["CreationTime"]
    del r["LastModifiedTime"]
    r = {**r, **r["DefaultUserSettings"]}
    del r["DefaultUserSettings"]

    i = {
        **r,
        **{t["Key"]:t["Value"]
            for t in sm.list_tags(ResourceArn=r["DomainArn"])["Tags"]
            if t["Key"] in ["EnvironmentName", "EnvironmentType"]}
    }

    for p in ssm_params:
        try:
            i[p["VariableName"]] = ssm.get_parameter(Name=f"{i['EnvironmentName']}-{i['EnvironmentType']}-{p['ParameterName']}")["Parameter"]["Value"]
        except:
            i[p["VariableName"]] = ""

    return i


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    ), sagemaker_client


def environment_data(project_name):
    # Dynamically load environmental SSM parameters - provide the list of the variables to load from SSM parameter store
    ssm_parameters = [
        {"VariableName": "DataBucketName", "ParameterName": "data-bucket-name"},
        {"VariableName": "ModelBucketName", "ParameterName": "model-bucket-name"},
        {"VariableName": "S3KmsKeyId", "ParameterName": "kms-s3-key-arn"},
        {"VariableName": "EbsKmsKeyArn", "ParameterName": "kms-ebs-key-arn"},
        {"VariableName": "TrustedDefaultKinesisAccount", "ParameterName": "TrustedDefaultKinesisAccount"},
    ]
    env_data = get_environment(project_name=project_name, ssm_params=ssm_parameters)
    env_data["ProcessingRole"] = env_data["ExecutionRole"]
    env_data["TrainingRole"] = env_data["ExecutionRole"]
    
    return env_data