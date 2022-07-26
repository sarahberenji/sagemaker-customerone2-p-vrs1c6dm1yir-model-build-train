Get pip setup

`apt-get update`
`apt-get install -y python3-pip`

`pip3 install -e .`


> Set env vars based on https://eu-north-1.console.aws.amazon.com/codesuite/codebuild/370702650160/projects/sagemaker-batch-ml-p-asyqvjvmsjlu-modelbuild/build/sagemaker-batch-ml-p-asyqvjvmsjlu-modelbuild%3A8e90fe3d-4567-4f33-bc61-e22fec980613/env_var?region=eu-north-1
SAGEMAKER_PROJECT_NAME=batch-ml
ENV_TYPE=dev
SAGEMAKER_PIPELINE_NAME=sagemaker-batch-ml
AWS_REGION=eu-north-1
DATA_BUCKET=s-370702650160-dev-eu-north-1-data
ENV_NAME=s-370702650160
SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::370702650160:role/sm-mlops-env-EnvironmentI-SageMakerPipelineExecuti-146SN5QVNMQPM
SAGEMAKER_PROJECT_ID=p-asyqvjvmsjlu
SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"

run-pipeline \
--module-name ml_pipelines.training.pipeline \
--role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
--tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}, {\"Key\":\"EnvironmentName\", \"Value\":\"${ENV_NAME}\"}, {\"Key\":\"EnvironmentType\", \"Value\":\"${ENV_TYPE}\"}]" \
--kwargs "{\"region\":\"${AWS_REGION}\",\"project_name\":\"${SAGEMAKER_PROJECT_NAME}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\"}"

