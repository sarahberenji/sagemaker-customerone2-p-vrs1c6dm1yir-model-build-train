#!/bin/bash
IFS=',' read -r -a SUBNETS <<< "$SM_SUBNETS"
echo "Subnets:1 ${SUBNETS[0]} 2: ${SUBNETS[1]}"
echo "PROCESSING_INSTANCE_TYPE  ${PROCESSING_INSTANCE_TYPE}"
echo "PIPELINE_NAME  ${PIPELINE_NAME}"
# echo "SAGEMAKER_PIPELINE_ROLE_ARN  ${SAGEMAKER_PIPELINE_ROLE_ARN}"
echo "EBS_KMS_KEY_ARN  ${EBS_KMS_KEY_ARN}"
echo "S3_KMS_KEY_ID  ${S3_KMS_KEY_ID}"
echo "SM_SG  ${SM_SG}"
echo "DATA_BUCKET_ENV  ${DATA_BUCKET_ENV}"
echo "SAGEMAKER_PROJECT_NAME  ${SAGEMAKER_PROJECT_NAME}"
echo "SOURCE_HEADHASH  ${SOURCE_HEADHASH}"
echo "TRUSTED_KINESIS_ACCOUNT  ${TRUSTED_KINESIS_ACCOUNT}"
 # @todo fix automatic datetime for batchdata, snapshotdata and inferenceoutput.
EPOCH_TIME=$(date +%s)
EXECUTION_NAME="execution-rev-${SOURCE_HEADHASH}"
aws sagemaker start-pipeline-execution \
  --pipeline-name "${PIPELINE_NAME}" \
  --pipeline-execution-display-name "${EXECUTION_NAME}" \
  --pipeline-parameters \
      Name=ProcessingInstanceType,Value="${PROCESSING_INSTANCE_TYPE}" \
      Name=DataBucket,Value="${DATA_BUCKET_ENV}" \
      Name=Subnet1,Value="${SUBNETS[0]}" \
      Name=Subnet2,Value="${SUBNETS[1]}" \
      Name=SecurityGroup,Value="${SM_SG}" \
      Name=EbsKmsKeyArn,Value="${EBS_KMS_KEY_ARN}" \
      Name=S3KmsKeyId,Value="${S3_KMS_KEY_ID}" \
      Name=SourceAccount,Value="${TRUSTED_KINESIS_ACCOUNT}" \
      Name=TriggerID,Value="${EPOCH_TIME}" \
      Name=ExecutionTime,Value="${EPOCH_TIME}" \
      Name=DataBase,Value="ml-test-datasets_rl" \
      Name=AbaloneTable,Value="ml_abalone" \
      Name=FilterRings,Value="9" \
  --pipeline-execution-description "${EXECUTION_NAME}"
# @todo dev is needing to use this role arn:aws:iam::370702650160:role/sm-mlops-env-EnvironmentIAM-SageMakerExecutionRole-XKF5KHYJ7X4
# for processing Name=ProcessingRole,Value="${SAGEMAKER_PIPELINE_ROLE_ARN}" \
# check how to add sufficient grants for pipeline execution role arn:aws:iam::370702650160:role/sm-mlops-env-EnvironmentI-SageMakerPipelineExecuti-KV1MWMTBN2U8
# pipeline execution role can be fetched from this ssm: "mlops-dev-sm-pipeline-execution-role-arn"
