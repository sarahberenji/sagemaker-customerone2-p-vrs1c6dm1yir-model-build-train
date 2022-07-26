#!/bin/bash
aws sagemaker create-pipeline \
  --pipeline-name "${PIPELINE_NAME}" \
  --pipeline-display-name "${PIPELINE_DISPLAY_NAME}" \
  --pipeline-definition-s3-location Bucket="${MODEL_BUCKET}",ObjectKey="${PIPELINE_PATH}" \
  --pipeline-description "Inference Pipeline" \
  --tags Key=sagemaker:project-name,Value="${SAGEMAKER_PROJECT_NAME}" Key=sagemaker:"project-id",Value="${SAGEMAKER_PROJECT_ID}" \
  --role-arn "${SAGEMAKER_PIPELINE_ROLE_ARN}" || \
aws sagemaker update-pipeline \
  --pipeline-name "${PIPELINE_NAME}" \
  --pipeline-display-name "${PIPELINE_DISPLAY_NAME}" \
  --pipeline-definition-s3-location Bucket="${MODEL_BUCKET}",ObjectKey="${PIPELINE_PATH}" \
  --pipeline-description "Inference Pipeline" \
  --role-arn "${SAGEMAKER_PIPELINE_ROLE_ARN}"