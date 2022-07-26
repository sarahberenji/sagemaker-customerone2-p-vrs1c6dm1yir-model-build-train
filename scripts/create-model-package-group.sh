#!/bin/bash
aws sagemaker create-model-package-group \
  --model-package-group-name "${MODEL_PACKAGE_GROUP_NAME}" \
  --model-package-group-description "${SAGEMAKER_PROJECT_NAME} Inference Pipeline - for sagemaker project to deploy into stage and prod, use this id: '${MODEL_PACKAGE_GROUP_NAME}'" \
  --tags Key=sagemaker:project-name,Value="${SAGEMAKER_PROJECT_NAME}" Key=sagemaker:"project-id",Value="${SAGEMAKER_PROJECT_ID}" \
  --region $AWS_REGION \
  || echo "Cant create model package, probably because it already exists"