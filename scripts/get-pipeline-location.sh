#!/bin/bash
echo "Get ARN for latest package in model group"
#
PACKAGE_GROUP_NAME=$1 #MODEL_PACKAGE_GROUP_NAME
APPROVAL_STATUS=$2 #PendingManualApproval
echo PACKAGE_GROUP_NAME: $PACKAGE_GROUP_NAME , APPROVAL: $APPROVAL_STATUS
model_arn=$(aws sagemaker list-model-packages --model-package-group ${PACKAGE_GROUP_NAME} --query "ModelPackageSummaryList[?ModelApprovalStatus==\`${APPROVAL_STATUS}\`].ModelPackageArn | [0]" --output text)
echo "Get s3 URI to inference pipeline"
PIPELINE_S3_URI=$(aws sagemaker describe-model-package --model-package-name ${model_arn} --query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text)
MODEL_BUCKET=$(awk '{ sub(/.*s3:\/\//, ""); sub(/\/.*/, ""); print}' <<< $PIPELINE_S3_URI)
PIPELINE_PATH=${PIPELINE_S3_URI#"s3://$MODEL_BUCKET/"}
printf "PIPELINE_S3_URI= $PIPELINE_S3_URI \n MODEL_BUCKET= $MODEL_BUCKET \n PIPELINE_PATH= $PIPELINE_PATH \n"
