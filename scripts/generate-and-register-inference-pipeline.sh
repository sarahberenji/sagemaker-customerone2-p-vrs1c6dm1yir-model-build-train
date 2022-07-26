#!/bin/bash
get-pipeline-definition \
  --module-name ml_pipelines.inference.standard_model.batch.pipeline \
  --file-name temp/pipeline.json \
  --kwargs "{\"region\":\"${AWS_REGION}\",\"project_name\":\"${SAGEMAKER_PROJECT_NAME}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}-inference\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\", \"revision\":\"${SOURCE_HEADHASH}\", \"source_scripts_path\":\"${SOURCE_SCRIPTS_PATH}\"}"
PIPELINE_PATH=lifecycle/max/${SAGEMAKER_PROJECT_NAME}/${SOURCE_HEADHASH}/pipeline/inference/pipeline.json

aws s3 cp temp/pipeline.json "s3://${MODEL_BUCKET}/${PIPELINE_PATH}"

echo ">> Register inference pipeline as model package"
IMAGE=662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3
INFERENCE_SPEC="{\"Containers\":[{\"Image\":\"${IMAGE}\",\"ModelDataUrl\":\"s3://${MODEL_BUCKET}/$PIPELINE_PATH\"}],\"SupportedContentTypes\":[\"text/csv\"],\"SupportedResponseMIMETypes\":[\"text/csv\"]}"
aws sagemaker create-model-package \
  --model-package-group-name "${MODEL_PACKAGE_GROUP_NAME}" \
  --model-package-description "Inference Pipeline rev. ${SOURCE_HEADHASH}" \
  --inference-specification $INFERENCE_SPEC \
  --model-approval-status "PendingManualApproval" \
  --customer-metadata-properties revision="${SOURCE_HEADHASH}" \
  --region $AWS_REGION