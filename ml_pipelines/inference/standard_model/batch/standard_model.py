from sagemaker.network import NetworkConfig
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model
from datetime import datetime
from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep

from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
)
from datetime import datetime
from sagemaker.workflow.functions import Join
import time

def standard_model_pipeline(base_job_prefix, default_bucket, env_data, model_package_group_name, pipeline_name, region,
                            sagemaker_session, base_dir, source_scripts_path, model_metadata, project = "standard_model", revision = "none", purpose = "p1033" ):
    #time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    epoch_time = int(time.time())

    #data_bucket + prefix = s3://${DATA_BUCKET_ENV}/lifecycle/60d/${SAGEMAKER_PROJECT_NAME}/${PIPELINE_NAME}/${TRIGGER_ID}/p1033"
    data_bucket = ParameterString(name="DataBucket", default_value=env_data["DataBucketName"])
    purpose_param = ParameterString(name="Purpose", default_value=purpose)
    trigger_id = ParameterString(name="TriggerID", default_value="0000000000") #from codebuild - use CODEBUILD_BUILD_ID env variable parsed after ":" The CodeBuild ID of the build (for example, codebuild-demo-project:b1e6661e-e4f2-4156-9ab9-82a19EXAMPLE).
    prefix_path = Join(on='/', values=["lifecycle/60d", project, pipeline_name, trigger_id, purpose_param])
    data_base_path = Join(on='/', values=['s3:/', data_bucket, prefix_path])
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge" )
    snapshot_data = Join(on='/', values=[data_base_path, 'data-snapshot'])
    batch_data = Join(on='/', values=[data_base_path, 'model-input'])
    inference_output = Join(on='/', values=[data_base_path, 'results'])
    subnet1 = ParameterString(name="Subnet1", default_value="{}".format(env_data["SubnetIds"][0]))
    subnet2 = ParameterString(name="Subnet2", default_value="{}".format(env_data["SubnetIds"][1]))
    securitygroup = ParameterString(name="SecurityGroup", default_value="{}".format(env_data["SecurityGroups"][0]))
    volume_kms_key = ParameterString(name="EbsKmsKeyArn", default_value="{}".format(env_data["EbsKmsKeyArn"]))
    output_kms_key = ParameterString(name="S3KmsKeyId", default_value="{}".format(env_data["S3KmsKeyId"]))
    processing_role = ParameterString(name="ProcessingRole", default_value=env_data["ProcessingRole"])


    source_account = ParameterString(name="SourceAccount")

    database = ParameterString(name="DataBase", default_value="ml-test-datasets_rl" )
    table = ParameterString(name="AbaloneTable", default_value="ml_abalone" )
    filter = ParameterString(name="FilterRings", default_value="disabled")

    nowgmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    execution_time = ParameterString(name="ExecutionTime", default_value=nowgmt)

    # configure network for encryption, network isolation and VPC configuration
    # Since the preprocessor job takes the data from S3, enable_network_isolation must be set to False
    # see https://github.com/aws/amazon-sagemaker-examples/issues/1689
    network_config = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=[securitygroup],
        subnets=[subnet1, subnet2],
        encrypt_inter_container_traffic=True)

    vpc_config = {
        "Subnets": network_config.subnets,
        "SecurityGroupIds": network_config.security_group_ids
    }

    step_process = preprocessing(base_job_prefix=base_job_prefix,
                                 network_config=network_config,
                                 processing_instance_count=processing_instance_count,
                                 processing_instance_type=processing_instance_type,
                                 sagemaker_session=sagemaker_session,
                                 preprocess_script_path=model_metadata["CustomerMetadataProperties"]["preprocess"],
                                 batch_data=batch_data,
                                 database=database,
                                 table=table,
                                 filter=filter,
                                 volume_kms_key=volume_kms_key,
                                 output_kms_key=output_kms_key,
                                 processing_role=processing_role,
                                 execution_time=execution_time
    )

    step_create_model = create_model_tasks(
        sagemaker_session=sagemaker_session,
        vpc_config=vpc_config,
        model_metadata=model_metadata,
        processing_role=processing_role,
    )

    step_inference = inference_tasks(
        model_name=step_create_model.properties.ModelName,
        batch_data=batch_data,
        output_data_path=inference_output,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key,
        instance_type="ml.m5.large",
    )

    post_process = postprocessing(base_job_prefix=base_job_prefix,
                                 network_config=network_config,
                                 processing_instance_count=processing_instance_count,
                                 processing_instance_type=processing_instance_type,
                                 sagemaker_session=sagemaker_session,
                                 postprocess_script_path=model_metadata["CustomerMetadataProperties"]["postprocess"],
                                 volume_kms_key=volume_kms_key,
                                 output_kms_key=output_kms_key,
                                 processing_role=processing_role,
                                 trigger_id=trigger_id,
                                 inference_output=inference_output,
                                source_account=source_account
    )

    step_inference.add_depends_on([step_process])


    post_process.add_depends_on([step_inference])

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            subnet1,
            subnet2,
            securitygroup,
            processing_instance_count,
            processing_instance_type,
            volume_kms_key,
            output_kms_key,
            processing_role,
            data_bucket,
            purpose_param,
            trigger_id,
            source_account,
            execution_time,
            database,
            table,
            filter
        ],
        steps=[step_process, step_create_model,step_inference, post_process],
        sagemaker_session=sagemaker_session,
    )
    return pipeline




def preprocessing(base_job_prefix,
                  network_config,
                  processing_instance_count,
                  processing_instance_type,
                  sagemaker_session,
                  preprocess_script_path,
                  batch_data,
                  volume_kms_key,
                  output_kms_key,
                  database,
                  table,
                  filter,
                  processing_role,
                  execution_time
                  ):

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=processing_role,
        network_config=network_config,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key
    )

    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="inference",
                             source="/opt/ml/processing/inference-test/",
                             destination=batch_data
                             ),
        ],
        code=preprocess_script_path,
        job_arguments=[
            "--context", "inference",
            "--executiontime", execution_time,
            "--database", database,
            "--table", table,
            "--filter", filter
        ],
    )


    return step_process

def create_model_tasks(sagemaker_session,
                       model_metadata, processing_role, vpc_config,
                       instance_type="ml.m5.large",
                       accelerator_type="ml.eia1.medium"):

    from sagemaker.inputs import CreateModelInput
    from sagemaker.workflow.steps import CreateModelStep
    image_uri = model_metadata["InferenceSpecification"]["Containers"][0]["Image"]
    model_data = model_metadata["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        sagemaker_session=sagemaker_session,
        role=processing_role,
        vpc_config=vpc_config
    )

    inputs = CreateModelInput(
        instance_type=instance_type,
        accelerator_type=accelerator_type,
    )
    step_create_model = CreateModelStep(
        name="AbaloneCreateModel",
        model=model,
        inputs=inputs,
    )

    return step_create_model

def inference_tasks(
                    model_name,
                    output_data_path,
                    batch_data,
                    volume_kms_key,
                    output_kms_key,
                    instance_type="ml.m5.large"
    ):

    from sagemaker.inputs import CreateModelInput
    from sagemaker.workflow.steps import CreateModelStep
    transformer = Transformer(
        model_name=model_name,
        instance_type=instance_type,
        instance_count=1,
        output_path=output_data_path,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key
    )

    step_inference = TransformStep(
        name="AbaloneTransform",
        transformer=transformer,
        inputs=TransformInput(data=batch_data, content_type="text/csv")
    )

    return step_inference

def postprocessing(base_job_prefix,
                   network_config,
                  processing_instance_count,
                  processing_instance_type,
                  sagemaker_session,
                  postprocess_script_path,
                  volume_kms_key,
                  output_kms_key,
                  processing_role,
                  trigger_id,
                  inference_output,
                   source_account
                  ):

    # processing step for post processing after inference
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-postprocess",
        sagemaker_session=sagemaker_session,
        role=processing_role,
        network_config=network_config,
        volume_kms_key=volume_kms_key,
        output_kms_key=output_kms_key
    )

    post_process = ProcessingStep(
        name="PostprocessAbaloneData",
        processor=sklearn_processor,
        code=postprocess_script_path,
        job_arguments=["--context", "postprocess",  "--triggerid", trigger_id, "--inferenceoutput", inference_output, "--sourceaccount", source_account])


    return post_process