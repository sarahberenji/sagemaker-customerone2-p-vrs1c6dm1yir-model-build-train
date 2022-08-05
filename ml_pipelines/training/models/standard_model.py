import os

import sagemaker
from sagemaker import ScriptProcessor, ModelMetrics, MetricsSource, TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.condition_step import JsonGet, ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
)
from datetime import datetime
import time


def standard_model_pipeline(base_job_prefix, default_bucket, env_data, model_package_group_name, pipeline_name,
                            region, sagemaker_session, base_dir, source_scripts_path, project="standard_model",
                            revision="none", purpose="p1033"):
    # parameters for pipeline execution
    print("Sarah: Start of standard_model_pipeline()")
    model_approval_status, processing_instance_count, processing_instance_type, training_instance_type = sagemaker_pipeline_parameters(data_bucket=default_bucket)
    # TODO: Sarah what are the following parameters? How do I set them dynamically? Shouldn't they go to the sagemaker_pipeline_parameters() method too?
    database = ParameterString(name="DataBase", default_value="customerone_mock_data_rl")
    table = ParameterString(name="AbaloneTable", default_value="master") # ??????
    filter = ParameterString(name="FilterRings", default_value="disabled")
    time_path = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    trigger_id = ParameterString(name="TriggerID", default_value="0000000000") #from codebuild - use CODEBUILD_BUILD_ID env variable parsed after ":" The CodeBuild ID of the build (for example, codebuild-demo-project:b1e6661e-e4f2-4156-9ab9-82a19EXAMPLE).
    nowgmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    execution_time = ParameterString(name="ExecutionTime", default_value=nowgmt)

    # configure network for encryption, network isolation and VPC configuration
    # Since the preprocessor job takes the data from S3, enable_network_isolation must be set to False
    # see https://github.com/aws/amazon-sagemaker-examples/issues/1689
    network_config = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=env_data["SecurityGroups"],
        subnets=env_data["SubnetIds"],
        encrypt_inter_container_traffic=True)
    print(f"Sarah: standard_model_pipeline > network_config: {network_config}")
    data_base_path="s3://{}/lifecycle/60d/{}/{}/{}/{}/output/training".format(env_data["DataBucketName"], project, revision, time_path, purpose)
    step_process, preprocessing_script = preprocessing(
        base_job_prefix=base_job_prefix,
        env_data=env_data,
        network_config=network_config,
        processing_instance_count=processing_instance_count,
        processing_instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
        source_scripts_path=source_scripts_path,
        snapshot_path="{}/data-snapshot/".format(data_base_path),
        training_path="{}/processed/training".format(data_base_path),
        validation_path="{}/processed/validation".format(data_base_path),
        test_path="{}/processed/test".format(data_base_path),
        database=database,
        table=table,
        filter=filter,
        execution_time=execution_time
    )
    print("Sarah: step_process is ready")
    # # training step for generating model artifacts (Specify the training container image URI)
    # image_uri = sagemaker.image_uris.retrieve(
    #     framework="xgboost",
    #     region=region,
    #     version="1.0-1",
    #     py_version="py3",
    #     instance_type=training_instance_type,
    # )

    # model_name = "abalone"
    # Specify the model path where you want to save the models from training:
    # model_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/training".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    # step_train, xgb_train = training_tasks(base_job_prefix=base_job_prefix,
    #                                        env_data=env_data,
    #                                        image_uri=image_uri,
    #                                        network_config=network_config,
    #                                        sagemaker_session=sagemaker_session,
    #                                        step_process=step_process,
    #                                        training_instance_type=training_instance_type,
    #                                        model_path=model_path
    #                                        )

    train_model_id, train_model_version, train_scope = "lightgbm-classification-model", "*", "training"
    # Retrieve the docker image
    image_uri = sagemaker.image_uris.retrieve(
        region=region,
        framework="lightgbm",
        # model_id=train_model_id,
        # model_version=train_model_version,
        # image_scope=train_scope,
        py_version="py3",
        instance_type=training_instance_type,
    )
    print(f"Sarah: image_uri is ready {image_uri}")
    # Retrieve the training script
    train_source_uri = sagemaker.script_uris.retrieve(model_id=train_model_id,
                                                      model_version=train_model_version,
                                                      script_scope=train_scope)
    print(f"Sarah: train_source_uri is ready {train_source_uri}")
    # Retrieve the pre-trained model tarball to further fine-tune
    train_model_uri = sagemaker.model_uris.retrieve(model_id=train_model_id,
                                                    model_version=train_model_version,
                                                    model_scope=train_scope)
    print(f"Sarah: train_model_uri is ready {train_model_uri}")
    model_name = "xsell_cust_voice_to_fixed"
    # Specify the model path where you want to save the models from training:
    model_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/training".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    step_train, lgbm_train = training_tasks_lgbm(base_job_prefix=base_job_prefix,
                                                env_data=env_data,
                                                image_uri=image_uri,
                                                source_uri=train_source_uri,
                                                model_uri=train_model_uri,
                                                model_id=train_model_id,
                                                model_version=train_model_version,
                                                network_config=network_config,
                                                sagemaker_session=sagemaker_session,
                                                step_process=step_process,
                                                training_instance_type=training_instance_type,
                                                model_path=model_path)

    # processing step for evaluation
    evaluation_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/evaluation".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    evaluation_report, model_metrics, step_eval = evaluation_tasks(base_job_prefix=base_job_prefix,
                                                                   env_data=env_data,
                                                                   image_uri=image_uri,
                                                                   network_config=network_config,
                                                                   sagemaker_session=sagemaker_session,
                                                                   step_process=step_process,
                                                                   processing_instance_type=processing_instance_type,
                                                                   step_train=step_train,
                                                                   source_scripts_path=source_scripts_path,
                                                                   evaluation_path=evaluation_path
                                                                   )

    postprocessing_script="{}/postprocessing/postprocess.py".format(source_scripts_path)
    # step_cond = model_register_tasks(evaluation_report,
    #                                  model_approval_status,
    #                                  model_metrics,
    #                                  model_package_group_name,
    #                                  network_config,
    #                                  step_eval,
    #                                  step_train,
    #                                  xgb_train,
    #                                  preprocessing_script,
    #                                  postprocessing_script,
    #                                  revision)
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            trigger_id,
            execution_time,
            database,
            table,
            filter
        ],
        # steps=[step_process, step_train, step_eval, step_cond],
        steps=[step_process, step_train, step_eval, ],
        # steps=[step_process],
        sagemaker_session=sagemaker_session,
    )
    return pipeline


def sagemaker_pipeline_parameters(data_bucket):
    # TODO: Sarah: input argument is not used!
    # TODO: Sarah: Shouldn't we pass our choices? model_approval_status will always be pending?!
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    return model_approval_status, processing_instance_count, processing_instance_type, training_instance_type


def preprocessing(base_job_prefix, env_data, network_config, processing_instance_count, processing_instance_type,
                  sagemaker_session, source_scripts_path, snapshot_path, training_path, validation_path, test_path,
                  database, table, filter, execution_time):

    preprocessing_script = "{}/preprocessing/preprocess.py".format(source_scripts_path)

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-c1-xsell--preprocess",
        sagemaker_session=sagemaker_session,
        role=env_data["ProcessingRole"],
        network_config=network_config,
        volume_kms_key=env_data["EbsKmsKeyArn"],
        output_kms_key=env_data["S3KmsKeyId"]
    )

    step_process = ProcessingStep(
        name="PreprocessC1XsellData",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=f'{source_scripts_path}/preprocessing/utils/',
                                destination="/opt/ml/processing/input/code/utils/")],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=training_path
            ),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation",
                             destination=validation_path
                             ),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test",
                             destination=test_path
                             )
        ],
        code=preprocessing_script,
        job_arguments=[
            "--context", "training",
            "--executiontime", execution_time,
            "--database", database,
            "--table", table,
            "--filter", filter
        ],
    )

    return step_process, preprocessing_script


def training_tasks(base_job_prefix, env_data, image_uri, network_config, sagemaker_session, step_process, training_instance_type, model_path):
    print("Sarah: standard_model_pipeline > Start of training_tasks()")
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=sagemaker_session,
        role=env_data["TrainingRole"],
        subnets=network_config.subnets,
        security_group_ids=network_config.security_group_ids,
        encrypt_inter_container_traffic=True,
        enable_network_isolation=False,
        volume_kms_key=env_data["EbsKmsKeyArn"],
        output_kms_key=env_data["S3KmsKeyId"]
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name="TrainAbaloneModel",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv",),
            "validation": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,content_type="text/csv",),
        },
    )
    print("Sarah: standard_model_pipeline > End of training_tasks()")
    return step_train, xgb_train


def training_tasks_lgbm(base_job_prefix, env_data, image_uri, source_uri, model_uri, model_id,train_model_version,
                        network_config, sagemaker_session, step_process, training_instance_type, model_path):
    print("Srah: Start of training_tasks_lgbm()")
    # Create SageMaker Estimator instance
    lightgbm_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=sagemaker_session,
        role=env_data["TrainingRole"],
        subnets=network_config.subnets,
        security_group_ids=network_config.security_group_ids,
        encrypt_inter_container_traffic=True,
        enable_network_isolation=False,
        volume_kms_key=env_data["EbsKmsKeyArn"],
        output_kms_key=env_data["S3KmsKeyId"],
        source_dir=source_uri,
        model_uri=model_uri,
        # entry_point="transfer_learning.py",
        # max_run=360000,
        # hyperparameters=hyperparameters,
    )
    # # Retrieve the default hyper-parameters for fine-tuning the model
    # hyperparameters = hyperparameters.retrieve_default(
    #     model_id=model_id, model_version=train_model_version
    # )
    # # [Optional] Override default hyperparameters with custom values
    # hyperparameters[
    #     "num_boost_round"
    # ] = "500"  # The same hyperparameter is named as "iterations" for CatBoost
    # print(hyperparameters)
    print("Sarah: lightgbm_train estimator is ready")
    lightgbm_train.set_hyperparameters(
        num_boost_roun=500,
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    print("Sarah: lightgbm_train set hyperparametr is done")
    step_train = TrainingStep(
        name="TrainXsellModel",
        estimator=lightgbm_train,
        inputs={
            "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv",),
            "validation": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,content_type="text/csv",),
        },
    )

    print("Sarah: End of training_tasks_lgbm")
    return step_train, lightgbm_train


def evaluation_tasks(base_job_prefix, env_data, image_uri, network_config, processing_instance_type, sagemaker_session,
                     step_process, step_train, source_scripts_path, evaluation_path):
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=sagemaker_session,
        role=env_data["ProcessingRole"],
        network_config=network_config,
        volume_kms_key=env_data["EbsKmsKeyArn"],
        output_kms_key=env_data["S3KmsKeyId"]
    )
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",
                             destination= evaluation_path
                             ),
        ],
        code="{}/evaluate/evaluate.py".format(source_scripts_path),
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(evaluation_path),
            content_type="application/json"
        )
    )
    return evaluation_report, model_metrics, step_eval


def model_register_tasks(evaluation_report, model_approval_status, model_metrics, model_package_group_name,
                         network_config, step_eval, step_train, xgb_train, preprocessing_script, postprocessing_script, revision):
    """
    There is a bug in RegisterModel implementation
    The RegisterModel step is implemented in the SDK as two steps, a _RepackModelStep and a _RegisterModelStep.
    The _RepackModelStep runs a SKLearn training step in order to repack the model.tar.gz to include any custom inference code in the archive.
    The _RegisterModelStep then registers the repacked model.

    The problem is that the _RepackModelStep does not propagate VPC configuration from the Estimator object:
    https://github.com/aws/sagemaker-python-sdk/blob/cdb633b3ab02398c3b77f5ecd2c03cdf41049c78/src/sagemaker/workflow/_utils.py#L88

    This cause the AccessDenied exception because repacker cannot access S3 bucket (all access which is not via VPC endpoint is bloked by the bucket policy)

    The issue is opened against SageMaker python SDK: https://github.com/aws/sagemaker-python-sdk/issues/2302
    """
    vpc_config = {
        "Subnets": network_config.subnets,
        "SecurityGroupIds": network_config.security_group_ids
    }
    step_register = RegisterModel(
        name="RegisterAbaloneModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        vpc_config_override=vpc_config,
        customer_metadata_properties={
            "preprocess" : preprocessing_script,
            "postprocess" : postprocessing_script,
            "git_revision" : revision
        }
    )
    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    return step_cond
