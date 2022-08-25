import os

import sagemaker
from sagemaker import ScriptProcessor, ModelMetrics, MetricsSource, TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn import SKLearnProcessor, SKLearn
from sagemaker.workflow.condition_step import JsonGet, ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
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
    model_approval_status, processing_instance_count, processing_instance_type, training_instance_type, training_instance_count, hpo_tuner_instance_type = sagemaker_pipeline_parameters(data_bucket=default_bucket)
    # TODO: Sarah what are the following parameters? How do I set them dynamically? Shouldn't they go to the sagemaker_pipeline_parameters() method too?
    database = ParameterString(name="DataBase", default_value="customerone_mock_data_rl")
    table = ParameterString(name="AbaloneTable", default_value="master") # ??????
    filter = ParameterString(name="FilterRings", default_value="disabled")
    time_path = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    trigger_id = ParameterString(name="TriggerID", default_value="0000000000") #from codebuild - use CODEBUILD_BUILD_ID env variable parsed after ":" The CodeBuild ID of the build (for example, codebuild-demo-project:b1e6661e-e4f2-4156-9ab9-82a19EXAMPLE).
    nowgmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    execution_time = ParameterString(name="ExecutionTime", default_value=nowgmt)
    image_uri = "370702650160.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-cross-validation-pipeline"
    framework_version = "0.23-1"

    print(f"SARAH: base_job_prefix = {base_job_prefix}")
    print(f"SARAH: default_bucket={default_bucket}")
    print(f"SARAH: env_data={env_data}")
    print(f"SARAH: model_package_group_name={model_package_group_name}")
    print(f"SARAH: pipeline_name={pipeline_name}")
    print(f"SARAH: base_dir={base_dir}")
    print(f"SARAH: source_scripts_path={source_scripts_path}")
    print(f"SARAH: region={region}")

    # SARAH: base_job_prefix = customerone-dev-branch-p-lwkq81p5gxnk
    # SARAH: default_bucket=***
    # SARAH: env_data={'DomainArn': 'arn:aws:sagemaker:eu-north-1:***:domain/d-tdizim9qnor9', 'DomainId': 'd-tdizim9qnor9', 'DomainName': 'mlops-dev-eu-north-1-sagemaker-domain', 'HomeEfsFileSystemId': 'fs-03fc3d37f8623fea2', 'Status': 'InService', 'AuthMode': 'IAM', 'AppNetworkAccessType': 'VpcOnly', 'SubnetIds': ['subnet-0724be5e7071e7070', 'subnet-01def51ffe7467c71'], 'Url': 'https://d-tdizim9qnor9.studio.eu-north-1.sagemaker.aws', 'VpcId': 'vpc-0459a28f3637e285c', 'KmsKeyId': 'f4664542-0f2e-42ca-b51f-2bec0ad62278', 'ExecutionRole': 'arn:aws:iam::***:role/sm-mlops-env-EnvironmentIAM-SageMakerExecutionRole-14AU65MVMBUGO', 'SecurityGroups': ['***'], 'EnvironmentName': 'mlops', 'EnvironmentType': 'dev', 'DataBucketName': '***', 'ModelBucketName': '***', 'S3KmsKeyId': '***', 'EbsKmsKeyArn': '***', 'TrustedDefaultKinesisAccount': '', 'ProcessingRole': 'arn:aws:iam::***:role/sm-mlops-env-EnvironmentIAM-SageMakerExecutionRole-14AU65MVMBUGO', 'TrainingRole': 'arn:aws:iam::***:role/sm-mlops-env-EnvironmentIAM-SageMakerExecutionRole-14AU65MVMBUGO'}
    # SARAH: model_package_group_name=customerone-dev-branch-p-lwkq81p5gxnk
    # SARAH: pipeline_name]customerone-dev-branch-p-lwkq81p5gxnk-training
    # SARAH: base_dir=/codebuild/output/src374925051/src
    # SARAH: source_scripts_path=s3://***/lifecycle/max/customerone-dev-branch/79a4d75/input/source_scripts
    # SARAH: region=eu-north-1
    # SARAH: data_base_path=s3://***/lifecycle/60d/customerone-dev-branch/79a4d75/2022_08_24_12_09_07/p1033/output/training
    # SARAH: s3_bucket_base_path = s3://s3://***/lifecycle/60d/customerone-dev-branch/79a4d75/2022_08_24_12_09_07/p1033/output/training/lifecycle/30d/customerone-dev-branch/

    # configure network for encryption, network isolation and VPC configuration
    # Since the preprocessor job takes the data from S3, enable_network_isolation must be set to False
    # see https://github.com/aws/amazon-sagemaker-examples/issues/1689
    network_config = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=env_data["SecurityGroups"],
        subnets=env_data["SubnetIds"],
        encrypt_inter_container_traffic=True)
    print(f"Sarah: standard_model_pipeline > network_config: {network_config}") # <sagemaker.network.NetworkConfig object at 0x7f27da7ce100>
    data_base_path = "s3://{}/lifecycle/60d/{}/{}/{}/{}/output/training".format(env_data["DataBucketName"], project, revision, time_path, purpose)
    print(f"SARAH: data_base_path={data_base_path}")
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
        execution_time=execution_time,
        framework_version=framework_version,
    )

    model_name = "xsell_cust_voice_to_fixed"
    # Specify the model path where you want to save the models from training:
    model_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/training".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    evaluation_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/evaluation".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    step_model_selection, step_cv_train_hpo, sklearn_estimator = lightgbm_training_tasks(base_job_prefix=base_job_prefix,
                                                    env_data=env_data,
                                                    image_uri=image_uri,
                                                    network_config=network_config,
                                                    sagemaker_session=sagemaker_session,
                                                    step_process=step_process,
                                                    training_instance_type=training_instance_type,
                                                    training_instance_count=training_instance_count,
                                                    model_path=model_path,
                                                    data_base_path=data_base_path,
                                                    evaluation_path=evaluation_path,
                                                    hpo_tuner_instance_type=hpo_tuner_instance_type,
                                                    region=region,
                                                    framework_version=framework_version,
                                                    source_scripts_path=source_scripts_path,
                                                    )

    # processing step for evaluation
    # evaluation_path = "s3://{}/lifecycle/max/{}/{}/{}/{}/output/evaluation".format(env_data["ModelBucketName"], project, revision, model_name, time_path)
    evaluation_report, model_metrics, step_eval = evaluation_tasks(base_job_prefix=base_job_prefix,
                                                                   env_data=env_data,
                                                                   image_uri=image_uri,
                                                                   network_config=network_config,
                                                                   sagemaker_session=sagemaker_session,
                                                                   step_process=step_process,
                                                                   processing_instance_type=processing_instance_type,
                                                                   step_train=step_model_selection,
                                                                   source_scripts_path=source_scripts_path,
                                                                   evaluation_path=evaluation_path
                                                                   )

    postprocessing_script = "{}/postprocessing/postprocess.py".format(source_scripts_path)
    step_cond = model_register_tasks(evaluation_report,
                                     model_approval_status,
                                     model_metrics,
                                     model_package_group_name,
                                     network_config,
                                     step_eval,
                                     step_model_selection,
                                     sklearn_estimator,
                                     preprocessing_script,
                                     postprocessing_script,
                                     revision)
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
        steps=[step_process, step_model_selection, step_eval, step_cond],
        # steps=[step_process],
        sagemaker_session=sagemaker_session,
    )
    return pipeline


def sagemaker_pipeline_parameters(data_bucket):
    # TODO: Sarah: input argument is not used!
    # TODO: Sarah: Shouldn't we pass our choices? model_approval_status will always be pending?!
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    hpo_tuner_instance_type = ParameterString(name="HPOTunerScriptInstanceType", default_value="ml.t3.medium")

    return model_approval_status, processing_instance_count, processing_instance_type, training_instance_type, training_instance_count, hpo_tuner_instance_type


def preprocessing(base_job_prefix, env_data, network_config, processing_instance_count, processing_instance_type,
                  sagemaker_session, source_scripts_path, snapshot_path, training_path, validation_path, test_path,
                  database, table, filter, execution_time, framework_version):

    print("SARAH: standard_model_pipeline > preprocessing starts")
    preprocessing_script = "{}/preprocessing/preprocess.py".format(source_scripts_path)
    # ## 1- processing step for feature engineering Step
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-c1-xsell--preprocess_kfold",
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
            # ProcessingOutput(output_name="validation",
            #                  source="/opt/ml/processing/validation",
            #                  destination=validation_path
            #                  ),
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
    print("SARAH: standard_model_pipeline > preprocessing ends!")
    return step_process, preprocessing_script # TODO: SARAH: why should we return preprocessing_script??!??!


def lightgbm_training_tasks(base_job_prefix, env_data, image_uri, network_config, sagemaker_session, step_process,
                            training_instance_type, training_instance_count, model_path, data_base_path,
                            evaluation_path, hpo_tuner_instance_type, region, framework_version, source_scripts_path):
    k = ParameterInteger(name="KFold", default_value=3)
    max_jobs = ParameterInteger(name="MaxTrainingJobs", default_value=3)
    max_parallel_jobs = ParameterInteger(name="MaxParallelTrainingJobs", default_value=1)
    min_c = ParameterInteger(name="MinimumC", default_value=0)
    max_c = ParameterInteger(name="MaximumC", default_value=1)
    min_gamma = ParameterFloat(name="MinimumGamma", default_value=0.0001)
    max_gamma = ParameterFloat(name="MaximumGamma", default_value=0.001)
    gamma_scaling_type = ParameterString(name="GammaScalingType", default_value="Logarithmic")

    cross_validation_with_hpo_script = "{}/preprocessing/cross_validation_with_hpo.py".format(source_scripts_path)
    scikit_learn_iris_script = "{}/preprocessing/scikit_learn_iris.py".format(source_scripts_path)

    s3_bucket_base_path_jobinfo = f"{data_base_path}/jobinfo" # TODO: SARAH: is this correct????
    # s3_bucket_base_path = f"s3://{default_bucket_data.default_value}/{bucket_prefix_data.default_value}"
    # s3_bucket_base_path_train = f"{s3_bucket_base_path}train"
    # bucket_prefix_data = ParameterString(name="S3BucketPrefixData", default_value="lifecycle/30d/customerone-dev-branch/")
    # s3_bucket_base_path = f"{data_base_path}/{bucket_prefix_data.default_value}"
    # s3_bucket_base_path_train = f"{s3_bucket_base_path}train"
    # s3_bucket_base_path_test = f"{s3_bucket_base_path}test"
    # s3_bucket_base_path_output = f"{s3_bucket_base_path}/output"
    s3_bucket_base_path_train = f"{data_base_path}train"
    s3_bucket_base_path_test = f"{data_base_path}test"
    s3_bucket_base_path_output = f"{data_base_path}/output"

    # print(f"SARAH: lightgbm_training_tasks > s3_bucket_base_path = {s3_bucket_base_path}")
    # print(f"SARAH: lightgbm_training_tasks > bucket_prefix_data={bucket_prefix_data.default_value}")
    print(f"SARAH: lightgbm_training_tasks > s3_bucket_base_path_train={s3_bucket_base_path_train}")
    print(f"SARAH: lightgbm_training_tasks > s3_bucket_base_path_test={s3_bucket_base_path_test}")
    print(F"SARAH: lightgbm_training_tasks > s3_bucket_base_path_output={s3_bucket_base_path_output}")
    print(f"SARAH: lightgbm_training_tasks > s3_bucket_base_path_jobinfo = {s3_bucket_base_path_jobinfo}")
    print(f"SARAH: lightgbm_training_tasks > from xgbm train ={TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri, content_type='text/csv')}")
    # SARAH: lightgbm_training_tasks > s3_bucket_base_path = s3://***/lifecycle/60d/customerone-dev-branch/dc8e2f8/2022_08_24_12_32_10/p1033/output/training/lifecycle/30d/customerone-dev-branch/
    # SARAH: lightgbm_training_tasks > bucket_prefix_data=lifecycle/30d/customerone-dev-branch/
    # SARAH: lightgbm_training_tasks > s3_bucket_base_path_train=s3://***/lifecycle/60d/customerone-dev-branch/dc8e2f8/2022_08_24_12_32_10/p1033/output/training/lifecycle/30d/customerone-dev-branch/train
    # SARAH: lightgbm_training_tasks > s3_bucket_base_path_test=s3://***/lifecycle/60d/customerone-dev-branch/dc8e2f8/2022_08_24_12_32_10/p1033/output/training/lifecycle/30d/customerone-dev-branch/test
    # SARAH: lightgbm_training_tasks > s3_bucket_base_path_output=s3://***/lifecycle/60d/customerone-dev-branch/dc8e2f8/2022_08_24_12_32_10/p1033/output/training/lifecycle/30d/customerone-dev-branch//output
    # SARAH: lightgbm_training_tasks > s3_bucket_base_path_jobinfo = s3://***/lifecycle/60d/customerone-dev-branch/dc8e2f8/2022_08_24_12_32_10/p1033/output/training/jobinfo
    # SARAH: lightgbm_training_tasks > from xgbm train =<sagemaker.inputs.TrainingInput object at 0x7f15f905faf0>

    # ## 2- Cross Validation Model Training Step
    # In Cross Validation Model Training workflow, a script processor is used for orchestrating k training jobs in parallel, each of the k jobs is responsible for training a model using the specified split samples. Additionally, the script processor leverages Sagemaker HyperparameterTuner to optimize the hyper parameters and pass these values to perform k training jobs. The script processor monitors all training jobs. Once the jobs are complete, the script processor captures key metrics, including the training accuracy and the hyperparameters from the best training job, then uploads the results to the specified S3 bucket location to be used for model evaluation and model selection steps.
    #
    # The components involved in orchestrating the cross validation model training, hyperparameter optimizations and key metrics capture:
    #
    # * PropertyFile - EvaluationReport, contains the performance metrics from the HyperparameterTuner job, expressed in JSON format.
    # * PropertyFile JobInfo, contains information about the best training job and the corresponding hyperparameters used for training, expressed in JSON format.
    # * ScriptProcessor - A python script that orchestrates a hyperparameter tuning job for cross validation model trainings.

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    jobinfo = PropertyFile(name="JobInfo", output_name="jobinfo", path="jobinfo.json")

    script_tuner = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=hpo_tuner_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/KFoldCrossValidationHyperParameterTuner",
        role=env_data["TrainingRole"],
        sagemaker_session=sagemaker_session,
        # subnets=network_config.subnets,
        # security_group_ids=network_config.security_group_ids,
        # encrypt_inter_container_traffic=True,
        # enable_network_isolation=False,
        volume_kms_key=env_data["EbsKmsKeyArn"],
        output_kms_key=env_data["S3KmsKeyId"],
        network_config=network_config,
    )

    step_cv_train_hpo = ProcessingStep(
        name="HyperParameterTuningStep",
        processor=script_tuner,
        code=cross_validation_with_hpo_script,
        outputs=[
            ProcessingOutput(output_name="evaluation",
                             source="/opt/ml/processing/evaluation",
                             destination=evaluation_path), # s3_bucket_base_path_evaluation
            ProcessingOutput(output_name="jobinfo",
                             source="/opt/ml/processing/jobinfo",
                             destination=s3_bucket_base_path_jobinfo)
        ],
        job_arguments=["-k", str(k.expr),
                       "--image-uri", image_uri,
                       "--train", s3_bucket_base_path_train,
                       "--test", s3_bucket_base_path_test,
                       "--instance-type", training_instance_type,
                       "--instance-count", str(training_instance_count.expr),
                       "--output-path", s3_bucket_base_path_output,
                       "--max-jobs", str(max_jobs.expr),
                       "--max-parallel-jobs", str(max_parallel_jobs.expr),
                       "--min-c", str(min_c.expr),
                       "--max-c", str(max_c.expr),
                       "--min-gamma", str(min_gamma.expr),
                       "--max-gamma", str(max_gamma.expr),
                       "--gamma-scaling-type", str(gamma_scaling_type.expr),
                       "--region", str(region)],
        property_files=[evaluation_report],
        depends_on=['PreprocessStep'])

    # ## 3- Model Selection Step
    # Model selection is the final step in cross validation model training workflow. Based on the metrics and hyperparameters acquired from the cross validation steps orchestrated through ScriptProcessor,
    # a Training Step is defined to train a model with the same algorithm used in cross validation training, with all available training data. The model artifacts created from the training process will be used
    # for model registration, deployment and inferences.
    #
    # Components involved in the model selection step:
    #
    # * SKLearn Estimator - A Sagemaker Estimator used in training a final model.
    # * TrainingStep - Workflow step that triggers the model selection process.
    sklearn_estimator = SKLearn(scikit_learn_iris_script,
                                framework_version=framework_version,
                                instance_type=training_instance_type,
                                py_version='py3',
                                source_dir="code",
                                output_path=s3_bucket_base_path_output,
                                role=env_data['role'])

    step_model_selection = TrainingStep(
        name="ModelSelectionStep",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=f'{step_process.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]}/all',
                content_type="text/csv"
            ),
            "jobinfo": TrainingInput(
                s3_data=f"{s3_bucket_base_path_jobinfo}",
                content_type="application/json"
            )
        }
    )

    return step_model_selection, step_cv_train_hpo, sklearn_estimator


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
