import argparse
import json
import os
import sys
import time

import boto3

import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.model_monitor.dataset_format import DatasetFormat

import stepfunctions
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

def create_experiment_step(create_experiment_function_name):
    create_experiment_step = steps.compute.LambdaStep(
        "Create Experiment",
        parameters={
            "FunctionName": create_experiment_function_name,
            "Payload": {
                "ExperimentName.$": "$.ExperimentName",
                "TrialName.$": "$.TrialName",
            }
        },
        result_path="$.CreateTrialResults"
    )
    return create_experiment_step

def create_baseline_step(input_data, execution_input, region, role):
    # Define the environment
    dataset_format = DatasetFormat.csv()
    env = {
        "dataset_format": json.dumps(dataset_format),
        "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
        "output_path": "/opt/ml/processing/output",
        "publish_cloudwatch_metrics": "Disabled"
    }
    # Define the inputs and outputs
    inputs = [
        ProcessingInput(
            source=input_data["BaselineUri"],
            destination="/opt/ml/processing/input/baseline_dataset_input",
            input_name="baseline_dataset_input"
        )
    ]
    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=execution_input["BaselineOutputUri"],
            output_name="monitoring_output"
        )
    ]
    # Get the default model monitor container
    model_monitor_container_uri = retrieve(
        region=region, framework="model-monitor", version="latest"
    )
    # Create the processor
    monitor_analyzer = Processor(
        image_uri=model_monitor_container_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1800,
        env=env
    )
    # Create the processing step
    baseline_step = steps.sagemaker.ProcessingStep(
        "Baseline Job",
        processor=monitor_analyzer,
        job_name=execution_input["BaselineJobName"],
        inputs=inputs,
        outputs=outputs,
        experiment_config={
            "ExperimentName": execution_input["ExperimentName"],
            "TrialName": execution_input["TrialName"],
            "TrialComponentDisplayName": "Baseline",
        },
        tags={
            "GitBranch": execution_input["GitBranch"],
            "GitCommitHash": execution_input["GitCommitHash"],
            "DataVersionId": execution_input["DataVersionId"],
        },
    )
    # Add the catch
    baseline_step.add_catch(
        steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=steps.states.Fail(
                "Baseline failed", cause="SageMakerBaselineJobFailed"
            )
        )
    )
    return baseline_step

def get_training_image(region):
    return sagemaker.image_uris.retrieve(region=region, framework="xgboost", version="latest")

def create_training_step(
        image_uri,
        hyperparameters,
        input_data,
        output_data,
        execution_input,
        query_training_function_name,
        region,
        role
):
    # Create the estimater
    xgb = sagemaker.estimator.Estimator(
        image_uri,
        role,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        output_path=output_data["ModelOutputUri"],
    )

    # Set the hyperparameters overriding with any defaults
    hp = {
        "max_depth": "9",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "300",
        "subsample": "0.8",
        "objective": "reg:linear",
        "early_stopping_rounds": "10",
        "num_round": "100"
    }
    xgb.set_hyperparameters(**{**hp, **hyperparameters})
    # Specify the data source
    s3_input_train = sagemaker.inputs.TrainingInput(
        s3_data=input_data["TrainingUri"], content_type="csv"
    )
    s3_input_val = sagemaker.inputs.TrainingInput(
        s3_data=input_data["ValidationUri"], content_type="csv"
    )
    data = {
        "train": s3_input_train, "validation": s3_input_val
    }
    # Create the training step
    training_step = steps.TrainingStep(
        "Training Job",
        estimator=xgb,
        data=data,
        job_name=execution_input["TrainingJobName"],
        experiment_config={
            "ExperimentName": execution_input["ExperimentName"],
            "TrialName": execution_input["TrialName"],
            "TrialComponentDisplayName": "Training"
        },
        tags={
            "GitBranch": execution_input["GitBranch"],
            "GitCommitHash": execution_input["GitCommitHash"],
            "DataVersionId": execution_input["DataVersionId"]
        },
        result_path="$.TrainingResults"
    )

    # Add the catch
    training_step.add_catch(
        stepfunctions.steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=stepfunctions.steps.states.Fail(
                "Training failed", cause="SageMakerTrainingJobFailed"
            ),
        )
    )

    # Must follow the training test
    model_step = steps.sagemaker.ModelStep(
        "Save Model",
        input_path="$.TrainingResults",
        model=training_step.get_expected_model(),
        model_name=execution_input["TrainingJobName"],
        result_path="$.ModelStepResults"
    )
    # Query the training step
    training_query_step = steps.compute.LambdaStep(
        "Query Training Results",
        parameters={
            "FunctionName": query_training_function_name,
            "Payload": {"TrainingJobName.$": "$.TrainingJobName"},
        },
        result_path="$.QueryTrainingResults",
    )

    check_accuracy_fail_step = steps.states.Fail(
        "Model Error Too Low", comment="RMSE accuracy higher than threshold"
    )
    check_accuracy_succeed_step = steps.states.Succeed("Model Error Acceptable")

    threshold_rule = steps.choice_rule.ChoiceRule.NumericLessThan(
        variable=training_query_step.output()["QueryTrainingResults"]["Payload"]["results"]["TrainingMetrics"][0]["Value"],
        value=10
    )

    check_accuracy_step = steps.states.Choice("RMSE < 10")
    check_accuracy_step.add_choice(rule=threshold_rule, next_step=check_accuracy_succeed_step)
    check_accuracy_step.default_choice(next_step=check_accuracy_fail_step)

    return steps.states.Chain([training_step, model_step, training_query_step, check_accuracy_step])

def create_graph(create_experiment_step, baseline_step, training_step):
    sagemaker_jobs = steps.states.Parallel("Sagemaker Jobs")
    sagemaker_jobs.add_branch(baseline_step)
    sagemaker_jobs.add_branch(training_step)

    sagemaker_jobs.add_catch(
        stepfunctions.steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=stepfunctions.steps.states.Fail(
                "Sagemaker Jobs failed", cause="SageMakerJobsFailed"
            ),
        )
    )
    return steps.states.Chain([create_experiment_step, sagemaker_jobs])

def get_dev_config(model_name, job_id, role, image_uri, kms_key_id, sagemaker_project_id):
    return {
        "Parameters": {
            "ImageRepoUri": image_uri,
            "ModelName": model_name,
            "TrainJobId": job_id,
            "DeployRoleArn": role,
            "ModelVariant": "dev",
            "KmsKeyId": kms_key_id
        },
        "Tags": {
            "mlops:model-name": model_name,
            "mlops:stage": "dev",
            "SageMakerProjectId": sagemaker_project_id
        },
    }

def get_prd_config(
        model_name, job_id, role, image_uri, kms_key_id, notification_arn, sagemaker_project_id
):
    dev_config = get_dev_config(model_name, job_id, role, image_uri, kms_key_id, sagemaker_project_id)
    prod_params = {
        "ModelVariant": "prd",
        "ScheduleMetricName": "feature_baseline_drift_total_amount",
        "ScheduleMetricThreshold": str("0.20"),
        "NotificationArn": notification_arn
    }
    prod_tags = {"mlops:stage": "prd", "SageMakerProjectId": sagemaker_project_id}
    return {
        "Parameters": dict(dev_config["Parameters"], **prod_params),
        "Tags": dict(dev_config["Tags"], **prod_tags),
    }

def get_pipeline_execution_id(pipeline_name, codebuild_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_state(name=pipeline_name)
    for stage in response["stageStates"]:
        for action in stage["actionStates"]:
            # Return the matching stage with the same external id
            if (
                "latestExecution" in action
                and "externalExecutionId" in action["latestExecution"]
                and action["latestExecution"]["externalExecutionId"] == codebuild_id
            ):
                return stage["latestExecution"]["pipelineExecutionId"]
            
def get_pipeline_revisions(pipeline_name, execution_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_execution(
        pipelineName=pipeline_name, pipelineExecutionId=execution_id
    )
    return dict(
        (r["name"], r["revisionId"]) for r in response["pipelineExecution"]["artifactRevisions"]
    )

def main(
        git_branch,
        codebuild_id,
        pipeline_name,
        model_name,
        deploy_role,
        sagemaker_role,
        sagemaker_bucket,
        data_dir,
        output_dir,
        ecr_dir,
        kms_key_id,
        workflow_role_arn,
        notification_arn,
        sagemaker_project_id,
        tags
):
    # Define the function name
    create_experiment_function_name = "mlops-create-experiment"
    query_training_function_name = "mlops-query-training"

    # Get the region
    region = boto3.Session().region_name
    print("region: {}".format(region))

    if ecr_dir:
        # Load the image uri and input data config
        with open(os.path.join(ecr_dir, "imageDetail.json"), "r") as f:
            image_uri = json.load(f)["ImageURI"]
    else:
        # Get the managed image uri for current region
        image_uri = get_training_image(region=region)
    print(f"image uri: {image_uri}")
    with open(os.path.join(data_dir, "inputData.json"), "r") as f:
        input_data = json.load(f)
        print(f"training uri: {input_data['TrainingUri']}")
        print(f"validation uri: {input_data['ValidationUri']}")
        print(f"baseline uri: {input_data['BaselineUri']}")

    # Get the job id and source revisions
    job_id = get_pipeline_execution_id(pipeline_name, codebuild_id)
    revisions = get_pipeline_revisions(pipeline_name, job_id)
    git_commit_id = revisions["ModelSourceOutput"]
    data_version_id = revisions["DataSourceOutput"]
    print("job id: {}".format(job_id))
    print(f"git commit: {git_commit_id}")
    print(f"data version: {data_version_id}")

    # Set the output data
    output_data = {
        "ModelOutputUri": f"s3://{sagemaker_bucket}/{model_name}",
        "BaselineOutputUri": f"s3://{sagemaker_bucket}/{model_name}/monitoring/baseline/{model_name}-pbl-{job_id}"
    }
    print(f"model output uri: {output_data['ModelOutputUri']}")

    # Pass these into the training method
    hyperparameters = {}
    if os.path.exists(os.path.join(data_dir, "hyperparameters.json")):
        with open(os.path.join(data_dir, "hyperparameters.json"), "r") as f:
            hyperparameters = json.load(f)
            for i in hyperparameters:
                hyperparameters[i] = str(hyperparameters[i])
    
    # Define the step functions execution input schema
    execution_input = ExecutionInput(
        schema={
            "GitBranch": str,
            "GitCommitHash": str,
            "DataVersionId": str,
            "ExperimentName": str,
            "TrialName": str,
            "BaselineJobName": str,
            "BaselineOutputUri": str,
            "TrainingJobName": str
        }
    )

    # Create experiment step
    experiment_step = create_experiment_step(create_experiment_function_name)
    baseline_step = create_baseline_step(input_data, execution_input, region, sagemaker_role)
    training_step = create_training_step(
        image_uri,
        hyperparameters,
        input_data,
        output_data,
        execution_input,
        query_training_function_name,
        region,
        sagemaker_role
    )
    workflow_definition = create_graph(experiment_step, baseline_step, training_step)

    # Create the workflow as the model name
    workflow = Workflow(model_name, workflow_definition, workflow_role_arn)
    print(f"Creating workflow: {model_name}-{sagemaker_project_id}")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Write the workflow graph to json
    with open(os.path.join(output_dir, "workflow-graph.json"), "w") as f:
        f.write(workflow.definition.to_json(pretty=True))
    
    # Write the workflow graph to yml
    with open(os.path.join(output_dir, "workflow-graph.yml"), "w") as f:
        f.write(workflow.get_cloudformation_template())

    # Write the workflow inputs to file
    with open(os.path.join(output_dir, "workflow-input.json"), "w") as f:
        workflow_inputs = {
            "ExperimentName": f"{model_name}",
            "TrialName": f"{model_name}-{job_id}",
            "GitBranch": git_branch,
            "GitCommitHash": git_commit_id,
            "DataVersionId": data_version_id,
            "BaselineJobName": f"{model_name}-pbl-{job_id}",
            "BaselineOutputUri": output_data["BaselineOutputUri"],
            "TrainingJobName": f"{model_name}-{job_id}"
        }
        json.dump(workflow_inputs, f)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-model-dev.json"), "w") as f:
        config = get_dev_config(
            model_name,
            job_id,
            deploy_role,
            image_uri,
            kms_key_id,
            sagemaker_project_id
        )
        json.dump(config, f)
    with open(os.path.join(output_dir, "deploy-model-prd.json"), "w") as f:
        config = get_prd_config(
            model_name,
            job_id,
            deploy_role,
            image_uri,
            kms_key_id,
            notification_arn,
            sagemaker_project_id
        )
        json.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="sagemaker_role",
        type=str,
        help="The role arn for the pipeline service execution role."
    )
    parser.add_argument(
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    parser.add_argument("--codebuild-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ecr-dir", required=False)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--deploy-role", required=True)
    parser.add_argument("--sagemaker-role", required=True)
    parser.add_argument("--sagemaker-bucket", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--workflow-role-arn", required=True)
    parser.add_argument("--notification-arn", required=True)
    parser.add_argument("--sagemaker-project-id", required=True)
    args = vars(parser.parse_args())
    print(f"args: {args}")
    main(**args)