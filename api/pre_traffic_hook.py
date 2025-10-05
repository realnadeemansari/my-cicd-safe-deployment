import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sm = boto3.client("sagemaker")
cd = boto3.client("codedeploy")

def lambda_handler(event, context):
    logger.debug("event %s", json.dumps(event))
    endpoint_name = os.environ["ENDPOINT_NAME"]
    logger.info("pre traffic for endpoint %s", endpoint_name)
    error_message = None

    try:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            error_message = f"SageMaker endpoint: {endpoint_name} status {status} not InService"
        else:
            endpoint_config_name = response["EndpointConfigName"]
            response = sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            if "DataCaptureConfig" in response and response["DataCaptureConfig"]["EnableCapture"]:
                logger.info(f"Data capture enabled for endpoint config {endpoint_config_name}")
            else:
                error_message = "SageMaker data capture not enabled for endpoint config"
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(f"Error checking endpoint {error_message}")

    try:
        if error_message != None:
            logger.info(f"put codepipeline failed: {error_message}")
            response = cd.put_lifecycle_event_hook_execution_status(
                deploymentId=event["DeploymentId"],
                lifecycleEventHookExecutionId=event["LifecycleEventHookExecutionId"],
                status="Failed"
            )
            return {"statusCode": 400, "message": error_message}
        else:
            logger.info("put codepipeline success")
            response = cd.put_lifecycle_event_hook_execution_status(
                deploymentId=event["DeploymentId"],
                lifecycleEventHookExecutionId=event["LifecycleEventHookExecutionId"],
                status="Succeeded",
            )
            return {
                "statusCode": 200
            }
    except ClientError as e:
        logger.error("Unexpected code pipeline error")
        logger.error(e)
        return {"statusCode": 500, "message": e.response["Error"]["Message"]}