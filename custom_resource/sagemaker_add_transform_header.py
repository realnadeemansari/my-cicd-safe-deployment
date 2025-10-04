import boto3
import logging
import json
from urllib.parse import urlparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.resource("s3")

def lambda_handler(event, context):
    if "TransformOutputUri" in event:
        s3_uri = event["TransformOutputUri"]
    else:
        raise KeyError(f"TransformOutputUri not found for event: {json.dumps(event)}")
    if "FileName" in event:
        file_name = event["FileName"]
    else:
        raise KeyError(f"FileName not found for event: {json.dumps(event)}")
    if "Header" in event:
        header = event["Header"]
    else:
        raise KeyError(f"Header not found for event: {json.dumps(event)}")
    
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:]

    obj = s3.Object(bucket_name, f"{prefix}/{file_name}.out")
    body = obj.get()["Body"].read().decode("utf-8")

    new_obj = s3.Object(bucket_name, "{}/{}".format(prefix, file_name))
    body = header + "\n" + body
    return new_obj.put(
        Body=body.encode("utf-8"), ContentType="text/csv", Metadata={"header": "true"}
    )