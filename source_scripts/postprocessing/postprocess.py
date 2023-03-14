"""Postprocess"""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys

import uuid
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

boto3.setup_default_session(region_name="eu-north-1")

if __name__ == "__main__":

    logger.info("SARAH >>> Starting postprocessing.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--triggerid", type=str, required=True)
    parser.add_argument("--inferenceoutput", type=str, required=True)
    # parser.add_argument("--sourceaccount", type=str, required=True)
    args = parser.parse_args()
    
    print(f"sarah: args={args}")

    STREAM_NAME = "engineering-ml-integration"
    
    ssm = boto3.client('ssm', region_name="eu-north-1")
    # source_account = args.sourceaccount
    env_type = ssm.get_parameter(Name='EnvType')['Parameter']['Value']
    team_name=ssm.get_parameter(Name='TeamName')['Parameter']['Value']
    STREAM_NAME = f"{team_name}-ml-integration"
    ASSUMED_ROLE = f"arn:aws:iam::{source_account}:role/{team_name}-{env_type}-ml-integration-role"

    sts_client = boto3.client('sts')
    assumed_role_object = sts_client.assume_role(
        RoleArn=ASSUMED_ROLE,
        RoleSessionName="ml-kinesis-access"
    )
    print(assumed_role_object)
    
    credentials = assumed_role_object['Credentials']
    session = boto3.session.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name="eu-north-1"
    )

    kinesis = session.client('kinesis')

    data = {}
    triggerid = str(uuid.uuid4())
    data['trigger_id'] = args.triggerid
    data['keys'] = [f"{args.inferenceoutput}/inference-data.csv.out"]

    print(data)

    kinesis.put_record(
                StreamName=STREAM_NAME,
                Data=json.dumps(data),
                PartitionKey=triggerid)

    logger.info(">>> End postprocessing.")


