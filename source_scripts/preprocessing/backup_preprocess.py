"""Feature engineers the abalone dataset."""
import argparse
import logging
import pathlib
import subprocess
import sys
from botocore.config import Config
import boto3


# To easily add packages at runtime from codeartifact, you can use this method. A more graceful way is to install packages in
# a container and use that container as the processing container.
# @todo adjust hardcoded values into fetching from ssm parameters.
def auth_codeartifact(mlops_domain='cirrus-ml-ds-domain', domain_owner='813736554012',
                      repository='cirrus-ml-ds-shared-repo',
                      region='eu-north-1'):
    # fetches temporary credentials with boto3 from codeartifact, creates the index_url for the pip config
    # and finally uses the index_url (url with token included) to update the global pip config
    # when pip install is run, this means that pip install will utilize codeartifact instead of trying to reach public pypi
    boto3_config = Config(
        region_name='eu-north-1',
        signature_version='v4',
        retries={
            'max_attempts': 10,
            'mode': 'standard'
        }
    )
    client = boto3.client('codeartifact', config=boto3_config)
    codeartifact_token = client.get_authorization_token(
        domain=mlops_domain,
        domainOwner=domain_owner,
        durationSeconds=10000
    )
    codeartifact_token["authorizationToken"]
    pip_index_url = f'https://aws:{codeartifact_token["authorizationToken"]}@{mlops_domain}-{domain_owner}.d.codeartifact.{region}.amazonaws.com/pypi/{repository}/simple/'
    subprocess.run(["pip", "config", "set", "global.index-url", pip_index_url],
                   capture_output=True)


def install(package):
    # installs package with pip install
    subprocess.run(["pip", "install", package],
                   capture_output=True)


auth_codeartifact()
install("awswrangler")

import awswrangler as wr

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


def preprocess_abalone(abalone_dataset):
    logger.info("Reading downloaded data.")
    df = abalone_dataset

    logger.info("Defining transformers.")
    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    logger.info("Applying transforms.")
    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X_pre), axis=1)
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    return np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])


if __name__ == "__main__":

    logger.info("Starting preprocessing.")
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # abalone_dataset = f"{base_dir}/input/athena-input.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--executiontime", type=str, required=False)
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--filter", type=str, required=True)
    args = parser.parse_args()

    boto3.setup_default_session(region_name="eu-north-1")
    ssm = boto3.client('ssm', region_name="eu-north-1")
    env_type = ssm.get_parameter(Name='EnvType')['Parameter']['Value']

    if args.filter == "disabled":
        query_filter = ""
    else:
        query_filter = f'WHERE rings > {args.filter}'
    abalone_dataset = wr.athena.read_sql_query(
        f'SELECT * FROM "{args.database}"."{args.table}" {query_filter};',
        database=args.database,
        workgroup=f"{env_type}-athena-workgroup")

    if args.context == "training":

        train, validation, test = preprocess_abalone(abalone_dataset)
        logger.info("Writing out datasets to %s.", base_dir)
        pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
        pd.DataFrame(validation).to_csv(
            f"{base_dir}/validation/validation.csv", header=False, index=False
        )
        pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    elif args.context == "inference":
        print("Some mock processing for now")

        print(f"execution time (UTC): {args.executiontime}")

        train, validation, test = preprocess_abalone(abalone_dataset)
        test_df = pd.DataFrame(test)
        first_column = test_df.columns[0]
        inference_data = test_df.drop([first_column], axis=1)
        pd.DataFrame(inference_data).to_csv(f"{base_dir}/inference-test/inference-data.csv", header=False, index=False)
    else:
        logger.info("missing context, allowed values are training or inference")
        sys.exit("missing supported context type")



