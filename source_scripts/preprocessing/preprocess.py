"""Feature engineers the abalone dataset."""
import argparse
import logging
import pathlib
import subprocess
import sys
from botocore.config import Config
import boto3

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# To easily add packages at runtime from codeartifact, you can use this method. A more graceful way is to install packages in
# a container and use that container as the processing container.
# @todo adjust hardcoded values into fetching from ssm parameters.
def auth_codeartifact(mlops_domain='cirrus-ml-ds-domain', domain_owner='813736554012', repository='cirrus-ml-ds-shared-repo', region='eu-north-1'):
    # fetches temporary credentials with boto3 from codeartifact, creates the index_url for the pip config
    # and finally uses the index_url (url with token included) to update the global pip config
    # when pip install is run, this means that pip install will utilize codeartifact instead of trying to reach public pypi
    boto3_config = Config(
        region_name = 'eu-north-1',
        signature_version = 'v4',
        retries = {
            'max_attempts': 10,
            'mode': 'standard'
        }
    )
    client = boto3.client('codeartifact',config=boto3_config)
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
    print(f"Sarah: install package = {package}")
    #installs package with pip install
    subprocess.run(["pip", "install", package], capture_output=True)


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        print(f"list_files > root={root}, dirs={dirs}, files={files}")
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

auth_codeartifact()
# TODO: Sarah, why here?! what about the requirements.txt file? Where is it used?
print("Sarah: installing pip packages")
install("awswrangler")
install("category-encoders")
install("imbalanced-learn")

import awswrangler as wr
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"test: BASE_DIR={BASE_DIR}")
try:
    print("test:##### OS CODEBUILD ENV")
    print(os.environ["CODEBUILD_SRC_DIR"])
    list_files(os.environ["CODEBUILD_SRC_DIR"])
except:
    print("test: no codebuild env")
print(f"test:#### Current Working Dir is '{os.getcwd()}'")
# list_files(os.getcwd())
print("test: list files under /opt/ml/processin, is it the same as BASE_DIR??")
print("test: #### BASE_DIR list_files")
list_files(BASE_DIR)
print("test: #### list_files under /opt/ml/processing")
list_files('/opt/ml/processing')

from source_scripts.preprocessing.utils_determine_feature_type import determine_feature_data_types
from .utils_split import make_splits, split_data, make_subsplit
from .utils_reading_data import to_pandas

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":

    logger.info("Starting preprocessing.")
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    #abalone_dataset = f"{base_dir}/input/athena-input.parquet"

    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--executiontime", type=str, required=False)
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--filter", type=str, required=True)
    args = parser.parse_args()

    boto3.setup_default_session(region_name="eu-north-1")
    ssm = boto3.client('ssm', region_name="eu-north-1")
    env_type=ssm.get_parameter(Name='EnvType')['Parameter']['Value']

    if args.filter == "disabled":
        query_filter = ""
    else:
        query_filter = f'WHERE rings > {args.filter}'

    print(f"\n\nSarah SELECT * FROM \"{args.database}\".\"{args.table}\" {query_filter};\', database=args.database, workgroup=f\"{env_type}-athena-workgroup \n\n")
    xsell_dataset = wr.athena.read_sql_query(
        f'SELECT * FROM "{args.database}"."{args.table}" {query_filter};',
        database=args.database,
        workgroup=f"{env_type}-athena-workgroup")

    print(xsell_dataset.head())
    print("*****************\n\n")

    target_col_name = 'tgt_xsell_cust_voice_to_fixed'

    spine_params_determine_feature_data_types = {'keys': ['customer_id'],
                                                 'date_column': 'current_dt',
                                                 'product_holdings_filter': {'product_category': 'fixedbroadband'},
                                                 'is_deepsell': 'N'}

    target_params_determine_feature_data_types = {'target_variable_column': 'tgt_xsell_cust_voice_to_fixed'
        , 'lead_time_window': '1d'
        , 'target_window': '45d'
        , 'product_activation_filter': {'product_sub_category': 'voice'}
        , 'campaign_keys': ['customer_id']
        , 'campaign_filter': {'campaign_name': ["C-452-O-06 Korsförsäljning Telia Life 2.0 - Sälja mobilt"
            , "b2c_cross-sell_pp_crossSellPpToBb"
            , "b2c_cross-sell_pp_crossSellPpToBb_oldContent"
            , "b2c_cross-sell_pp_crossSellPpToBb_REM1"
            , "C-700-O-03 Cross-sell Mobile to Broadband customers (TM only)"
            , "C-700-O-01 Cross-sell Mobile to Broadband customers (A)"
            , "C-752-O GEOF 2021 - X-sell PP"
            , "C-652-O Black Friday Erbjudande 2019  Mobilt till BB-kund - activity 1"
            , "C-700-O-02 Cross-sell Mobile to Broadband customers (B)"
            , "b2c_cross-sell_Pp_PpToBb_default"
            , "C-752-O GEOF 2021 - Xsell PP"
            , "b2c_cross-sell_Pp_PpToBb_simOnly"
            , "b2c_cross-sell_Pp_PpToBb_samS215G"
            , "b2c_cross-sell_Pp_PpToBb_iphone12Mini"
            , "b2c_cross-sell_Pp_PpToBb_iphoneSE"
            , "b2c_cross-sell_Pp_PpToBb_iphone12"
            , "b2c_cross-sell_Pp_PpToBb_samS20FE5G"
            , "b2c_cross-sell_Pp_PpToBb_sonyXp10lll"
            , "b2c_cross-sell_Pp_PpToBb_default_short_8pm"
            , "b2c_cross-sell_Pp_PpToBb_default_8pm"]
            , 'customer_actioned_flg_column': {'Email': 'actioned_ind'}}}

    # Sarah: is it ok to use xsell_dataset for creating features list?
    feature_dict = determine_feature_data_types(xsell_dataset,
                                                spine_params_determine_feature_data_types,
                                                target_params_determine_feature_data_types)
    print(f"len(feature_dict)={len(feature_dict)}, \nlen(feature_dict['numeric'])={len(feature_dict['numeric'])}, \nlen(feature_dict['categorical'])={len(feature_dict['categorical'])}")

    features = feature_dict['categorical'] + feature_dict['numeric']

    if args.context == "training":
        # skipped sample_training_df()
        # skipped drop_invalid_features()
        # skipped filter_features()

        # skipped to_pandas() and just filtered the null target rows:
        df_training = xsell_dataset[xsell_dataset['tgt_xsell_cust_voice_to_fixed'].notnull()]
        df_training_idx = df_training.reset_index(drop=True)
        print(f"{df_training.shape}, {df_training_idx.shape}")

        # Splitting data
        split_train_test = make_splits(df_training_idx, target_col_name)
        print(f"{df_training.shape}, {df_training_idx.shape}, {split_train_test.shape}")
        print(f"split_train_test['split'].value_counts(dropna=False) = {split_train_test['split'].value_counts(dropna=False)}")
        print(f"df_training_idx['split'].value_counts(dropna=False) = {df_training_idx['split'].value_counts(dropna=False)}")
        print(f"split_train_test['iteration_id'].value_counts(dropna=False) = {split_train_test['iteration_id'].value_counts(dropna=False)}")

        splitter_args = {'test_size': 0.1, 'random_state': 42}
        subsplit_params = {"iteration_col_name": 'iteration_id', "split_col_name": "split", 'source_split': 'TRAIN',
                           'target_splits': ['TRAIN', 'CAL']}

        split_train_test_cal = make_subsplit(split_train_test, subsplit_params, splitter_args, target_col_name)
        print(f"split_train_test_cal['split'].value_counts(dropna=False) = {split_train_test_cal['split'].value_counts(dropna=False)}")

        train = split_train_test_cal[split_train_test_cal['split'] == 'TRAIN']
        test = split_train_test_cal[split_train_test_cal['split'] == 'TEST']
        validation = split_train_test_cal[split_train_test_cal['split'] == 'CAL']

        logger.info("Writing out datasets to %s.", base_dir)
        # TODO: sarah, do we have to store the data as csv files?
        pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
        pd.DataFrame(validation).to_csv(
            f"{base_dir}/validation/validation.csv", header=False, index=False
        )
        pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    elif args.context == "inference":
        print("Some mock processing for now")
        
        print(f"execution time (UTC): {args.executiontime}")

        pandas_params = {'include_target': False}
        spine_params = {'keys': ['customer_id'],
                        'date_column': 'current_dt',
                        'product_holdings_filter': {'exact_match': {'product_category': 'fixedbroadband'}},
                        'is_deepsell': 'N'}
        ref_date = '2021-06-30'
        data_dictionary = None

        # inference_master_table = inference_data
        inference_data= to_pandas(xsell_dataset, pandas_params, spine_params, target_col_name, data_dictionary, features, ref_date)
        print(f"inference_master_table inference_data.shape = {inference_data.shape}")

        pd.DataFrame(inference_data).to_csv(f"{base_dir}/inference-test/inference-data.csv", header=False, index=False)
    else:
        logger.info("missing context, allowed values are training or inference")
        sys.exit("missing supported context type")


# feature_columns_names = [
#     "sex",
#     "length",
#     "diameter",
#     "height",
#     "whole_weight",
#     "shucked_weight",
#     "viscera_weight",
#     "shell_weight",
# ]

