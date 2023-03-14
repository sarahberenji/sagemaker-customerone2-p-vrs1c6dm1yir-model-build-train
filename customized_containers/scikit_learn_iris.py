import subprocess
from botocore.config import Config
import boto3

def auth_codeartifact(mlops_domain='cirrus-ml-ds-domain', domain_owner='813736554012', repository='cirrus-ml-ds-shared-repo', region='eu-north-1'):
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
    print(f"installing package = {package}")
    #installs package with pip install
    subprocess.run(["pip", "install", package], capture_output=True)

def list_files(startpath):
    print("***************** List_files:")
    for root, dirs, files in os.walk(startpath):
        print(f"list_files > root={root}, dirs={dirs}, files={files}")
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

auth_codeartifact()
install("lightgbm")

# import sys
# sys.path.append("/usr/local/lib/python3.7/site-packages")
import os
import sys
print(f"SARAH: sys.path = {sys.path}")
# sys.path = ['/opt/ml/code', '/opt/ml/code', '/miniconda3/bin', '/miniconda3/lib/python37.zip',
# '/miniconda3/lib/python3.7', '/miniconda3/lib/python3.7/lib-dynload', '/miniconda3/lib/python3.7/site-packages']
print(f"SARAH: PYTHONPATH = {os.environ.get('PYTHONPATH', '').split(os.pathsep)}")
# PYTHONPATH = ['/opt/ml/code', '/miniconda3/bin', '/miniconda3/lib/python37.zip', '/miniconda3/lib/python3.7',
# '/miniconda3/lib/python3.7/lib-dynload', '/miniconda3/lib/python3.7/site-packages']

# sys.path.append('/usr/local/lib/python3.7/site-packages')
# print(f"SARAH: after append, sys.path = {sys.path}")

print(f"SARAH: pip list =  {os.system('pip list')}")

import argparse
import pandas as pd
# import os
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
from numpy import genfromtxt
from joblib import load, dump
import logging
import json

logging.basicConfig(level=logging.INFO)

base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation"


def train(train=None, test=None):
    """Trains a model using the specified algorithm with given parameters.

       Args:
          train : location on the filesystem for training dataset
          test: location on the filesystem for test dataset

       Returns:
          trained model object
    """
    # print("SARAH: scikit_learn_iris.py > train() with SVM")
    # Take the set of files and read them all into a single pandas dataframe
    train_files = [os.path.join(train, file) for file in os.listdir(train)]
    if test:
        test_files = [os.path.join(test, file) for file in os.listdir(test)]
    if len(train_files) == 0 or (test and len(test_files)) == 0:
        raise ValueError((f'There are no files in {train}.\n' +
                          'This usually indicates that the channel train was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.'))

    X_train_df = pd.read_csv(f'{train}/train_x.csv')
    y_train_df = pd.read_csv(f'{train}/train_y.csv')

    X_train_df = X_train_df.set_index(X_train_df.columns[0])
    y_train_df = y_train_df.set_index(y_train_df.columns[0])

    X_train_df = X_train_df.fillna(0)
    y_train_df = y_train_df.fillna(0)
    print(f"SARAH: scikit_learn_iris.py > train() > Removing nulls")
    print(f"SARAH: scikit_learn_iris.py > train() > X_train_df.shape={X_train_df.shape}")
    print(f"SARAH: scikit_learn_iris.py > train() > y_train_df.shape={y_train_df.shape}")
    print(f"SARAH: scikit_learn_iris.py > train() > X_train_df={X_train_df}")
    print(f"SARAH: scikit_learn_iris.py > train() > y_train_df={y_train_df}")


    # Now use scikit-learn's decision tree classifier to train the model.
    if "SM_CHANNEL_JOBINFO" in os.environ:
        jobinfo_path = os.environ.get('SM_CHANNEL_JOBINFO')
        print(f"SARAH: scikit_learn_iris.py > train() > jobinfo_path={jobinfo_path}")
        with open(f"{jobinfo_path}/jobinfo.json", "r") as f:
            jobinfo = json.load(f)
            hyperparams = jobinfo['hyperparams']
            print(f"SARAH: scikit_learn_iris.py > train() > hyperparams={hyperparams}")
            # clf = svm.SVC(kernel='linear',
            #               C=float(hyperparams['estimator_learning_rate']),
            #               gamma=float(hyperparams['over_sampler_sampling_strategy']),
            #               verbose=1).fit(X_train_df.values, y_train_df.values)
            lgbm_model = LGBMClassifier(learning_rate=hyperparams['estimator_learning_rate'],
                                        )
            lgbm_model.fit(X_train_df.values, y_train_df.values.ravel())
    else:
        # clf = svm.SVC(kernel='linear',
        #               C=args.estimator_learning_rate,
        #               gamma=args.over_sampler_sampling_strategy,
        #               verbose=1).fit(X_train_df.values, y_train_df.values)
        print(f"SARAH: scikit_learn_iris.py > train() > else lgbm_modellllllll, args.estimator_learning_rate={args.estimator_learning_rate}")
        lgbm_model = LGBMClassifier(boosting_type='gbdt',
                                    learning_rate=args.estimator_learning_rate,
                                    # =args.over_sampler_sampling_strategy,
                                    # preprocessing_categorical_encoder_min_samples_leaf,
                                    # preprocessing_categorical_encoder_smoothing,
                                    )
        lgbm_model.fit(X_train_df.values, y_train_df.values.ravel())

        # LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        #                importance_type='split', learning_rate=0.1, max_depth=-1,
        #                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        #                n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
        #                random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        #                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)


    return lgbm_model


def evaluate(test=None, model=None):
    """Evaluates the performance for the given model.

       Args:
          test: location on the filesystem for test dataset
    """
    print("SARAH: scikit_learn_iris.py > evaluate()")
    if test:
        # X_test = genfromtxt(f'{test}/test_x.csv', delimiter=',')
        # y_test = genfromtxt(f'{test}/test_y.csv', delimiter=',')
        X_test_df = pd.read_csv(f'{test}/test_x.csv')
        y_test_df = pd.read_csv(f'{test}/test_y.csv')

        X_test_df = X_test_df.set_index(X_test_df.columns[0])
        y_test_df = y_test_df.set_index(y_test_df.columns[0])

        X_test_df = X_test_df.fillna(0)
        y_test_df = y_test_df.fillna(0)
        print(f"SARAH: scikit_learn_iris.py > evaluate() > Removing nulls")
        print(f"SARAH: scikit_learn_iris.py > evaluate() > X_test_df.shape={X_test_df.shape}")
        print(f"SARAH: scikit_learn_iris.py > evaluate() > y_test_df.shape={y_test_df.shape}")
        print(f"SARAH: scikit_learn_iris.py > evaluate() > X_test_df={X_test_df}")
        print(f"SARAH: scikit_learn_iris.py > evaluate() > y_test_df={y_test_df}")
        accuracy_score = model.score(X_test_df.values, y_test_df.values)
        logging.info(f'model test score:{accuracy_score};')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    print("SARAH: scikit_learn_iris.py > __main__")
    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    # C1 parameters
    parser.add_argument('--preprocessing_categorical_encoder_min_samples_leaf', type=int)
    parser.add_argument('--preprocessing_categorical_encoder_smoothing', type=float)
    parser.add_argument('--over_sampler_sampling_strategy', type=float)  # TODO: float or string????
    parser.add_argument('--estimator_learning_rate', type=float)

    args = parser.parse_args()
    print(f"SARAH: scikit_learn_iris.py > __main__ > args={args}")

    model = train(train=args.train, test=args.test)

    evaluate(test=args.test, model=model)
    dump(model, os.path.join(args.model_dir, "model.joblib"))

    model_dir = '/opt/ml/model'
    output_data_dir = '/opt/ml/output/data'
    list_files(model_dir)
    list_files(output_data_dir)


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    print("SARAH: scikit_learn_iris.py > model_fn()")
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf
