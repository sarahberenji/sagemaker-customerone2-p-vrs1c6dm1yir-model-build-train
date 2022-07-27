# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""Helper method to extract features and model parameters for training and inference."""
from typing import Any, Dict, Iterable, List, Mapping, Union

import numpy as np
import pandas as pd

# from customerone_lib.common.pkgs.feature_selection.helpers import get_sublist
# from customerone_lib.common.pkgs.model_development.data_dict.generator import DataDict


def extract_features(
    features: Union[pd.DataFrame, List[str], Mapping[str, Any], DataDict],
    data: pd.DataFrame,
    iteration_col_name: str,
) -> Dict[int, List[str]]:
    """Extracts features which are used for model fitting and inference.

    If a list, the same features will be used for all iterations, if a dataframe,
    it must conform to the format used by the `select_features` node in this connector.

    This procedure allows using different features for different iterations, which is
    useful e.g. if we want to do k fold validation of the entire data science pipeline.

    Args:
        features: Features to use, either a list or a dataframe.
        data: Data on which these features will be used, including the iteration id
        iteration_col_name: Name of the iteration id column.

    Returns:
        A dictionary in which the key is the iteration id and the values are a list of
            features to use in that iteration.

    Raises:
        TypeError: If the data provided is neither a list nor a dataframe.
    """
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(features, Mapping):
        feature_params = features
        inclusion_list_columns = feature_params.get("inclusion_list")
        inclusion_regex = feature_params.get("inclusion_regex")
        block_list_columns = feature_params.get("block_list", [])

        features = get_sublist(
            super_list=list(data.columns),
            sub_regex=inclusion_regex,
            sub_list=inclusion_list_columns,
            block_list=block_list_columns,
        )

    if isinstance(features, pd.DataFrame):
        extracted_features = _extract_features_from_selection_report(
            features, iteration_col_name
        )
    elif isinstance(features, list):
        extracted_features = _extract_features_from_list(
            features, data, iteration_col_name
        )
    elif isinstance(features, DataDict):
        extracted_features = _extract_features_from_list(
            features.get_features(), data, iteration_col_name
        )
    else:
        raise TypeError(f"Expected a pandas dataframe or a list, got {type(features)}")

    return extracted_features


def _extract_features_from_selection_report(
    features: pd.DataFrame, iteration_col_name: str
) -> Dict[int, List[str]]:
    """Returns a dictionary {iteration_id: [selected_features].

    Args:
        features: The feature selection report
        iteration_col_name: The name of the iteration_id column

    Returns:
        A dictionary with key iteration id and value the list of selected features.

    input
    ::

        +------------+-------+--------+----------------+-------------------+
        |iteration_id|feature|selected|kept_until_stage|dropped_by_selector|
        +------------+-------+--------+----------------+-------------------+
        |0           |A      |False  |0                |VarianceThreshold  |
        |0           |B      |True   |1                |VifSelector        |
        |0           |C      |True   |2                |SelectFromModel    |
        |1           |B      |False  |0                |VarianceThreshold  |
        |1           |C      |False  |0                |VarianceThreshold  |
        |1           |A      |True   |2                |SelectFromModel    |
        +------------+-------+--------+----------------+-------------------+

    output:
    ::
        {0: [B, C], 1: [C, A]}
    """
    selected = features[features["selected"]]
    return dict(selected.groupby(iteration_col_name)["feature"].apply(list))


def _extract_features_from_list(
    features: List[str], data: pd.DataFrame, iteration_col_name: str
) -> Dict[int, List[str]]:
    unique_iterations = data[iteration_col_name].unique()

    return {iter: features for iter in unique_iterations}


def extract_model_parameters(
    parameters: Union[Dict[int, Dict], Dict[str, Any]],
    data: pd.DataFrame,
    iteration_col_name: str,
) -> Dict[int, Dict[str, Any]]:
    """Casts model parameters into a common [iteration_id: model_parameters] format."""
    if np.all([isinstance(key, str) for key in parameters.keys()]):
        return_params = _extract_features_from_single_parameters(
            parameters, data, iteration_col_name
        )
    elif np.all([isinstance(key, int) for key in parameters.keys()]) and np.all(
        [isinstance(key, dict) for key in parameters.values()]
    ):
        return_params = parameters
    else:
        raise TypeError(
            "Expected either a dictionary with string keys or a dictionary with integer"
            + " keys and dictionary values."
        )

    return return_params


def _extract_features_from_single_parameters(
    parameters, data: pd.DataFrame, iteration_col_name: str
) -> Dict[int, Dict[str, Any]]:
    return {iteration: parameters for iteration in data[iteration_col_name].unique()}


def cast_to_list(items: Union[str, Iterable]) -> List:
    """Cast input to list, if input is a single string or another iterable."""
    return [items] if isinstance(items, str) else list(items)


def check_available_in_data(features: List, data: pd.DataFrame) -> None:
    """Checks whether features specified in the config exist in the data.

    Raises an error if they do not exist.
    """
    available_features = data.columns
    missing = set(features) - set(available_features)
    if len(missing) > 0:
        raise ValueError(
            f"The columns {missing} were specified in model_params but are missing "
            + "in the data. Please make sure these columns are available and correctly "
            + "spelled or remove them from the feature list."
        )
