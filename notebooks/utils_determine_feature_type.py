from typing import Any, Callable, Dict, Iterable, List, Optional, Union 

def to_list(item: Union[Any, List[Any]]) -> List[Any]:
    """
    Wraps an item to a list or returns the existing list

    Args:
        item (Union[Any, List[Any]]): a list of a single object

    Returns:
        item wrapped in a list, if item is not a list. Otherwise the
        original list is returned
    """

    return [item] if not isinstance(item, list) else item


def determine_feature_data_types(data, spine_params, target_params):
# def determine_feature_data_types(data: pd.DataFrame, spine_params: Dict[str, Any], target_params: Dict[str, Any],) -> Dict:
    """
    determine the feature data types and split them up into numerical, datetime 
    and categorical features

    Args:
        data: input dataframe
        spine_params: dictionary of parameters:
            keys (List[str]): key columns that defines a row in the master table
        target_params: dictionary of parameters:
            target_variable_column (str): the resulting target variable column

    Returns:
        dictionary containing the keys 'numeric', 'datetime' and 'categorical'. 
        Each key contains a list of all the features of the corresponding data type
    """
    
    # spine_params = {'keys': ['customer_id'], 'date_column': 'current_dt', 'product_holdings_filter': {'{'exact_match': {'product_category': 'fixedbroadband'}}'}, 'is_deepsell': 'N'}
    # {'target_variable_column': 'tgt_xsell_cust_voice_to_fixed', 'lead_time_window': '1d', 'target_window': '45d', 'product_activation_filter': {'exact_match': {...}}, 'campaign_keys': ['customer_id'], 'campaign_filter': {'exact_match': {...}, 'customer_actioned_flg_column': {...}}}
    
    none_features = (  #['customer_id', 'current_dt', 'tgt_xsell_cust_voice_to_fixed']
        to_list(spine_params["keys"]) +  
        to_list(spine_params["date_column"]) +
        to_list(target_params["target_variable_column"])
    )
#     print(f"none_features = {none_features}\n")
    feature_cols =  set(data.columns) - set(none_features)
#     print(f"feature_cols={feature_cols}")

    features = {}
    features["numeric"] = data[feature_cols].select_dtypes("number").columns.to_list()
    features["datetime"] = data[feature_cols].select_dtypes("datetime").columns.to_list()
    features["categorical"] = (
        data[feature_cols].select_dtypes("category").columns.to_list() +
        data[feature_cols].select_dtypes("object").columns.to_list() +
        data[feature_cols].select_dtypes("bool").columns.to_list() +
        [c for c in  features["numeric"] if any([match in c for match  in ["_cd", "_code", "_flg", "_flag"]])]
    )
    features["numeric"] = list(set(features["numeric"]) - set(features["categorical"]))

    return features