"""Performs hyperparameter tuning.

Supported hyperparameter tuning classes:
    - GridSearchCV (default)
    - RandomizedSearchCV
    - BayesSearchCV

`tuning_params` expects the following key value pairs:
    `tuner`: Path to tuning class.
    `tuner_args`: Arguments required for the chosen hyperparameter tuning class
    `param_grid`: A dictionary of lists with possible values for each parameter.
    `scoring`: (Optional) Name of a scorer or the path to an sklearn scorer class.
        see: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    `cv`: (Optional) Integer for k-fold validation or the path to a splitter
    `cv_args`: (Optional) Instantiation arguments for a splitter passed to `cv`
    `splits_to_fit`: (Optional) Which splits to fit, default TRAIN.
        The tuner will do its own cross validation splitting within the specified
        split.

`splitting_params` expects the following key value pairs:
    `iteration_col_name`: Name of column in which the iteration id is to be saved.
    `split_col_name`: Name of column in which the TRAIN/TEST split is saved.

Examples:
    The example configuration below shows how to use custom cross validation,
    and how to fix some values in the search while keeping others flexible:

    ::
        predictive_modeling.growth.tuning_params:
          tuner: sklearn.model_selection.RandomizedSearchCV
          tuner_args:
            param_distributions:
              booster: ["gbtree"]
              max_depth: [3, 5, 10]
              learning_rate: scipy.stats.uniform(loc=0.1, scale=0.4)
          cv: sklearn.model_selection.GroupShuffleSplit
          cv_args:
            n_splits: 2
          cv_split_group_col_name: observation_dt

    The configuration tunes an estimator, using RandomizedSearchCV as a
    tuner. The `booster` parameter is fixed, since there is only one possible value
    in the search space. `max_depth` is a categorical choice between 3 possible
    values. The `learning_rate` is sampled from a uniform distribution.

    The tuner uses a custom cross validation method `GroupShuffleSplit` here.
    It produces 2 splits, and uses a column named `observation_dt` as a grouping
    column.

Args:
    data: A dataframe containing features, targets, fold and split information
    estimator: estimator object that is assumed to implement the scikit-learn 
        estimator interface
    tuning_params: Describes the tuner, its arguments and its search space
    features: The features to fit the model on. If a list, the same
        features will be used for all iterations, if a dataframe, it must conform
        to the format used by the `select_features` node in this connector.
    splitting_params: Describe where fold and splitting information are located
    target_col_name: Name of target column

Returns:
    The best hyperparameters and the best score for each iteration.
"""

    
from typing import Any, Callable, Dict, Iterable, List, Optional, Union 
from sklearn.model_selection import RandomizedSearchCV
from .test_train_splitter import OutOfTimeSplitter
import copy
from typing import Union
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import numpy as np
# from helper import extract_features
# from skopt.space.space import check_dimension

def tune_parameters(data, estimator, tuning_params, features, splitting_params, target_col_name):
# def tune_parameters(data: pd.DataFrame, estimator: BaseEstimator, tuning_params: Dict[str, Any], features: Union[pd.DataFrame, List[str], DataDict], splitting_params: Dict[str, Any], target_col_name: str,) -> Dict[int, Any]:
#     estimator = Ridge()


    iteration_col_name = splitting_params.get("iteration_col_name", "iteration_id")
    split_col_name = splitting_params.get("split_col_name", "split")
#     print(f"iteration_col_name={iteration_col_name}, data[iteration_col_name].unique = {data[iteration_col_name].unique()}")
    
    cv = OutOfTimeSplitter(tuning_params.get('cv_args').get('test_size'), 
                           tuning_params.get('cv_args').get('date_col'), 
                           tuning_params.get('cv_args').get('n_splits'),
                           tuning_params.get('cv_args').get('random_state'), 
                           tuning_params.get('cv_args').get('verbose'))
#     print(f"\n\n cv={cv}")
#     print(f"test_size={tuning_params.get('cv_args').get('test_size')}, date_col={tuning_params.get('cv_args').get('date_col')}, n_splits={tuning_params.get('cv_args').get('n_splits')}, random_state={tuning_params.get('cv_args').get('random_state')}, verbose={tuning_params.get('cv_args').get('verbose')}")
    
    space = tuning_params.get('tuner_args').get('param_distributions')
#     print(f"\n\n space={space}")

#     tuner = RandomizedSearchCV(estimator, space, 
#                                 n_iter=tuning_params.get('tuner_args').get('n_iter'), 
#                                 scoring=tuning_params.get('tuner_args').get('scoring'), 
#                                 cv=cv)
#     print(f"\n\n serach={tuner}")
#     print(f"n_iter={tuning_params.get('tuner_args').get('n_iter')}, scoring={tuning_params.get('tuner_args').get('scoring')}")


    splits_to_fit = to_list(tuning_params.get("splits_to_fit", "TRAIN"))
    group_col_name = tuning_params.get("cv_split_group_col_name", None)
#     print(f"\n\nsplits_to_fit={splits_to_fit}, group_col_name={group_col_name}")

#     features_per_iteration = extract_features(features, data, iteration_col_name) 
    unique_iterations = data[iteration_col_name].unique()
    features_per_iteration = {iter: features for iter in unique_iterations}
#     print(f"\n\n len(features_per_iteration)={len(features_per_iteration)}")
    
#     print(f"data.shape={data.shape}, data.groupby(iteration_col_name)= {data.groupby(iteration_col_name).count()}")

    tune_results = (
        data.groupby(iteration_col_name)
        .apply(
            _find_best_params_per_fold,
            estimator,
            split_col_name,
            cv,
#             tuner,
            features_per_iteration,
            target_col_name,
            tuning_params,
            splits_to_fit,
            group_col_name,
        )
        .to_dict()
    )

    best_tuning_score = np.mean([r["best_tuning_score"] for r in tune_results.values()])

    return _assemble_model_parameters(tune_results, tuning_params), best_tuning_score






def _find_best_params_per_fold(partition, estimator, split_col_name, cv_splitter, features_per_iteration, target_col_name, tuning_params, splits_to_fit, group_col_name):
# def _find_best_params_per_fold(partition: pd.DataFrame, estimator: BaseEstimator, split_col_name: str, features_per_iteration: Mapping[int, Any], target_col_name: str, tuning_params: Dict[str, Any], splits_to_fit: Dict[str, Any], group_col_name: Optional[str] = None,) -> Dict[str, Any]:
    """Tunes any sklearn compatible model with a compatible hyperparameter tuner.

    Supported hyperparameter tuning classes:
        - GridSearchCV (default)
        - RandomizedSearchCV
        - BayesSearchCV

    Compatible methods like TuneGridSearchCV, TuneSearchCV are implicitly supported,
    but use them at your own risk.
    """
#     print(f"\n\n _find_best_params_per_fold:\n partition=, estimator, split_col_name={split_col_name}, cv_splitter={cv_splitter}, tuner={tuner}")
#     print(f", features_per_iteration={features_per_iteration}, target_col_name={target_col_name}, splits_to_fit={splits_to_fit}, tuning_params={tuning_params}, group_col_name={group_col_name}, verbose={verbose}")
    feature_col_names = features_per_iteration[partition.name]

    tuning_params = copy.deepcopy(tuning_params)
    train_index = partition[split_col_name].isin(splits_to_fit)

#     cv_splitter = tuning_params.get("cv", None)
#     tuner_class = tuning_params.get("tuner", GridSearchCV)

    # keep all columns in X_train in case any of them are needed to make the split
    partition_train = partition.loc[train_index].copy()
    X_train = partition_train
    y_train = partition_train[target_col_name]
    
#     print(f"\n\nBefore: tuning_params[cv]={tuning_params['cv']}")

    if hasattr(cv_splitter, "split") and not isinstance(cv_splitter, str):
        group_col = (
            partition_train[group_col_name]
            if group_col_name is not None
            else group_col_name
        )
#         print(f"group_col={group_col}")
        tuning_params["cv"] = list(cv_splitter.split(X=X_train, y=y_train, groups=group_col))

#     print(f"\nAfter: len(tuning_params[cv])={len(tuning_params['cv'])}")
#     print(f"After: tuning_params[cv]={tuning_params['cv']}")

    # use only selected features for tuning 
    X_train = X_train[feature_col_names]


#     tuner_args = tuning_params.get("tuner_args", {})

    # Checking if a parsing function has been defined for a given key in tuner_args
    # If true, parse the value for a given key.
#     for arg in tuner_args:
#         if arg in STRING_PARSE_MAP:
#             tuner_args[arg] = STRING_PARSE_MAP[arg](tuner_args[arg], estimator)

#     # Build kwargs dict for tuner class choosing the valid keys from tuner_args
#     search_args = {
#         key: tuner_args.get(key, None)
#         for key in tuner_args.keys()
#         if key in list(inspect.signature(tuner_class.__init__).parameters.keys())
#     }
#     # Append to kwargs dict for tuner class choosing the additional valid keys from
#     # tuner_params (e.g. cv)
#     search_args.update(
#         {
#             key: tuning_params.get(key, None)
#             for key in tuning_params.keys()
#             if key in list(inspect.signature(tuner_class.__init__).parameters.keys())
#         }
#     )

#     search_args['estimator'] = estimator
#     tuner = tuner_class(**search_args)
    space = tuning_params.get('tuner_args').get('param_distributions')
#     print(f"\n\n space={space}")
#     print(f"n_iter= {tuning_params.get('tuner_args').get('n_iter')}")
#     print(f"scoring= {tuning_params.get('tuner_args').get('scoring')}")


    tuner = RandomizedSearchCV(estimator, space, 
                                n_iter=tuning_params.get('tuner_args').get('n_iter'), 
                                scoring=tuning_params.get('tuner_args').get('scoring'), 
                                verbose=tuning_params.get('tuner_args').get('verbose'), 
#                                 error_score='raise',
                                cv=tuning_params["cv"])
    
    
#     print(f"y_train {y_train.unique()}")
#     [print("\n", col, "\n", X_train[col].unique()) for col in X_train.columns]
#     print(f"\n x[rev_m_bill_shock_eom_total_bill_amt_1m_to_avg_12m_flg]:\n {type(X_train['rev_m_bill_shock_eom_total_bill_amt_1m_to_avg_12m_flg'].unique()[0])}")


#     X_train = X_train.replace(pd.NA, np.nan)
#     [print("\n", col, "\n", X_train[col].unique()) for col in X_train.columns]

#     print(f"\n\ny_train[0]={y_train[0]}, type(y_train)={type(y_train[0])}")
#     test = 1
#     print(f"type(test)={type(test)}")
    y_train = y_train.astype(int)
#     print(f"\ntest[0]={y_train[0]}, type(y_train[0])={type(y_train[0])}")
#     print("\n\n")
    tuner.fit(X_train, y_train)

    print("\n\n\n ***************************************************")
    
    tune_result = {
        "estimator_args": tuner.best_params_,
        "best_tuning_score": tuner.best_score_,
    }
    print(f"\n\ntune_result = {tune_result}")
    print(f"\n\n uner.cv_results_ = {tuner.cv_results_}")
    
#     print(f"\n\nverbose={verbose}")
#     if verbose:
#         print("Best estimator")
#         print("CV results")

    return tune_result


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


def _assemble_model_parameters(tune_results, tune_config):
# def _assemble_model_parameters(tune_results: Dict[int, Dict[str, Any]], tune_config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Adds information from the original tuning configs back to search results."""
    tune_results = copy.deepcopy(tune_results)
    for result in tune_results.values():
        result.update(tune_config)

    return tune_results
