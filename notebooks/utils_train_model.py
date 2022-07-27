from imblearn.pipeline import Pipeline as imblearn_pipeline
from sklearn.pipeline import Pipeline as sklearn_pipeline

def train_model(data, model_params, features, splitting_params, target_col_name, estimator):
# def train_model(data: pd.DataFrame, model_params: Union[Dict[int, Dict], Dict[str, Any]], features: Union[pd.DataFrame, Mapping[str, Any]], splitting_params: Mapping[str, Any], target_col_name: str, estimator: BaseEstimator=None) -> Mapping[str, Any]:
    """Trains a sklearn compatible model.

    `model_params` expects the following key value pairs:
        `estimator`: Path to estimator class
        `estimator_args`: (Optional) Arguments to be passed to estimator instantiation
        `splits_to_fit`: (Optional) Which splits to fit, default TRAIN
        `weight_samples`: (Optional) True/False param whether to re-weight samples

    `splitting_params` expects the following key value pairs:
        `iteration_col_name`: Name of column in which the fold id is to be saved
        `split_col_name`: Name of column in which the TRAIN/TEST split is saved

    Args:
        data: A dataframe containing features, targets, fold and split information
        model_params: Describe the model, it's instantiation arguments and target,
            whether to reweight the samples.
            Model parameters may either be a single parameter dictionary which is used
            across all iteration or a dictionary in which the key is an iteration id
            and the value is a parameter dictionary.
        features: The features to fit the model on. If a dataframe, it must conform
            to the format used by the `select_features` node in this connector. If a
            Dict, then you can pass the following information:
            `block_list`: (Optional) List of columns which will not be considered as
                features, e.g. the primary key.
            `inclusion_list`: (Optional) List of features to include.
            `inclusion_regex`: (Optional) List of regular expressions. Features must
                match at least one of them to be included. Either `inclusion_list` or
                `inclusion_regex` can be specified, not both. If none of them is
                specified, the model is run on all the features.
        splitting_params: Describe where fold and splitting information are located
        target_col_name: Name of target column
        estimator: (optional) estimator object that is assumed to implement the 
            scikit-learn estimator interface

    Returns:
        A dictionary with the fold id as key and the model fit on that fold as value

    Raises:
        ValueError: If a feature specified in model_params is not present in the data
    """
    iteration_col_name = splitting_params.get("iteration_col_name", "iteration_id")
    split_col_name = splitting_params.get("split_col_name", "split")
    print(f"data.shape={data.shape}")
#     print(data[iteration_col_name].unique())

#     model_params = extract_model_parameters(
#         parameters=model_params, data=data, iteration_col_name=iteration_col_name
#     )

#     features_per_iteration = extract_features(features, data, iteration_col_name)
    unique_iterations = data[iteration_col_name].unique()
#     print(f"unique_iterations={unique_iterations}")
    features_per_iteration = {iter: features for iter in unique_iterations}
    print(f"len(features_per_iteration)={len(features_per_iteration[0])}")    
    print(f"type(features_per_iteration)={type(features_per_iteration[0])}")

#     print(f"before groupBy data.shape={data.shape}")
    return (
        data.groupby(iteration_col_name)
        .apply(
            _train_model_per_fold,
            split_col_name,
            features_per_iteration,
            model_params,
            target_col_name,
            estimator
        )
        .to_dict()
    )


def _train_model_per_fold(partition, split_col_name, features_per_iteration, model_params, target_col_name, estimator):
# def _train_model_per_fold(partition: pd.DataFrame, split_col_name: AnyStr, features_per_iteration: Mapping[int, Any], model_params: Dict[int, Dict], target_col_name: str, estimator: BaseEstimator = None,) -> Any:
    """Trains a single sklearn compatible model."""
#     print("\n")
#     print(f"model_params={model_params}")
#     print(f"\nafter groupBy partition.shape={partition.shape}")
    feature_col_names = features_per_iteration[partition.name]
#     print(f"len(feature_col_names)={len(feature_col_names)}")    
#     print(f"Before cast_to_list type(feature_col_names)={type(feature_col_names)}")

    feature_col_names = cast_to_list(feature_col_names) # might not need it!!! 
#     print(f"After cast_to_list type(feature_col_names)={type(feature_col_names)}")

    check_available_in_data(feature_col_names, partition)
#     print(f"type(feature_col_names)={type(feature_col_names)}")
    
    
    """Unpacks the parameters to store in the modelling container."""
    estimator_name = model_params.get("estimator", None)
    estimator_args = model_params.get("estimator_args", {})
    splits_to_fit = model_params.get("splits_to_fit", "TRAIN")
    weight_samples = model_params.get("weight_samples", False)    
#     if estimator:
    model = estimator
    model.set_params(**estimator_args)
    fitted_model = None
    weights = None
    calibration_method = None
    feature_importances_ = None

    print(f"  weight_samples={weight_samples}, \n  splits_to_fit={splits_to_fit}, \n  estimator_name={estimator_name}, \n  estimator_args={estimator_args}")
#     print(f"\n  model={model}")
#     wrapped_model = ModelContainer(
#         model_params[partition.name], feature_col_names, target_col_name, estimator
#     )

#     print(f"\nbefore check_available_in_data partition.shape={partition.shape}")

    # Fit model from ModelContainer starts!!! 
#     fitted_model = wrapped_model.fit(partition, split_col_name)
#     check_available_in_data(cast_to_list(target_col_name), partition)
#     print(f"after check_available_in_data data.shape={data.shape}")

    
    train_index = partition[split_col_name].isin([splits_to_fit])
#     print(f"\ntrain_index.shape={train_index.shape}")
#     print(f"train_index={train_index}")

    data_train = partition.loc[train_index].copy()
    X = data_train[feature_col_names]
    y = data_train[target_col_name]
#     print(f"y={type(y[0])}")
    y=y.astype(int)
#     print(f"y={type(y[0])}")

#     print(f"data_train.shape = {data_train.shape}")
#     print(f"X.shape = {X.shape}")
#     print(f"y.shape = {y.shape}")

#     if self.weight_samples:
#         weights = compute_class_weight(
#             class_weight="balanced", classes=np.unique(y.values), y=y.values
#         )
#         sample_weights = _create_sample_weight_array(weights, y)
#         self.weights = sample_weights
#         fitted_model = self.model.fit(
#             X, y, sample_weight=sample_weights
#         )
#     else:
#         fitted_model = self.model.fit(X, y)    
    fitted_model = model.fit(X, y)

    if hasattr(fitted_model, 'feature_importances_'):
        print("\n1111111111")
        feature_importances_ = fitted_model.feature_importances_
    elif isinstance(fitted_model, (imblearn_pipeline, sklearn_pipeline)):
        print("\n222222222")
        last_estimator = fitted_model[-1]
        print(f"last_estimator={last_estimator}")
        if hasattr(last_estimator, 'feature_importances_'):
            print(f"\n333333333 last_estimator.feature_importances_ = {last_estimator.feature_importances_}")
            feature_importances_ = last_estimator.feature_importances_
            print(f"\tteature_importances_={feature_importances_}")
    if feature_importances_ is None:
        print("\n44444444")
        print("Could not find 'feature_importances_' in estimator. Setting importance to all zeros.")
        feature_importances_ = np.zeros(len(feature_col_names))

    # These are all in the ModelContainer self variable!! 
    fitted_model = {'feature_col_names': feature_col_names,
                    'estimator_name': estimator_name, 
                    'estimator_args': estimator_args,
                    'splits_to_fit': splits_to_fit, 
                    'weight_samples': weight_samples,
                    'model': model, 
                    'target_col_name': target_col_name,
                    'weights': weights,
                    'calibration_method': calibration_method,
                    'feature_importances_': feature_importances_,
                    'fitted_model': fitted_model,
                   }
    return fitted_model


def cast_to_list(items):
    """Cast input to list, if input is a single string or another iterable."""
    return [items] if isinstance(items, str) else list(items)


def check_available_in_data(features, data):
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
