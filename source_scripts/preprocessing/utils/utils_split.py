# from typing import Any, AnyStr, Mapping
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(X, y, groups, splitter_args):
    test_size = splitter_args['test_size']
    date_col = splitter_args['date_col']
    n_splits = splitter_args['n_splits']
    random_state = splitter_args['random_state']
    verbose = splitter_args['verbose']

    if isinstance(test_size, int):
        test_size = test_size / len(X)  # len(X.columns)=163

    size_per_split = test_size / n_splits

    idx_map = X.rename_axis('pre_shuffle_idx').reset_index()[['pre_shuffle_idx']]
    # shuffle
    X = X.sample(frac=1.0, random_state=random_state)
    # order on date, but order within each date is shuffled
    X.sort_values(date_col, ascending=False, inplace=True)

    for split in range(n_splits):
        lower = test_size - size_per_split * split
        upper = test_size - size_per_split * (split + 1)

        lower_idx = int(len(X) * lower)
        upper_idx = int(len(X) * upper)
        lower_date = X.iloc[lower_idx][date_col]
        upper_date = X.iloc[upper_idx][date_col]

        post_shuffle_train_idx = X[X[date_col] < lower_date].index
        post_shuffle_test_idx = X[(lower_date <= X[date_col]) & (X[date_col] <= upper_date)].index

        train_idx = idx_map[idx_map.pre_shuffle_idx.isin(post_shuffle_train_idx)].index
        test_idx = idx_map[idx_map.pre_shuffle_idx.isin(post_shuffle_test_idx)].index

        if verbose:
            print(f"\nSplit {split}")
            for type, idx in zip(["train", "test"], [post_shuffle_train_idx, post_shuffle_test_idx]):
                loc = X.loc[idx]
                print(
                    f"\t{type}: \t{loc[date_col].min()} -> {loc[date_col].max()}\tshape: {loc.shape}\tratio: {len(idx) / len(X):.2f}")

        yield train_idx, test_idx


def _extract_group_column_arg(data: pd.DataFrame, group_col_name, target_col_name: str):
    # def _extract_group_column_arg(data: pd.DataFrame, splitting_params: Mapping[str, Any], target_col_name: str,) -> Mapping[str, Any]:
    """Extracts a group column and target column.

    This is for group based splitters like GroupShuffleSplit and stratified splitters.
    See relevant sklearn documentation on details of groups.
    """
    split_func_args = {}

    print(f"\n\n_extract_group_column_arg: group_col_name={group_col_name}")
    print(f"_extract_group_column_arg: target_col_name={target_col_name}")

    if target_col_name is not None:  # 'tgt_xsell_cust_voice_to_fixed'
        split_func_args["y"] = data[target_col_name].copy()

    if group_col_name is not None:
        split_func_args["groups"] = data[group_col_name].copy()
    return split_func_args  # a dict with two keys: y and groups


def _subsplit_within_fold(fold_df, splitter, split_col_name, source_split, subsplit_params, split_a_name, split_b_name,
                          target_col_name) -> pd.DataFrame:
    # def _subsplit_within_fold(fold_df: pd.DataFrame, splitter: Any, split_col_name: AnyStr, source_split: AnyStr, subsplit_params: Mapping[str, Any], split_a_name: AnyStr, split_b_name: AnyStr, target_col_name: str,) -> pd.DataFrame:
    """Within each fold, splits a `source_split` into two further splits."""
    source_split_indices = fold_df[split_col_name] == source_split
    data_to_split = fold_df.loc[source_split_indices]

    split_func_args = _extract_group_column_arg(
        data_to_split, subsplit_params, target_col_name
    )
    indices = list(splitter.split(data_to_split, **split_func_args))
    split_a_index, split_b_index = indices[0]
    data_to_split.loc[data_to_split.index[split_a_index], split_col_name] = split_a_name
    data_to_split.loc[data_to_split.index[split_b_index], split_col_name] = split_b_name
    fold_df[source_split_indices] = data_to_split
    return fold_df


def make_splits(data: pd.DataFrame, target_col_name: str = None, ):
    # def make_splits(data: pd.DataFrame, splitting_params: Mapping[str, Any], target_col_name: str = None,) -> pd.DataFrame:
    """Splits data into TRAIN and TEST using sklearn compatible splitters.

    `splitting_params` expects the following keys:
        splitter: Path to splitter class
        splitter_args: Arguments to be passed to splitter instantiation
        iteration_col_name: Name of column in which the fold id is to be saved
        split_col_name: Name of column in which the TRAIN/TEST split is saved
        group_col_name: (Optional) Name of column to use for group based splitters

    Args:
        data: Model input data containing spine, features and target
        splitting_params: Dictionary with splitting configuration
        target_col_name: (Opt) Name of target column

    Returns:
        A new pandas dataframe containing the splits, Fold ID and TRAIN/TEST split
        columns are saved in columns.
    """
    splitter_args = {'n_splits': 1, 'test_size': 0.2, 'date_col': 'current_dt', 'verbose': 1, 'random_state': 42}

    iteration_col_name = 'iteration_id'
    split_col_name = 'split'

    data[iteration_col_name] = np.nan
    data[split_col_name] = np.nan

    split_func_args = _extract_group_column_arg(data, 'customer_id', target_col_name)

    all_data_frames = []
    for iteration_id, (train_index, test_index) in enumerate(
            split_data(data, split_func_args['y'], split_func_args['groups'], splitter_args)
    ):
        all_indices_in_this_fold = list(set(train_index).union(test_index))
        fold_data = data.loc[all_indices_in_this_fold, :].copy()
        fold_data.loc[:, iteration_col_name] = iteration_id
        fold_data.loc[train_index, split_col_name] = "TRAIN"
        fold_data.loc[test_index, split_col_name] = "TEST"
        all_data_frames.append(fold_data)

    return pd.concat(all_data_frames, axis=0, ignore_index=True)


def make_subsplit(data, subsplit_params, splitter_args, target_col_name) -> pd.DataFrame:
    # def make_subsplit(data: pd.DataFrame, subsplit_params: Mapping[str, Any], splitting_params: Mapping[str, Any], target_col_name: str = None,) -> pd.DataFrame:
    """Subdivides a `split` category into smaller splits within each iteration fold.

    This is useful to e.g. divide the test set into test and validation sets,
    giving us train / test / validation splits.

    ``subsplit_params`` expects the following keys:
        splitter: Path to splitter class
        iteration_col_name: Name of column in which the fold id is saved
        split_col_name: Name of column in which the TRAIN/TEST split is saved
        source_split: Name of the split to split further, e.g. TRAIN
        target_splits: Names of the two new splits to be created, e.g. TEST / VAL
        splitter_args: (Optional) Arguments to be passed to splitter instantiation
        group_col_name: (Optional) Name of column to use for group based splitters

    `splitting_params` expects the following keys:
        iteration_col_name: Name of column in which the fold id is saved
        split_col_name: Name of column in which the TRAIN/TEST split is saved

    Args:
        data: Data containing folds and splits
        subsplit_params: Dictionary with subsplit configuration
        splitting_params: Dictionary with splitting configuration
        target_col_name: (Opt) Name of target column

    Returns:
         A dataframe with subsplit splits
    """
    #     splitter_name = subsplit_params.get("splitter") # 'sklearn.model_selection.StratifiedShuffleSplit'
    #     splitter_args = subsplit_params.get("splitter_args", {}) # {'test_size': 0.1, 'random_state': 42}

    #     splitter = load_obj(splitter_name)(n_splits=1, **splitter_args)
    splitter = StratifiedShuffleSplit(n_splits=1, **splitter_args)

    iteration_col_name = subsplit_params["iteration_col_name"]  # 'iteration_id'
    split_col_name = subsplit_params["split_col_name"]  # 'split'

    source_split = subsplit_params["source_split"]  # 'TRAIN'
    target_splits = subsplit_params["target_splits"]  # ['TRAIN', 'CAL']

    train_name = target_splits[0]
    test_name = target_splits[1]

    #     print(f"\n\n\niteration_col_name={iteration_col_name}")
    #     print(f"ata[iteration_col_name].value_counts(dropna=False): {data[iteration_col_name].value_counts(dropna=False)}")
    #     print(f"\n data.columns = {data.columns}")

    return data.groupby(iteration_col_name).apply(
        _subsplit_within_fold,
        splitter,
        split_col_name,
        source_split,
        subsplit_params,
        train_name,
        test_name,
        target_col_name,
    )


def _subsplit_within_fold(fold_df, splitter, split_col_name, source_split, subsplit_params, split_a_name, split_b_name,
                          target_col_name) -> pd.DataFrame:
    # def _subsplit_within_fold(fold_df: pd.DataFrame, splitter: Any, split_col_name: AnyStr, source_split: AnyStr, subsplit_params: Mapping[str, Any], split_a_name: AnyStr, split_b_name: AnyStr, target_col_name: str,) -> pd.DataFrame:
    """Within each fold, splits a `source_split` into two further splits."""

    #     print(f"\n\nfold_df.columns= {fold_df.columns}")
    #     print("****",splitter, split_col_name, source_split, subsplit_params, split_a_name, split_b_name, target_col_name)

    #     print(f"\n fold_df[split_col_name] == source_split = {fold_df[split_col_name] == source_split}")
    source_split_indices = fold_df[split_col_name] == source_split
    #     print(f"source_split_indices={source_split_indices}")
    data_to_split = fold_df.loc[source_split_indices]
    #     print(f"\n\ndata_to_split= {data_to_split}")

    split_func_args = _extract_group_column_arg(data_to_split, None, target_col_name)

    #     print(f"\nsplit_func_args={split_func_args}")
    indices = list(splitter.split(data_to_split, **split_func_args))
    split_a_index, split_b_index = indices[0]
    data_to_split.loc[data_to_split.index[split_a_index], split_col_name] = split_a_name
    data_to_split.loc[data_to_split.index[split_b_index], split_col_name] = split_b_name
    fold_df[source_split_indices] = data_to_split

    return fold_df

