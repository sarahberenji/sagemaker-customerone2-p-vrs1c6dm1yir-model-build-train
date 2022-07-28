import re
import pandas as pd

def to_pandas(data, pandas_params, spine_params, target_col, data_dictionary, features, ref_date):
    # def to_pandas(data: DataFrame, pandas_params: Dict[str, Any], spine_params: Dict[str, Any], target_params: Dict[str, Any], data_dictionary: Optional[DataDict] = None, features: List[str] = None, ref_date: Optional[str] = None,) -> pd.DataFrame:
    ref_period = pandas_params.get("ref_period")
    include_target = pandas_params.get("include_target")
    key_col = to_list(spine_params.get('keys'))
    date_col = spine_params.get('date_column')
    #     target_col = target_params.get('target_variable_column')

    if ref_date and ref_period:
        raise AttributeError(
            "Cannot specify both ref_date and ref_period at"
            "the same time. Specify only one argument."
        )

    print(f"\tref_period={ref_period}, \n\tinclude_target={include_target}, \n\tkey_col={key_col}, \n\tdate_col={date_col}, \n\ttarget_col={target_col}, \n\tdata_dictionary={data_dictionary}")
    print(f"len(features)= {len(features)}")

    feat_col, target_switch_col = [None] * 2

    if data_dictionary or features:
        if data_dictionary:
            feat_col = data_dictionary.get_features()
            target_switch_col = data_dictionary.get_target_switch_variable()
        elif features:
            feat_col = features

        export_cols = []
        if date_col:
            export_cols += [date_col]
        if key_col:
            export_cols += key_col
        export_cols += feat_col
        if include_target and target_col:
            export_cols += [target_col]

        cols_diff = set(export_cols) - set(data.columns)
        if len(cols_diff) != 0:
            # logger.warning(f"The following columns are not available in the DataFrame and will be ignored: {cols_diff}")
            export_cols = list(set(export_cols) - cols_diff)

        print(f"len(export_cols)={len(export_cols)}")
        print(f"Before: {data.shape}")
        data = data[export_cols]
        print(f"After: {data.shape}")

    if ref_date and date_col:
        try:
            if not re.match(r"\d{4}-\d{2}-\d{2}", ref_date):
                raise ValueError("Date format should follow yyyy-mm-dd.")

            ref_date = pd.to_datetime(ref_date, infer_datetime_format=True).strftime("%Y-%m-%d")
            print(f"ref_date={ref_date}, type = {type(ref_date)}")

            data['dt'] = pd.to_datetime(data[date_col], format='%Y-%m-%d %H:%M:%S')
            data = data[data['dt'] == ref_date]
            print(f"After filter for day {ref_date} data.shape = {data.shape}")
            data = data.drop(columns=['dt'])
            print(f"After drop filter for day {ref_date} data.shape = {data.shape}")
        except (ValueError, TypeError) as exception:
            raise exception("'ref_date' type casting failed") from exception

    # TODO: Sarah The followings are not tested!!
    print(
        f"first if = {include_target and target_col}, 2nd if={include_target and target_switch_col}, 3rd if={ref_period and date_col}")
    if include_target and target_col:
        print(f"\nBefore include_target shape is {data.shape}")
        data = data[data[target_col.notnull()]]
        print(f"After include_target shape is {data.shape}")

    if include_target and target_switch_col:
        print(f"\nBefore include_target and target_switch_col shape is {data.shape}")
        # data = data.filter(~f.col(target_switch_col))
        data = data[data[target_switch_col]]
        print(f"After include_target and target_switch_col shape is {data.shape}")

    if ref_period and date_col:
        print(f"\nBefore filter_ref_window shape is {data.shape}")
        # data = filter_ref_window(data, ref_period, date_col)
        print(f"After filter_ref_window shape is {data.shape}")

    return data


def to_list(item):
    """
    Wraps an item to a list or returns the existing list

    Args:
        item (Union[Any, List[Any]]): a list of a single object

    Returns:
        item wrapped in a list, if item is not a list. Otherwise the
        original list is returned
    """

    return [item] if not isinstance(item, list) else item
