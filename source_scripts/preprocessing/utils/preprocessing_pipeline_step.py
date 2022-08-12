
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder


class PandasSimpleImputer(SimpleImputer):
    """
    A wrapper around `SimpleImputer` to return data frames with columns.
    """
    print("\n\n ****** PandasSimpleImputer > fit ******** ")

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        print(" ****** PandasSimpleImputer > transform ******** ")
        return pd.DataFrame(super().transform(X), columns=self.columns, index=X.index)


def preprocessing_pipeline_step(categorical_features, numerical_features):
    # def preprocessing_pipeline_step(params: Dict[str, Any],  **kwargs) -> ColumnTransformer:

    print(" ****** preprocessing_pipeline_step  ******** ")
    # args = _parse_args(params)
    # args = {'categorical': {'impute': , 'encoder': TargetEncoder()}, 'numerical': {'imputer': PandasSimpleImputer()}}
    preprocessing = []

    # preproc_name = 'categorical'
    # steps = {'impute': PandasSimpleImputer(strategy='most_frequent'), 'encoder': }
    pipeline_cat = Pipeline(steps=[('impute', PandasSimpleImputer(strategy='most_frequent')),
                                   ('encoder', TargetEncoder())])
    preprocessing.append(('categorical', pipeline_cat, categorical_features))

    # preproc_name = 'numerical'
    # steps = {'imputer': PandasSimpleImputer()}
    pipeline_num = Pipeline(steps=[('imputer', PandasSimpleImputer())])
    preprocessing.append(('numerical', pipeline_num, numerical_features))

    print(f"\n\n len preprocessing = {len(preprocessing)}")

    return ColumnTransformer(preprocessing)


