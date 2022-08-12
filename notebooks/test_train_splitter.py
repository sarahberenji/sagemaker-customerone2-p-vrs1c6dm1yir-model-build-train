"""Classes with test train splitters"""
from datetime import datetime
from typing import Union

from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GroupShuffleSplit


class OutOfTimeSplitter:
    """ 
    This is a class that allows to create test train split that is
    out of time.
    """

    def __init__(self, test_size, date_col, n_splits = 5, random_state = None, verbose = None):
        """
        Creates an instance of class OutOfTimeSplitter

        Args:
            test_size: float or int
                sample size of test set. If float, then it represents the proportion of 
                the dataset to include in the test split. If int, then it represents the 
                number of test samples. Test_size ratio for each split is ``test_size / n_splits``
            date_col: str
                name of date column
            n_splits: int
                number of splits
            random_state: int
                seed
            verbose: int
                controls the verbosity
        """
        self.test_size = test_size
        self.date_col = date_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """
        Split data into train and test

        Args:
            X: array-like of shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.

            y: array-like of shape (n_samples,), default=None
                Always ignored, exists for compatibility.

            groups: array-like of shape (n_samples,), default=None
                Always ignored, exists for compatibility.

        Returns:
            train : ndarray
                The training set indices for that split.

            test : ndarray
                The testing set indices for that split.
        """
        test_size = self.test_size
        date_col = self.date_col
        n_splits = self.n_splits
        random_state = self.random_state

        if isinstance(test_size, int):
            test_size = test_size / len(X)

        size_per_split = test_size / n_splits

        idx_map = X.rename_axis('pre_shuffle_idx').reset_index()[['pre_shuffle_idx']]
        # shuffle
        X = X.sample(frac=1.0, random_state=random_state)
        # order on date, but order within each date is shuffled
#         print(f"\n\n 1-OutOfTimeSplitter.split > X.columns={X.columns}")
#         print(f"\n\n 2-OutOfTimeSplitter.split > date_col={date_col}")
        X.sort_values(date_col, ascending=False, inplace=True)

        print(f"\n\n 3-OutOfTimeSplitter.split > n_splits = {n_splits}")
        for split in range(n_splits):
            lower = test_size - size_per_split * split
            upper = test_size - size_per_split * (split + 1)

            lower_idx = int(len(X)*lower)
            upper_idx = int(len(X)*upper)
            lower_date = X.iloc[lower_idx][date_col]
            upper_date = X.iloc[upper_idx][date_col]

            post_shuffle_train_idx = X[X[date_col] < lower_date].index
            post_shuffle_test_idx = X[(lower_date <= X[date_col]) & (X[date_col] <= upper_date)].index

            train_idx = idx_map[idx_map.pre_shuffle_idx.isin(post_shuffle_train_idx)].index
            test_idx = idx_map[idx_map.pre_shuffle_idx.isin(post_shuffle_test_idx)].index

            if self.verbose:
                print(f"\n OutOfTimeSplitter.split > split={split}")
                for type, idx in zip(["train", "test"], [post_shuffle_train_idx, post_shuffle_test_idx]):
                    loc = X.loc[idx]
                    print(f"\t{type}: \t{loc[date_col].min()} -> {loc[date_col].max()}\tshape: {loc.shape}\tratio: {len(idx)/len(X):.2f}")

            print(f"\n OutOfTimeSplitter.split > train_idx={train_idx}, test_idx={test_idx}")
            yield train_idx, test_idx
            
#     def get_n_splits(self, X, y, groups):
#         print(f"\n 5-self.n_splits={self.n_splits}")
# #         print(f"X={X}, y={y}, groups={groups}")
#         return self.n_splits
# X= [4188 rows x 65 columns], y=0       0
# 2       0
# 4       0
# 5       0
# 6       0
#        ..
# 5891    1
# 5892    0
# 5893    0
# 5894    1
# 5895    1
# Name: tgt_xsell_cust_voice_to_fixed, Length: 4188, dtype: Int32, groups=None