# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Union, Text
import numpy as np
import pandas as pd

from module.utils import clock


EPS = 1e-12


def robust_zscore(x: pd.Series, zscore=False):
    """Robust ZScore Normalization
    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826
    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    x = x - x.median()
    mad = x.abs().median()
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())


def get_level_index(df: pd.DataFrame, level=Union[str, int]) -> int:
    """
    get the level index of `df` given `level`
    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level
    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    """
    fetch data from `data` with `selector` and `level`
    selector are assumed to be well processed.
    `fetch_df_by_index` is only responsible for get the right level
    Parameters
    ----------
    selector : Union[pd.Timestamp, slice, str, list]
        selector
    level : Union[int, str]
        the level to use the selector
    Returns
    -------
    Data of the given index.
    """
    # level = None -> use selector directly
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    # Try to get the right index
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[
                    pd.IndexSlice[idx_slc],
                ]
        else:  # pylint: disable=W0120
            return df
    else:
        return df.loc[
            pd.IndexSlice[idx_slc],
        ]


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    get a group of columns from multi-index columns DataFrame
    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class Processor:
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters
        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.
        """

    @abc.abstractmethod
    @clock
    def __call__(self, df: pd.DataFrame):
        """
        process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside
        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.
        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True

    def readonly(self) -> bool:
        """
        Does the processor treat the input data readonly (i.e. does not write the input data) when processing
        Knowning the readonly information is helpful to the Handler to avoid uncessary copy
        """
        return False

    def config(self, **kwargs):
        attr_list = {"fit_start_time", "fit_end_time"}
        for k, v in kwargs.items():
            if k in attr_list and hasattr(self, k):
                setattr(self, k, v)

        for attr in attr_list:
            if attr in kwargs:
                kwargs.pop(attr)
        super().config(**kwargs)


class DropnaProcessor(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    @clock
    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))

    def readonly(self):
        return True


class DropnaFeature(Processor):
    def __init__(self, fields_group=None, max_na=0):
        self.fields_group = fields_group
        self.max_na = max_na

    @clock
    def __call__(self, df):
        slc = get_group_columns(df, self.fields_group)
        return df.dropna(subset=slc, 
                    thresh=max(len(slc) - self.max_na, 0))

    def readonly(self):
        return True


class DropnaLabel(DropnaProcessor):
    def __init__(self, fields_group="label"):
        super().__init__(fields_group=fields_group)

    def is_for_infer(self) -> bool:
        """The samples are dropped according to label. So it is not usable for inference"""
        return False


class DropCol(Processor):
    def __init__(self, col_list=[]):
        self.col_list = col_list

    @clock
    def __call__(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            mask = df.columns.get_level_values(-1).isin(self.col_list)
        else:
            mask = df.columns.isin(self.col_list)
        return df.loc[:, ~mask]

    def readonly(self):
        return True


class FilterCol(Processor):
    def __init__(self, fields_group="feature", col_list=[]):
        self.fields_group = fields_group
        self.col_list = col_list

    @clock
    def __call__(self, df):

        cols = get_group_columns(df, self.fields_group)
        all_cols = df.columns
        diff_cols = np.setdiff1d(all_cols.get_level_values(-1), cols.get_level_values(-1))
        self.col_list = np.union1d(diff_cols, self.col_list)
        mask = df.columns.get_level_values(-1).isin(self.col_list)
        return df.loc[:, mask]

    def readonly(self):
        return True


class TanhProcess(Processor):
    """Use tanh to process noise data"""

    @clock
    def __call__(self, df):
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            col = df.columns[~mask]
            data[col] = data[col] - 1
            data[col] = np.tanh(data[col])

            return data

        return tanh_denoise(df)


class ProcessInf(Processor):
    """Process infinity"""

    @clock
    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: Such behavior is very weird
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = data.groupby(axis=0, level='datetime').apply(process_inf)
            data.sort_index(inplace=True)
            return data

        return replace_inf(df)


class Fillna(Processor):
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    @clock
    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            # this implementation is extremely slow
            # df.fillna({col: self.fill_value for col in cols}, inplace=True)

            # So we use numpy to accelerate filling values
            nan_select = np.isnan(df.values)
            nan_select[:, ~df.columns.isin(cols)] = False
            df.values[nan_select] = self.fill_value
        return df


class MinMaxNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.min_val = np.nanmin(df[cols].values, axis=0)
        self.max_val = np.nanmax(df[cols].values, axis=0)
        self.ignore = self.min_val == self.max_val
        self.cols = cols

    @clock
    def __call__(self, df):
        def normalize(x, min_val=self.min_val, max_val=self.max_val, ignore=self.ignore):
            if (~ignore).all():
                return (x - min_val) / (max_val - min_val)
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - min_val) / (max_val - min_val)
            return x

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class ZScoreNorm(Processor):
    """ZScore Normalization"""

    def __init__(self, fit_start_time, fit_end_time, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[cols].values, axis=0)
        self.std_train = np.nanstd(df[cols].values, axis=0)
        self.ignore = self.std_train == 0
        self.cols = cols

    @clock
    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train, ignore=self.ignore):
            if (~ignore).all():
                return (x - mean_train) / std_train
            for i in range(ignore.size):
                if not ignore[i]:
                    x[i] = (x[i] - mean_train) / std_train
            return x

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization
    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826
    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        self.cols = get_group_columns(df, self.fields_group)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += EPS
        self.std_train *= 1.4826

    @clock
    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        df[self.cols] = X
        if self.clip_outlier:
            df.clip(-3, 3, inplace=True)
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @clock
    def __call__(self, df):
        # try not modify original dataframe
        if not isinstance(self.fields_group, list):
            self.fields_group = [self.fields_group]
        for g in self.fields_group:
            cols = get_group_columns(df, g)
            df.loc[:, cols] = df[cols].groupby("datetime").apply(self.zscore_func)
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.
    "Cross Sectional" is often used to describe data operations.
    The operations across different stocks are often called Cross Sectional Operation.
    For example, CSRankNorm is an operation that grouping the data by each day and rank `across` all the stocks in each day.
    Explanation about 3.46 & 0.5
    .. code-block:: python
        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # for any variable
        x_rank = pd.Series(x).rank(pct=True)  # if it is converted to rank, it will be a uniform distributed
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # Normally, we will normalize it to make it like normal distribution
        x_rank.mean()   # accounts for 0.5
        1 / x_rank.std()  # accounts for 3.46
    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    @clock
    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        t = df[cols].groupby("datetime").rank(pct=True)
        t -= 0.5
        t *= 3.46  # NOTE: towards unit std
        df.loc[:, cols] = t
        return df


class CSZFillna(Processor):
    """Cross Sectional Fill Nan"""

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    @clock
    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime").apply(lambda x: x.fillna(x.mean()))
        return df