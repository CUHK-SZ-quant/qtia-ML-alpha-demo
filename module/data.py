from typing import Dict

import pandas as pd
import numpy as np
from .proessor import CSZScoreNorm, CSRankNorm, DropnaFeature, DropnaLabel, DropnaProcessor, Fillna

def add_column_level(df: pd.DataFrame, name='prefix'):
    """Add a level to the first level of column"""
    if len(df.columns.shape) > 1:
        columns = list(zip(*df.columns))
    else:
        columns = [df.columns.tolist()]
    df.columns = pd.MultiIndex.from_arrays([[name]*df.shape[1]] + columns)
    return df


class Dataset:

    def __init__(self, df: pd.DataFrame) -> None:
        self.data_source = df
        self.data_split = {}

    def prepare(self, segments: Dict[str, slice]):
        for split, segment in segments.items():
            if segment is None:
                continue
            print('Preparing', split, 'data ...')
            self.data_split[split] = self.prepare_split(self.data_source, 
                                                        'learn' if split != 'test' else 'infer', 
                                                        segment)
            print('Done.')

    def prepare_split(self, df: pd.DataFrame, split: str, segment: slice):
        df = df.loc[segment]
        if split == 'learn':
            # dropna
            df = DropnaFeature('feature')(df)
            df = DropnaLabel('label')(df)
        else:
            # inference
            df = DropnaFeature('feature', max_na=5)(df)
            df = Fillna('feature')(df)
        return df

    def get_data_split(self, split: str) -> pd.DataFrame:
        if len(self.data_split) == 0:
            raise RuntimeError('Dataset is not prepared yet. Please run Dataset.prepare first.')
        return self.data_split[split]

def reorganize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['object_id'], axis=1)
    df.index = pd.MultiIndex.from_arrays([df['trade_dt'], df['s_info_windcode']],
                                         names=['datetime', 'instrument'])
    df = df.drop(['s_info_windcode', 'trade_dt'], axis=1)
    df = df.sort_index()
    return df

def calculate_feature_from_price_vol(df: pd.DataFrame, max_lookback=5) -> pd.DataFrame:
    df = df[['open', 'high', 'low', 'close', 'vol', 'vwap']]
    feature_dfs = []
    for i in range(1, max_lookback):
        lookback_df = df.groupby('instrument').shift(i)
        # normalize history price by close price
        lookback_df.loc[:, ['open', 'high', 'low', 'close', 'vwap']] = \
            (lookback_df.loc[:, ['open', 'high', 'low', 'close', 'vwap']].transpose() / df['close']).transpose()
        # normalize history price by close price
        lookback_df.loc[:, 'vol'] /= df['vol'].transpose()
        lookback_df.columns = [f'open{i}', f'high{i}', f'low{i}', f'close{i}', f'vol{i}', f'vwap{i}']
        feature_dfs.append(lookback_df)
    feature_df = pd.concat(feature_dfs, axis=1)
    return feature_df

def calculate_return(df: pd.DataFrame, target=5) -> pd.DataFrame:
    # using close to calculate return
    close_df = df[['close']]
    label_df = close_df.groupby('instrument').shift(-target-1) / \
        close_df.groupby('instrument').shift(-1) - 1
    label_df.columns = [f'label{target}']
    return label_df

def construct_feature_target_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[['s_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 
                 's_dq_adjclose', 's_dq_volume', 's_dq_avgprice', 's_dq_tradestatuscode']]
    df.columns = ['open', 'high', 'low', 'close', 'vol', 'vwap', 'statuscode']
    # drop unnormal data (volume = 0)
    df = df[df['vol'] > 0]
    # add missing values
    df = df.unstack().stack(dropna=False)
    df = df.loc[~df.groupby('instrument').ffill().isnull().all(axis=1)]
    df = df.loc[~df.groupby('instrument').bfill().isnull().all(axis=1)]
    feature_df = calculate_feature_from_price_vol(df)
    target_df = calculate_return(df, target=5)
    data = pd.concat([add_column_level(feature_df, 'feature'), 
                      add_column_level(target_df, 'label'), 
                      add_column_level(df[['statuscode']], 'status')],
                     axis=1)
    return data

def preprocess_df(df: pd.DataFrame):
    # drop untraded stocks
    df = df[df[('status', 'statuscode')] == -1]
    # cross-sectional zscore norm
    cszscore = CSZScoreNorm(fields_group='feature', method='robust')
    df = cszscore(df)
    # cross-sectional rank
    csrank = CSRankNorm(fields_group='label')
    df = csrank(df)
    return df

def prepare_dataset(source_path: str):
    raw_df = pd.read_csv(source_path)
    df = reorganize_raw_df(raw_df)
    df = construct_feature_target_df(df)
    data = preprocess_df(df)
    dataset = Dataset(data)
    return dataset


if __name__ == '__main__':
    source_path = 'data\AShareEODPrices.csv'
    raw_df = pd.read_csv(source_path)
    dataset = prepare_dataset(source_path)
    dataset.prepare({
        'train': slice(20150101, 20181231-7),
        'valid': slice(20190101, 20201231-7),
        'test':  slice(20210101, 20220915),
    })
    df = dataset.get_data_split('train')
    print(df)
    print(dataset.get_data_split('valid'))
    print(dataset.get_data_split('test'))