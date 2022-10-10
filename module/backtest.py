from datetime import datetime
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from module.data import reorganize_raw_df


class BackTest:

    def __init__(self, start_date, end_date, data_path, 
            holding_periods: List[int], n_groups: int,
            price='close') -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.holding_periods = holding_periods
        self.n_groups = n_groups
        self.data_path = data_path
        self.price = price
        self.data = None

    def setup_data(self):
        df = pd.read_csv(self.data_path)
        df = reorganize_raw_df(df)
        df = df[['s_dq_adjopen', 's_dq_adjhigh', 's_dq_adjlow', 
                 's_dq_adjclose', 's_dq_volume', 's_dq_avgprice', 's_dq_tradestatuscode']]
        df.columns = ['open', 'high', 'low', 'close', 'vol', 'vwap', 'statuscode']
        # drop unnormal data (volume = 0)
        df = df[df['vol'] > 0]
        df = df[self.price]
        # add missing values
        df = df.unstack().stack(dropna=False)
        df = df.loc[~df.groupby('instrument').ffill().isnull()]
        df = df.loc[~df.groupby('instrument').bfill().isnull()]
        df = df.loc[self.start_date:self.end_date]
        self.data = df

    def shift_date(self, alpha: pd.Series, n_shift: int = 0) -> pd.Series:
        return alpha.groupby('instrument').shift(n_shift)

    def alpha_backtest(self, alpha: pd.Series, alpha_shifted=True, 
            plot=False, save_dir: str = 'outputs') -> dict:
        if not alpha_shifted:
            self.shift_date(alpha, 1)
        results = {}
        for holding_period in self.holding_periods:
            stock_return = self.get_future_return(holding_period, normalize=True)
            ic_results = self.alpha_ic_test(alpha, stock_return)
            group_results = self.group_test(alpha, stock_return, plot=plot, 
                                            save_path=os.path.join(save_dir, f'{holding_period}.png'))
            info = {}
            info.update(ic_results)
            info.update(group_results)
            results[holding_period] = info
        return results

    def get_future_return(self, n_days_ahead, normalize):
        if self.data is None:
            self.setup_data()
        future_price = self.data.groupby('instrument').shift(-n_days_ahead)
        future_return = future_price / self.data - 1
        future_return = future_return.dropna()
        if normalize:
            future_return = future_return / n_days_ahead
        return future_return

    def group_test(self, alpha: pd.Series, future_return: pd.Series, plot: bool, save_path: str):
        df = pd.concat([alpha, future_return], axis=1, join='inner')
        df.columns = ['alpha', 'return']
        df['group'] = df.groupby('datetime')['alpha'].rank(pct=True
            ).map(lambda x: int(self.n_groups * x) if x != 1 else self.n_groups-1)
        top_avg_return = df[df['group'] == self.n_groups-1].groupby('datetime')['return'].mean()
        avg_return = df.groupby('datetime')['return'].mean()
        df = pd.concat([top_avg_return, avg_return], axis=1)
        df.columns = ['top', 'avg']
        cum_value = (df + 1).cumprod()
        info = {}
        begin_date = datetime.strptime(str(cum_value.index.min()), '%Y%m%d')
        end_date = datetime.strptime(str(cum_value.index.max()), '%Y%m%d')
        n_days = (end_date - begin_date).days
        info['top_return'] = cum_value['top'].iloc[-1] ** (365/n_days) - 1
        info['avg_return'] = cum_value['avg'].iloc[-1] ** (365/n_days) - 1
        info['excess'] = info['top_return'] - info['avg_return']
        if plot:
            # do plot
            cum_value.index = cum_value.index.map(lambda x: datetime.strptime(str(x), '%Y%m%d'))
            cum_value.plot()
            parent_dir = os.path.dirname(save_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            plt.savefig(save_path, dpi=1000)
            print(f'Plot saved to {save_path}.')
        return info

    def alpha_ic_test(self, alpha: pd.Series, future_return: pd.Series):
        df = pd.concat([alpha, future_return], axis=1, join='inner')
        ic = df.groupby('datetime').corr().groupby(level=1).mean().to_numpy()[0, 1]
        rankic = df.groupby('datetime').corr(method='spearman').groupby(level=1).mean().to_numpy()[0, 1]
        return {'IC': ic, 'RankIC': rankic}
