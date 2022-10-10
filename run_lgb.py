import pandas as pd
import numpy as np
import argparse
from module.backtest import BackTest

from module.data import prepare_dataset
from module.gbdt_model import GBDTModel


def parse_args():
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--loss', type=str, default='mse')

    # training settings
    parser.add_argument('--num-round', type=int, default=500)
    parser.add_argument('--early-stopping-round', type=int, default=50)

    # data settings
    parser.add_argument('--data', type=str, required=True, help='data source path')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    dataset = prepare_dataset(args.data)

    model = GBDTModel(args)

    train_info = model.train(dataset,
                             train_seg=slice(20150101, 20191231 - 7),
                             valid_seg=slice(20200101, 20201231 - 7),
                             test_seg=slice(20210101, 20220915))

    pred = model.predict(dataset.get_data_split('test'))
    # pred.to_pickle('pred.pkl')
    # print(pred)
    # back-testing
    # print('reading pred.pkl')
    # pred = pd.read_pickle('pred.pkl')
    print(f'Start backtesting ...')
    backtester = BackTest(20210101, 20220915, args.data, [1, 2, 5, 10], 10)
    results = backtester.alpha_backtest(pred, alpha_shifted=False, plot=True)
    print(pd.DataFrame.from_dict(results))

if __name__ == '__main__':
    main()