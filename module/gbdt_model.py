import lightgbm as lgb
import numpy as np
import pandas as pd

from module.data import Dataset


class GBDTModel:

    def __init__(self, args) -> None:
        self.args = args
        self.model = None

        self.param = {
            'num_leaves': args.num_leaves,
            'loss': args.loss,
        }

    def train(self, dataset: Dataset, train_seg: slice, valid_seg: slice, test_seg: slice = None):
        dataset.prepare({
            'train': train_seg,
            'valid': valid_seg,
            'test': test_seg
        })
        train_df = dataset.get_data_split('train')
        valid_df = dataset.get_data_split('valid')

        train_dataset = lgb.Dataset(train_df['feature'].to_numpy(), 
                                    label=train_df['label'].to_numpy().squeeze())
        valid_dataset = lgb.Dataset(valid_df['feature'].to_numpy(), 
                                    label=valid_df['label'].to_numpy().squeeze())

        train_info = {}
        callbacks = [
            lgb.log_evaluation(period=10),
            lgb.record_evaluation(eval_result=train_info),
            lgb.early_stopping(self.args.early_stopping_round)
        ]

        self.model = lgb.train(self.param, train_dataset, num_boost_round=self.args.num_round,
                        valid_sets=[train_dataset, valid_dataset], callbacks=callbacks)
        return train_info

    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise RuntimeError('Models need to be trained before doing prediction')
        pred = self.model.predict(data['feature'].to_numpy(), num_iteration=self.model.best_iteration)
        pred_series = pd.Series(pred, index=data.index)
        return pred_series