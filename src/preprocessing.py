from enum import IntEnum
import pandas as pd
import numpy as np


class Symbol(IntEnum):
    UP = 0,
    DOWN = 1,
    LEVEL = 2


def read_data(file_name='../data/stacked_data.csv', drop_columns=("open", "high", "low", "volume",)):
    dfall = pd.read_csv(file_name).drop(columns=list(drop_columns))
    return dfall


def parse_data(dfall):
    sids = dfall.stock_id.unique()
    df_parallel = pd.DataFrame()
    df_parallel['tdate'] = dfall[dfall.stock_id == 13]['tdate']
    df_parallel = df_parallel.iloc[1:, :]
    def add_symbol(stock_id, threshold=0.01):
        df_one = dfall[dfall.stock_id == sid].drop(columns=['stock_id'])
        close_one = df_one.close.to_numpy()
        close_diff_one = np.diff(close_one)
        symbols = []
        for diff, pre_val in zip(close_diff_one, close_one[:-1]):
            vrc = diff/pre_val
            if abs(vrc) < threshold:
                symbols.append(Symbol.LEVEL)
            elif vrc > 0:
                symbols.append(Symbol.UP)
            else:
                symbols.append(Symbol.DOWN)
        df_one = df_one.iloc[1:,:]
        df_one = df_one.rename(columns={'close': 'close_%d' % stock_id})
        df_one['symbol_%d' % stock_id] = symbols
        return df_one
    for sid in sids:
        df_parallel = pd.merge(df_parallel, add_symbol(sid), how='left',
                               on=['tdate'])
    df_parallel.to_csv('../cache/parallel.csv')
    return sids, df_parallel


if __name__ == '__main__':
    dfall = read_data()
    dfall = parse_data(dfall)
