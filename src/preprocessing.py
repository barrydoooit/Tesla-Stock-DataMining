import os
from enum import IntEnum
import pandas as pd
import numpy as np
import seaborn as sns

class Symbol(IntEnum):
    UP = 0,
    DOWN = 1,
    LEVEL = 2


def read_data(file_name='../data/stacked_data.csv', drop_columns=("open", "high", "low", "volume",)):
    dfall = pd.read_csv(file_name).drop(columns=list(drop_columns))
    return dfall


def parse_data(dfall):
    sids = [(i, j) for i, j in enumerate(dfall.stock_id.unique())]
    if os.path.isfile('../cache/parallel.csv'):
        return sids, pd.read_csv('../cache/parallel.csv', index_col=0)
    df_parallel = pd.DataFrame()
    df_parallel['tdate'] = dfall[dfall.stock_id == 13]['tdate']
    df_parallel = df_parallel.iloc[1:, :]
    def add_symbol(stock_idx, stock_id, threshold=0.01):
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
        df_one = df_one.rename(columns={'close': 'close_%d' % stock_idx})
        df_one['symbol_%d' % stock_idx] = symbols
        return df_one
    for idx, sid in sids:
        df_parallel = pd.merge(df_parallel, add_symbol(idx, sid), how='left',
                               on=['tdate'])
    df_parallel.to_csv('../cache/parallel.csv')

    return sids, df_parallel


if __name__ == '__main__':
    dfall = read_data()
    sids, dfall = parse_data(dfall)
    import matplotlib.pyplot as plt

    dfall = dfall.set_index('tdate')
    dfall1 = dfall.drop(columns=['close_%d' % sidx for sidx, _ in sids])\
        .rename({'symbol_%d' % i: e for i, e in sids}, axis=1)
    dfall2 = dfall.drop(columns=['symbol_%d' % sidx for sidx, _ in sids]).pct_change() \
        .rename({'close_%d' % i: e for i, e in sids}, axis=1)
    """plt.matshow(dfall1.corr())
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()"""
    sns.heatmap(
        dfall1.corr(),
        vmin=0, vmax=1, center=0.2,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    plt.title("Inter-Stock Movement Trend Correlation (Qualitative)", fontsize=15)
    plt.show()