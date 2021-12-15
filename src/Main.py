import os

import numpy as np
import pandas as pd

from preprocessing import read_data, parse_data
from mining import intra_stock, inter_stock


def get_intra_info():
    if os.path.isfile('../cache/intra_info.csv'):
        return pd.read_csv('../cache/intra_info.csv')
    else:
        np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
        dfall = read_data()
        sids, dfall = parse_data(dfall)
        print(dfall.info())
        df_intra_info = pd.DataFrame(index=sids)
        for sid in sids:
            rule_set = intra_stock(dfall['symbol_%d' % sid].dropna().to_numpy(), sid)
            print(rule_set)
            for rules_of_len in rule_set:
                rule_len = len(rules_of_len[0]) - 2
                for rule in rules_of_len:
                    rule_name = '-'.join(str(round(i)) for i in rule[:rule_len - 1]) + '=>' + str(round(rule[rule_len - 1]))
                    df_intra_info.loc[sid, rule_name + ' (Support)'] = rule[-2] / dfall['symbol_%d' % sid].dropna().size
                    df_intra_info.loc[sid, rule_name + ' (Confidence)'] = rule[-1]
        df_intra_info.to_csv('../cache/intra_info.csv')
        return df_intra_info

def get_inter_info():
    if os.path.isfile('../cache/inter_info.csv'):
        return pd.read_csv('../cache/inter_info.csv')
    else:
        np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
        dfall = read_data()
        sids, dfall = parse_data(dfall)
        dfall = dfall[['symbol_%d' % sid for sid in sids]]
        inter_stock(dfall, sids)
if __name__ == '__main__':
    df_intra = get_inter_info()


