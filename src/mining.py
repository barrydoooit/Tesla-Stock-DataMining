from itertools import combinations, combinations_with_replacement, product

import numpy
import pandas as pd
import numpy as np

from src.Utils import count_occurrence,  is_subsequence
from src.preprocessing import Symbol, read_data, parse_data


def intra_stock(traces, sid, win_size=5, min_support_ratio=0.03):
    trace_string = '-'.join([str(round(t)) for t in traces])
    min_support = round(min_support_ratio*traces.size)
    print(trace_string)
    itemset_all = []

    # 1st phase
    unique, counts = np.unique(traces, return_counts=True)
    itemset_1 = np.asarray((unique, counts)).T
    itemset_1 = itemset_1[itemset_1[:, -1] >= min_support]

    # 2nd phase
    min_support = round(min_support_ratio*(traces.size-1))
    itemset_2 = np.zeros(shape=(round(np.power(itemset_1.shape[0], 2)), 3))
    for idx_i, (i, _) in enumerate(itemset_1):
        for idx_j, (j, _) in enumerate(itemset_1):
            itemset_2[idx_i * 3 + idx_j][0] = i
            itemset_2[idx_i * 3 + idx_j][1] = j
            print("%d-%d" % (i, j))
            itemset_2[idx_i * 3 + idx_j][2] = count_occurrence(trace_string, "%d-%d" % (i, j))
            print(i, j, count_occurrence(trace_string, "%d-%d" % (i, j)))
    itemset_2 = itemset_2[itemset_2[:, -1] >= min_support]
    itemset_all.append(itemset_2)

    # 3rd phase and later
    itemset_prev = itemset_2
    for w in range(3, win_size + 1):
        min_support = round(min_support_ratio * (traces.size-w+1))
        itemset_next = []
        for idx_i, row_i in enumerate(itemset_prev):
            for idx_j, row_j in enumerate(itemset_prev):
                rightmost_i = row_i[1:-1]
                rightmost_j = row_j[1:-1]
                if (rightmost_i == rightmost_j).all():
                    left_i = row_i[0]
                    left_j = row_j[0]
                    new_entry = [left_i, left_j] if left_i < left_j else [left_j, left_i]
                    for r in rightmost_i:
                        new_entry.append(r)
                    do_add = True
                    for row in itemset_next:
                        if np.asarray([a == b for a, b in zip(new_entry, row[:-1])]).all():
                            do_add = False
                            break

                    if not do_add:
                        continue
                    prune = False
                    sub_seqs = [np.asarray(s) for s in combinations(new_entry, w - 1)]
                    for sub_seq in sub_seqs:
                        found = False
                        for row in itemset_prev:
                            if (row[:-1] == sub_seq).all():
                                found = True
                                break
                        if not found:
                            prune = True
                            break
                    if prune:
                        continue
                    new_entry.append(count_occurrence(trace_string, '-'.join([str(int(t)) for t in new_entry])))
                    if new_entry[-1] >= min_support:
                        itemset_next.append(new_entry)
        if len(itemset_next) == 0:
            break
        itemset_prev = np.asarray(itemset_next)
        itemset_all.append(itemset_prev)
    # Confidence
    ruleset_all = []
    for idx, itemset in enumerate(itemset_all):
        ruleset_all.append(np.c_[itemset, np.zeros(itemset.shape[0])])
    for idx, ruleset in enumerate(ruleset_all):
        for e_idx, entry in enumerate(ruleset):
            rule = np.array(entry[:-2])
            r_sum = sum([count_occurrence(trace_string,
                                          '-'.join([str(int(t)) for t in np.concatenate((rule[:-1], np.array([sbl])))]))
                         for sbl in list(map(int, Symbol))])
            ruleset[e_idx, -1] = entry[-2]/r_sum
    return ruleset_all

def inter_stock(traces_all, sids, min_support_ratio=0.03):
    traces_all = traces_all.dropna().astype(int)
    unique_vals = np.unique(traces_all.to_numpy())
    itemset_1 = np.asarray(np.unique(traces_all.to_numpy(), return_counts=True)).T
    itemset_all = []
    # 2nd phase

    def calc_sup(vals, stks):
        sup_cnt = 0
        traces_sub = traces_all[['symbol_%d' % s for s in stks]]
        for idx, row in traces_sub.iterrows():
            row = np.asarray(row)
            for r, v in zip(row, vals):
                if r != v:
                    break
            else:
                sup_cnt += 1
        return sup_cnt

    def can_ignore(vals, stks, last_sup_df):
        cbs = list(combinations([(s,v) for s, v in zip(vals, stks)], len(stks)-1))
        for cb in cbs:
            cbarr = np.asarray(cb)
            try:
                if last_sup_df.at['-'.join(str(round(x)) for x in cbarr[:,0]),'-'.join(str(round(x)) for x in cbarr[:,1])] == 0:
                    return True
            except KeyError:
                return True
        return False
    n=2
    min_sup = 30
    df_list = []
    val_permutations = list(product(unique_vals, repeat=n))
    stock_combinations = list(combinations(sids, r=n))
    df = pd.DataFrame()
    for p in val_permutations:
        for c in stock_combinations:
            sup = calc_sup(p, c)
            df.loc['-'.join(str(round(x)) for x in p), '-'.join(str(round(x)) for x in c)] = 0 if sup < min_sup else sup
    df.to_csv('../cache/test1.csv')
    df_list.append(df)
    for n in range(3, traces_all.shape[1]+1):
        val_permutations = list(product(unique_vals, repeat=n))
        stock_combinations = list(combinations(sids, r=n))
        df2 = pd.DataFrame()
        for p in val_permutations:
            for c in stock_combinations:
                if can_ignore(p, c, df_list[-1]):
                    continue
                sup = calc_sup(p, c)
                df2.loc['-'.join(str(round(x)) for x in p), '-'.join(str(round(x)) for x in c)] = 0 if sup < min_sup else sup
        df2.fillna(0)
        df2.to_csv('../cache/test.csv')
        df_list.append(df2)
# 最后组成的样本可使样本之间的相关系数降至最低。
# 这样保证在一定的风险下使收益更高。
if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
    dfall = read_data()
    sids, dfall = parse_data(dfall)
    rule_set = intra_stock(dfall['symbol_857'].dropna().to_numpy(), 857)
    print(rule_set)