from itertools import combinations

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

def inter_stock(traces_all, min_support_ratio=0.03):
    traces_all = traces_all.dropna().astype(int)
    itemset_1 = np.asarray(np.unique(traces_all.to_numpy(), return_counts=True)).T
    itemset_all = []
    # 2nd phase
    min_support = round(min_support_ratio * traces_all.shape[1] * (traces_all.shape[1] - 1) / 2 * traces_all.shape[0])
    print(min_support,traces_all.shape[0])
    itemset_2 = np.zeros(shape=(round(np.power(itemset_1.shape[0], 2)), 3))
    for idx_i, (i, _) in enumerate(itemset_1):
        for idx_j, (j, _) in enumerate(itemset_1):
            itemset_2[idx_i * 3 + idx_j][0] = i
            itemset_2[idx_i * 3 + idx_j][1] = j
            sup_cnt = 0
            for idx_r, row in traces_all.iterrows():
                row = np.asarray(row)
                if is_subsequence((i, j,), row):
                    sup_cnt += 1
            itemset_2[idx_i * 3 + idx_j][2] = sup_cnt
    print(itemset_2)
    itemset_all.append(itemset_2)
    # 3rd phase and later
    itemset_prev = itemset_2
    for w in range(3, traces_all.shape[1] + 1):
        min_support = 600
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

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
    dfall = read_data()
    sids, dfall = parse_data(dfall)
    rule_set = intra_stock(dfall['symbol_857'].dropna().to_numpy(), 857)
    print(rule_set)