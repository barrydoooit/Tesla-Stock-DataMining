import numpy
import pandas as pd
import numpy as np


def intra_stock(traces, sid, win_size=4, min_support=60):
    trace_string = '-'.join([str(t) for t in traces])
    #1st phase
    unique, counts = np.unique(traces, return_counts=True)
    itemset_1 = np.asarray((unique, counts)).T
    itemset_1 = itemset_1[itemset_1[:, -1] >= min_support]
    #2nd phase
    itemset_2 = np.zeros(shape=(round(np.power(itemset_1.shape[0], 2)), 3))
    for idx_i, (i, _) in enumerate(itemset_1):
        for idx_j, (j, _) in enumerate(itemset_1):
            itemset_2[idx_i*3 + idx_j][0] = i
            itemset_2[idx_i*3 + idx_j][1] = j
            itemset_2[idx_i*3 + idx_j][2] = trace_string.count("%d-%d" % (i, j))
    itemset_2 = itemset_2[itemset_2[:, -1] >= min_support]

    #3rd phase and later

    itemset_prev = itemset_2
    for _ in range(3, win_size+1):
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
                    new_entry.append(trace_string.count('-'.join([str(int(t)) for t in new_entry])))
                    if new_entry[-1] >= min_support:
                        itemset_next.append(new_entry)
        itemset_prev = np.asarray(itemset_next)
    print(itemset_next)