import os
from itertools import combinations, chain

import numpy as np
import pandas as pd
import ray

from preprocessing import read_data, parse_data
from mining import intra_stock, inter_stock


def get_intra_info():
    if os.path.isfile('../cache/intra_info.csv'):
        return pd.read_csv('../cache/intra_info.csv', index_col=0)
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
    dfall = read_data()
    sids, dfall = parse_data(dfall)
    file_names = ['../cache/test%d.csv' % i for i in range(2,7)]
    file_ready = [os.path.isfile(i) for i in file_names]
    if all(file_ready):
        print("Inter-stack data ready")
        return sids, [pd.read_csv('../cache/test%d.csv' % i, index_col=0) for i in range(2,7)]
    else:
        np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
        print("data parsed")
        dfall = dfall[['symbol_%d' % sidx for sidx, _ in sids]]
        print(dfall)
        df_sup_list = inter_stock(dfall, [sidx for sidx, _ in sids])
        return sids, df_sup_list
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
def intra_cluster():
    df_intra = get_intra_info()
    print(df_intra)
    df_intra = df_intra.dropna(axis=1)
    print(df_intra.info())
    data_mat = np.array(df_intra)
    data_mat = StandardScaler().fit_transform(data_mat)
    pca = PCA(n_components=3)
    new_mat = pca.fit_transform(data_mat)
    kmeans = KMeans(n_clusters=4, random_state=1).fit(new_mat)
    print(kmeans.labels_, df_intra.index.to_numpy())
    Scene = dict(xaxis = dict(title='First Component'), yaxis = dict(title='Second Component'),
                 zaxis = dict(title='Third Component'))
    labels = kmeans.labels_
    trace = go.Scatter3d(x=new_mat[:, 0], y=new_mat[:, 1], z=new_mat[:, 2], mode='markers',
                         marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
    layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=800, width=800)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()
if __name__ == '__main__':
    ray.init(num_cpus=8)
    sids, df_sup_list = get_inter_info()
    pow_10 = [1, 10, 100, 1000, 10000, 100000]
    sidxs = [i for i, _ in sids]
    _, parrallel = parse_data(read_data())
    parallel = parrallel[['symbol_%d' % sidx for sidx, _ in sids]].dropna()

    @ray.remote
    def calc_predict_suc_rate(ante, cons):
        n = len(ante) + len(cons)
        col = []
        cons_idx_in_sb_lst = []
        ante_idx_in_sb_lst = []
        for i, _ in sids:
            if i in ante or i in cons:
                col.append(i)
        col = sum(e * pow_10[i] for i, e in enumerate(col))
        sup_list = df_sup_list[n - 2][[str(col)]]
        sup_list = sup_list[sup_list[str(col)] > 0]
        sup_arr = sup_list.loc[:,str(col)].to_numpy()
        symbols_list = [list(map(int, str(s).zfill(n))) for s in sup_list.index]
        stks = list(map(int, str(col).zfill(n)))
        for a in ante:
            for j, s in enumerate(stks):
                if a==s:
                    ante_idx_in_sb_lst.append(j)
        for c in cons:
            for j, s in enumerate(stks):
                if c==s:
                    cons_idx_in_sb_lst.append(j)
        conf_list = pd.DataFrame()
        for sbs, sup in zip(symbols_list, sup_arr):
            sup_sum = 0
            for ot_sbs, ot_sup in zip(symbols_list, sup_arr):
                if all([sbs[i] == ot_sbs[i] for i in ante_idx_in_sb_lst]):
                    sup_sum += ot_sup
            new_conf = sup/sup_sum
            sbs_row_idx = sum([sbs[i]*pow_10[pi] for pi, i in enumerate(ante_idx_in_sb_lst)])
            try:
                if conf_list.loc[sbs_row_idx, 'conf'] < new_conf:
                    conf_list.loc[sbs_row_idx, 'conf'] = new_conf
                    conf_list.loc[sbs_row_idx, 'cons'] = sum([sbs[i]*pow_10[pi] for pi, i in enumerate(cons_idx_in_sb_lst)])
            except KeyError:
                conf_list.loc[sbs_row_idx, 'conf'] = new_conf
                conf_list.loc[sbs_row_idx, 'cons'] = sum([sbs[i]*pow_10[pi] for pi, i in enumerate(cons_idx_in_sb_lst)])

        count = 0
        correct = 0
        for (_, row_ante), (_, row_cons) in \
            zip(parrallel[['symbol_%d' % sidx for sidx in ante]].iterrows(),
                parrallel[['symbol_%d' % sidx for sidx in cons]].iterrows()):
            row_ante_val =  sum([e*pow_10[i] for i, e in enumerate(row_ante)])
            row_cons_val = sum([e * pow_10[i] for i, e in enumerate(row_cons)])
            try:
                if conf_list.loc[row_ante_val, 'cons'] == row_cons_val:
                    correct += 1
                count += 1
            except KeyError:
                pass
        res=0 if count==0 else correct/count
        return res

    df_pred_acc = pd.DataFrame()
    stock_combinations = chain(*map(lambda x: combinations(sidxs, x), range(1, len(sidxs))))
    lst = list(stock_combinations)


    for r in lst:
        for c in lst:
            if  len(r) < len(c) or any(np.isin(np.array(r), np.array(c))):
                continue
            row_name = sum(e*pow_10[i] for i, e in enumerate(r))
            col_name = sum(e*pow_10[i] for i, e in enumerate(c))
            df_pred_acc.loc[row_name, col_name] = calc_predict_suc_rate.remote(r, c)
        print(r)

    df_pred_acc.to_csv("../cache/confidence.csv")
