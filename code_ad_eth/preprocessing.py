import ast

import pandas as pd


# 数据读入
def data_load(path="dataset"):
    df_account_stats = pd.read_csv(path + "\\account_stats.csv",  index_col=0)
    df_account_stats_normalized = account_stats_normalize(df_account_stats)

    file_label = path + "\\label.txt"
    df_label = pd.read_csv(file_label, delimiter=' ', names=['Idx', 'Label'])

    df_tx = pd.read_csv(path + "\\TransEdgelist.txt", delimiter=',', names=['From', 'To', 'Value', 'TimeStamp']).sort_values(
        by='TimeStamp').reset_index(drop=True)
    df_tx_stats = pd.read_csv(path + "\\tx_stats.csv", index_col=0)
    df_tx_stats["Node_pair"] = df_tx_stats["Node_pair"].apply(ast.literal_eval)

    return df_account_stats_normalized, df_tx_stats, df_tx, df_label


# 数据归一化
def account_stats_normalize(df_account_stats_dropped):
    label = df_account_stats_dropped['label']

    df_account_stats_normalized = (df_account_stats_dropped - df_account_stats_dropped.mean()) / df_account_stats_dropped.std(ddof=0)
    df_account_stats_normalized["label"] = label

    return df_account_stats_normalized


if __name__ == '__main__':
    data_load()


