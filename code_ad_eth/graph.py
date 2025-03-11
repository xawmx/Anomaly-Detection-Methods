import math

import torch
from tqdm import tqdm

from torch_geometric.data import Data

from sklearn.model_selection import train_test_split

from utils import *
from preprocessing import *


class ewTxGraph(Data):
    def __init__(self, tx, node_feature, seed=42, sample_method='topk', k=0.5, agg_method='mean'):
        super().__init__()
        self.sample_method = sample_method
        self.k = k
        self.agg_method = agg_method

        self.x = torch.tensor(node_feature.drop(["label"], axis=1).values, dtype=torch.float)  # 节点特征
        self.y = torch.tensor(node_feature["label"].values)  # 节点标签
        self.num_nodes = self.y.shape[0]  # 节点数量

        # 创建训练和测试集的节点掩码
        idxs_train, idxs_test = train_test_split(torch.nonzero(self.y != -1).squeeze().numpy(), test_size=0.2,
                                                 random_state=seed)
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.train_mask[idxs_train] = True
        self.test_mask[idxs_test] = True

        # 基于相关性系数为每个节点采样邻居节点
        nodes_dict = {value: idx for idx, value in enumerate(node_feature.index.tolist())}
        edge_set = tx["Node_pair"].map(lambda x: (nodes_dict[x[0]], nodes_dict[x[1]])).tolist()
        edge_feature = tx[["Sum_amount", "Txs", "TimeSpan", "Avg_amount"]].values

        sampled_edges = set()
        for node in tqdm(range(node_feature.shape[0]), desc='Sampling edges'):
            sampled_edge_indices = self.sample_edges_for_node(node, edge_set, edge_feature)
            for i in sampled_edge_indices:
                edge = (edge_set[i][0] if edge_set[i][0] != node else edge_set[i][1], node)
                if edge not in sampled_edges:
                    sampled_edges.add(edge)

        # 获得采样后的边集
        self.edge_index = torch.tensor(list(sampled_edges), dtype=torch.long).t()


    def compute_edge_weights(self, connected_features):
        w = []
        for i, feature in connected_features:
            w.append(feature)
        w = np.array(w)

        # w = np.average((w / w.sum(axis=0)), weights=np.array([0.4, 0.2, 0.1, 0.3]), axis=1).tolist()
        if self.agg_method == 'mean':
            w = (w / w.sum(axis=0)).mean(axis=1).tolist()
        elif self.agg_method == 'sum':
            w = (w / w.sum(axis=0)).sum(axis=1).tolist()
        elif self.agg_method == 'max':
            w = (w / w.sum(axis=0)).max(axis=1).tolist()
        elif self.agg_method in ['w0', 'w1', 'w2', 'w3']:
            n = int(self.agg_method[1:])
            w = (w / w.sum(axis=0))[:, n].tolist()

        connected_weights = []
        for i in range(len(connected_features)):
            connected_weights.append((connected_features[i][0], w[i]))

        return connected_weights

    def sample_edges_for_node(self, node, edges, edge_feature):
        # 找到与该节点相连的边
        connected_edges = [(i, (u, v)) for i, (u, v) in enumerate(edges) if u == node or v == node]
        # 提取权重
        connected_features = [(i, edge_feature[i]) for i, (u, v) in connected_edges]
        connected_weights = self.compute_edge_weights(connected_features)

        if self.sample_method == 'topk':
            # 选择权重最大的前 K * len(edges) 条边
            connected_weights = sorted(connected_weights, key=lambda x: x[1], reverse=True)[
                                :math.ceil(self.k * len(connected_weights))]
        elif self.sample_method == 'threshold':
            # 选择权重大于阈值的边
            threshold = self.k  # 可以根据需要调整阈值
            connected_weights = [(i, w) for i, w in connected_weights if w >= threshold]

        # 返回采样到的边索引
        sampled_edge_indices = [i for i, w in connected_weights]
        return sampled_edge_indices


if __name__ == '__main__':
    seed = 42
    random_seed(seed)

    df_account_stats_normalized, df_tx_stats, _, _ = data_load()

    for w in ['w0', 'w1', 'w2', 'w3']:
        data = ewTxGraph(df_tx_stats, df_account_stats_normalized, seed, agg_method=w)
        torch.save(data, graph_path + f'ewTxGraph_42_topk_0.5_{w}_1.pth')