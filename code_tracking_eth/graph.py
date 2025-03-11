import torch
from torch_geometric.data import Data


class txGraph(Data):
    def __init__(self, tx, feature, train_test_edges):
        super().__init__()

        nodes_dict = {value: idx for idx, value in enumerate(feature.index.tolist())}

        edges = self.edge_idx_map(nodes_dict, set(zip(tx["From"].tolist(), tx["To"].tolist())))
        train_edges_pos = self.edge_idx_map(nodes_dict, train_test_edges['train_edges_pos'])
        train_edges_neg = self.edge_idx_map(nodes_dict, train_test_edges['train_edges_false'])
        test_edges_pos = self.edge_idx_map(nodes_dict, train_test_edges['test_edges_pos'])
        test_edges_neg = self.edge_idx_map(nodes_dict, train_test_edges['test_edges_false'])

        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        self.train_edge_pos_index = torch.tensor(train_edges_pos, dtype=torch.long).t()
        self.train_edge_neg_index = torch.tensor(train_edges_neg, dtype=torch.long).t()
        self.test_edge_pos_index = torch.tensor(test_edges_pos, dtype=torch.long).t()
        self.test_edge_neg_index = torch.tensor(test_edges_neg, dtype=torch.long).t()

        self.time = torch.tensor(tx["TimeStamp"].tolist(), dtype=torch.long)

        self.x = torch.tensor(feature.values)  # 节点特征
        self.num_nodes = self.x.shape[0]  # 节点数量

    def edge_idx_map(self, nodes_dict, edges):
        return list(map(lambda x: (nodes_dict[x[0]], nodes_dict[x[1]]), list(edges)))


class tempTxGraph(Data):
    def __init__(self, tx_list, feature_list, all_nodes, train_test_edges):
        super().__init__()

        self.temporal_steps = len(tx_list)

        nodes_dict = {value: idx for idx, value in enumerate(all_nodes)}

        # 边索引
        self.edge_indexs = []
        for tx in tx_list:
            edges = self.edge_idx_map(nodes_dict, set(zip(tx["From"].tolist(), tx["To"].tolist())))
            self.edge_indexs.append(torch.tensor(edges, dtype=torch.long).t())

        train_edges_pos = self.edge_idx_map(nodes_dict, train_test_edges['train_edges_pos'])
        train_edges_neg = self.edge_idx_map(nodes_dict, train_test_edges['train_edges_false'])
        test_edges_pos = self.edge_idx_map(nodes_dict, train_test_edges['test_edges_pos'])
        test_edges_neg = self.edge_idx_map(nodes_dict, train_test_edges['test_edges_false'])
        self.train_edge_pos_index = torch.tensor(train_edges_pos, dtype=torch.long).t()
        self.train_edge_neg_index = torch.tensor(train_edges_neg, dtype=torch.long).t()
        self.test_edge_pos_index = torch.tensor(test_edges_pos, dtype=torch.long).t()
        self.test_edge_neg_index = torch.tensor(test_edges_neg, dtype=torch.long).t()

        # 节点特征
        self.x = []
        for feature in feature_list:
            self.x.append(torch.tensor(feature.values))

    def edge_idx_map(self, nodes_dict, edges):

        return list(map(lambda x: (nodes_dict[x[0]], nodes_dict[x[1]]), list(edges)))

