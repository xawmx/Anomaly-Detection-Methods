import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from data_load import temporal_graph_load, edge_feature_load
from utils import *
from graph import tempTxGraph


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausalConv1d, self).__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        output = self.conv(x)
        return output


class TGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_steps, kernel_size):
        super(TGNN, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        #
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # self.conv1 = GATConv(in_channels, hidden_channels)
        # self.conv2 = GATConv(hidden_channels, hidden_channels)

        self.temporal_conv = CausalConv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size  # 使用当前及前kernel_size - 1个时间步
        )
        # self.temporal_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)

        self.temporal_steps = temporal_steps

    def forward(self, data):
        node_features = []

        # 将所有时间步的节点特征堆叠以进行时间卷积
        temporal_features = torch.stack([data.x[t] for t in range(self.temporal_steps)],
                                        dim=2)  # Shape: [num_nodes, in_channels, temporal_steps]
        # 因果时间卷积增强特征
        enhanced_features = self.temporal_conv(temporal_features)  # Shape: [num_nodes, in_channels, temporal_steps]

        for t in range(self.temporal_steps):
            edge_index = data.edge_indexs[t]
            x = enhanced_features[:, :, t]  # Enhanced node features at time t

            x = torch.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

            node_features.append(x)

        node_features = torch.stack(node_features, dim=1)  # Shape: [num_nodes, temporal_steps, hidden_channels]

        lstm_out, _ = self.lstm(node_features)
        x = lstm_out[:, -1, :]  # Shape: [num_nodes, hidden_channels]

        return x


class TransformerModel(nn.Module):
    def __init__(self, in_channels, num_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()
        # 定义Transformer的编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features):
        # Transformer要求输入维度为 [sequence_length, batch_size, feature_size]
        # 这里需要对输入特征维度进行转换
       # features = features.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(features)
        # 取最后一个时间步的输出（类似于LSTM中取最后时刻的隐藏状态）
        x = transformer_out[-1, :, :]

        return x


class Model(nn.Module):
    def __init__(self, in_channels_n, in_channels_e, hidden_channels_n, temporal_steps, kernel_size):
        super(Model, self).__init__()

        self.se = TGNN(in_channels_n, hidden_channels_n, temporal_steps, kernel_size)

        self.re = TransformerModel(in_channels_e, num_heads=4, num_layers=2)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels_n + in_channels_e, 1),  # 拼接后的输入
            nn.Sigmoid()  # 输出一个0到1之间的值，表示连接的概率
        )

        # self.fc1 = nn.Linear(2 * hidden_channels_n, hidden_channels_n)
        # self.fc2 = nn.Linear(hidden_channels_n, 1)
        # self.fc3 = nn.Linear(hidden_channels_n, 1)

        self.temporal_steps = temporal_steps

    def forward(self, data, edge_features):
        h = self.se(data)
        a = self.re(edge_features)

        return h, a

    def link_predict(self, h_i, h_j, a_ij):
        # # 获得结构特征
        # h_ij = self.fc1(torch.cat((h_i, h_j), dim=1))
        #
        # # 对结构特征与关系特征进行加权融合
        # h_a = torch.stack([h_ij, a_ij], dim=0)
        # attn_scores = self.fc2(h_a)                 # 计算注意力分数
        # attn_weights = F.softmax(attn_scores, dim=0)  # 使用 softmax 归一化注意力分数，得到注意力系数
        # h_a = torch.sum(attn_weights * h_a, dim=0)  # 将注意力系数与输入特征进行加权
        #
        # return torch.sigmoid(self.fc3(h_a)).squeeze()

        embeddings = torch.cat((h_i, h_j, a_ij), dim=1)

        return self.mlp(embeddings).squeeze()


def lp_model(data, train_edges_features, test_edges_features,
             hidden_channels_n=16, num_epochs=100, lr=1e-2, kernel_size=3):
    # 将 NumPy 数组转换为 PyTorch 张量
    train_features = torch.tensor(train_edges_features)  # Shape: [temporal_steps, num_edges, num_features]
    test_features = torch.tensor(test_edges_features)

    num_features_n = data.x[0].shape[1]
    num_features_e = train_features.shape[2]
    model = Model(num_features_n, num_features_e, hidden_channels_n, data.temporal_steps,
                  kernel_size)
    model.double()

    # 训练模型
    print("Model Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # 计算节点嵌入
        h, a = model(data, train_features)

        # 对每条正负边进行预测
        pos_edge_embeddings = [h[data.train_edge_pos_index[0]], h[data.train_edge_pos_index[1]],
                               a[:data.train_edge_pos_index.shape[1]]]
        neg_edge_embeddings = [h[data.train_edge_neg_index[0]], h[data.train_edge_neg_index[1]],
                               a[data.train_edge_pos_index.shape[1]:]]

        # 计算正负边的预测值
        pos_pred = model.link_predict(pos_edge_embeddings[0], pos_edge_embeddings[1], pos_edge_embeddings[2])
        neg_pred = model.link_predict(neg_edge_embeddings[0], neg_edge_embeddings[1], neg_edge_embeddings[2])

        # 计算损失 (交叉熵损失)
        labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(torch.float64)
        pred = torch.cat([pos_pred, neg_pred], dim=0).to(torch.float64)

        loss = F.binary_cross_entropy(pred, labels)

        loss.backward()
        optimizer.step()

        l = labels.detach().numpy(); p = pred.detach().numpy()
        auc = roc_auc_score(l, p)
        f1 = f1_score((p >= 0.5).astype(int), l)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Auc: {auc:.4f}, F1: {f1:.4f}')
    print("\n")

    # 评估模型
    model.eval()

    print("Link Predicting...")

    # 计算节点嵌入
    h, a = model(data, test_features)

    # 对每条正负边进行预测
    pos_edge_embeddings = [h[data.test_edge_pos_index[0]], h[data.test_edge_pos_index[1]],
                           a[:data.test_edge_pos_index.shape[1]]]
    neg_edge_embeddings = [h[data.test_edge_neg_index[0]], h[data.test_edge_neg_index[1]],
                           a[data.test_edge_pos_index.shape[1]:]]

    # 计算正负边的预测值
    pos_pred = model.link_predict(pos_edge_embeddings[0], pos_edge_embeddings[1], pos_edge_embeddings[2])
    neg_pred = model.link_predict(neg_edge_embeddings[0], neg_edge_embeddings[1], neg_edge_embeddings[2])

    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).int().numpy()
    pred = torch.cat([pos_pred, neg_pred], dim=0).detach().numpy()

    auc = roc_auc_score(labels, pred)
    ap = average_precision_score(labels, pred)
    print(f"auc: {auc:.4f}")
    print(f"ap: {ap:.4f}")

    pred = (pred >= 0.5).astype(int)

    acc = accuracy_score(pred, labels)
    f1 = f1_score(pred, labels)
    print(f"f1_score: {f1:.4f}")
    print(f"accuracy: {acc:.4f}")


if __name__ == '__main__':
    seed = 42
    random_seed(seed)

    # 数据读取
    temporal_steps = 8
    graph_id = 3
    emb_ratio = 0.7
    overlap_ratio = 0.1

    data = temporal_graph_load(graph_path, temporal_steps=temporal_steps, graph_id=graph_id, emb_ratio=emb_ratio,
                               overlap_ratio=overlap_ratio)
    train_edges_features, train_edges_labels, test_edges_features, test_edges_labels = (
        edge_feature_load(data_path, temporal_steps=temporal_steps, graph_id=graph_id, emb_ratio=emb_ratio,
                          overlap_ratio=overlap_ratio))

    # 方法测试
    lp_model(data, train_edges_features, test_edges_features, num_epochs=200, kernel_size=4, lr=1e-2)

    # Input: Temporal Graph with node features, edge features
    # GraphSAGE+TE for node embedding, Casual Conv for node feature enhancement between snapshots, TE for edge embedding
    # MLP for LP

    # graph_id = 1, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 2, num_epochs = 120, lr = 1e-2, LP = MLP
    # auc = 0.9933, ap = 0.9926, f1_score = 0.9662, accuracy = 0.9659

    # graph_id = 2, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 3, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9939, ap = 0.9930, f1_score = 0.9658, accuracy = 0.9657

    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0, kernel_size = 5, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9953, ap = 0.9957, f1_score = 0.9721, accuracy = 0.9722
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0, kernel_size = 4, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9942, ap = 0.9948, f1_score = 0.9727, accuracy = 0.9729
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0, kernel_size = 3, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9944, ap = 0.9950, f1_score = 0.9734, accuracy = 0.9735
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0, kernel_size = 2, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9948, ap = 0.9952, f1_score = 0.9706, accuracy = 0.9707

    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 6, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9952, ap = 0.9955, f1_score = 0.9742, accuracy = 0.9743
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 5, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9957, ap = 0.9961, f1_score = 0.9754, accuracy = 0.9756
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 4, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9956, ap = 0.9960, f1_score = 0.9775, accuracy = 0.9776
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 3, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9953, ap = 0.9957, f1_score = 0.9765, accuracy = 0.9767
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.1, kernel_size = 2, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9959, ap = 0.9963, f1_score = 0.9738, accuracy = 0.9741

    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.3, kernel_size = 5, num_epochs = 200, lr = 1e-2, LP = MLP
    # auc = 0.9938, ap = 0.9945, f1_score = 0.9719, accuracy = 0.9720
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.3, kernel_size = 4, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9938, ap = 0.9944, f1_score = 0.9703, accuracy = 0.9705
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.3, kernel_size = 3, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9939, ap = 0.9944, f1_score = 0.9683, accuracy = 0.9686
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.3, kernel_size = 2, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9941, ap = 0.9947, f1_score = 0.9709, accuracy = 0.9711

    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.5, kernel_size = 5, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9942, ap = 0.9948, f1_score = 0.9675, accuracy = 0.9675
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.5, kernel_size = 4, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9941, ap = 0.9949, f1_score = 0.9724, accuracy = 0.9726
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.5, kernel_size = 3, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9949, ap = 0.9955, f1_score = 0.9719, accuracy = 0.9722
    # graph_id = 3, emb_ratio = 0.7, temporal_steps = 8, overlap_ratio = 0.5, kernel_size = 2, num_epochs = 180, lr = 1e-2, LP = MLP
    # auc = 0.9939, ap = 0.9946, f1_score = 0.9712, accuracy = 0.9714







