import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.metrics import f1_score, accuracy_score

from utils import *
from preprocessing import *
from graph import ewTxGraph


class MHGCN(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(MHGCN, self).__init__()
        self.conv10 = GCNConv(num_features, hidden_channels)
        self.conv11 = GCNConv(num_features, hidden_channels)
        self.conv12 = GCNConv(num_features, hidden_channels)
        self.conv13 = GCNConv(num_features, hidden_channels)

        self.conv20 = GCNConv(hidden_channels, hidden_channels)
        self.conv21 = GCNConv(hidden_channels, hidden_channels)
        self.conv22 = GCNConv(hidden_channels, hidden_channels)
        self.conv23 = GCNConv(hidden_channels, hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, 1)
        self.fc2 = nn.Linear(hidden_channels, 1)

        self.fc_final = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, datas):
        x00, edge_index_0 = datas[0].x, datas[0].edge_index
        x01, edge_index_1 = datas[1].x, datas[1].edge_index
        x02, edge_index_2 = datas[2].x, datas[2].edge_index
        x03, edge_index_3 = datas[3].x, datas[3].edge_index

        x10 = torch.relu(self.conv10(x00, edge_index_0))
        x11 = torch.relu(self.conv11(x01, edge_index_1))
        x12 = torch.relu(self.conv12(x02, edge_index_2))
        x13 = torch.relu(self.conv13(x03, edge_index_3))

        x1_stack = torch.stack([x10, x11, x12, x13], dim=0)
        attn_scores = self.fc1(x1_stack)  # 计算注意力分数
        attn_weights = F.softmax(attn_scores, dim=0)  # 使用 softmax 归一化注意力分数，得到注意力系数
        x1 = torch.sum(attn_weights * x1_stack, dim=0)  # 将注意力系数与输入特征进行加权
        # x1 = self.fc1(torch.cat((x10, x11, x12, x13), dim=1))
        # x1 = torch.mean(torch.stack([x10, x11, x12, x13], dim=0), dim=0)
        # x1 = torch.sum(torch.stack([x10, x11, x12, x13], dim=0), dim=0)

        x20 = torch.relu(self.conv20(x10, edge_index_0))
        x21 = torch.relu(self.conv21(x11, edge_index_1))
        x22 = torch.relu(self.conv22(x12, edge_index_2))
        x23 = torch.relu(self.conv23(x13, edge_index_3))

        x2_stack = torch.stack([x20, x21, x22, x23], dim=0)
        attn_scores = self.fc2(x2_stack)
        attn_weights = F.softmax(attn_scores, dim=0)
        x2 = torch.sum(attn_weights * x2_stack, dim=0)
        # x2 = self.fc2(torch.cat((x20, x21, x22, x23), dim=1))
        # x2 = torch.mean(torch.stack([x20, x21, x22, x23], dim=0), dim=0)
        # x2 = torch.sum(torch.stack([x20, x21, x22, x23], dim=0), dim=0)

        x = self.fc_final(torch.cat((x1, x2), dim=1))

        x = F.log_softmax(x, dim=1)

        return x


def ad_mhgcn(datas, hidden_channles=16, num_epochs=150, lr=0.1):
    num_features = datas[0].x.shape[1]
    num_classes = 2
    model = MHGCN(num_features, hidden_channles, num_classes)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        out = model(datas)
        _, pred = out.max(dim=1)
        acc = accuracy_score(pred[datas[0].test_mask], datas[0].y[datas[0].test_mask])
        loss = F.nll_loss(out[datas[0].train_mask], datas[0].y[datas[0].train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')

    # 评估模型
    model.eval()

    _, pred = model(datas).max(dim=1)
    acc = accuracy_score(pred[datas[0].test_mask], datas[0].y[datas[0].test_mask])
    f1 = f1_score(pred[datas[0].test_mask], datas[0].y[datas[0].test_mask])
    print(f"accuracy: {acc:.4f}")
    print(f"f1_score: {f1:.4f}")


if __name__ == '__main__':
    seed = 42
    random_seed(seed)

    datas = []
    for i in range(4):
        datas.append(torch.load(graph_path + 'ewTxGraph_42_topk_0.5_w{}.pth'.format(i)))

    ad_mhgcn(datas, num_epochs=130, lr=5e-3)   # num_epochs=130  lr=5e-3  0.9213 0.9239
