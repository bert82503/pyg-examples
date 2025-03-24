
# PyG搭建GCN实现链接预测 - https://blog.csdn.net/Cyril_KI/article/details/125956540
# 链接预测中训练集、验证集以及测试集的划分(以PyG的RandomLinkSplit为例) - https://blog.csdn.net/Cyril_KI/article/details/125952150

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# 1. 数据处理
# CiteSeer网络是一个引文网络，节点为论文，一共3327篇论文。
# 论文一共分为六类：Agents、AI（人工智能）、DB（数据库）、IR（信息检索）、ML（机器语言）和HCI。
# 如果两篇论文间存在引用关系，那么它们之间就存在链接关系。
# 加载数据：
dataset = Planetoid(root=root_path + '/dataset', name='CiteSeer')
# print(dataset[0])
# Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])
# 一共有3327个节点，节点的特征维度为3703，一共9104条边，数据一共两行，每一行都表示节点编号。

# 使用PyG封装的RandomLinkSplit，很容易实现数据集的划分：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])
dataset = Planetoid(root=root_path + '/dataset', name='CiteSeer', transform=transform)
train_data, val_data, test_data = dataset[0]
# 原始数据集、训练集、验证集、测试集
# print(train_data)
# print(val_data)
# print(test_data)
# Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])
# Data(x=[3327, 3703], edge_index=[2, 7284], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327], edge_label=[3642], edge_label_index=[2, 3642])
# Data(x=[3327, 3703], edge_index=[2, 7284], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327], edge_label=[910], edge_label_index=[2, 910])
# Data(x=[3327, 3703], edge_index=[2, 8194], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327], edge_label=[910], edge_label_index=[2, 910])
# print(train_data.edge_label.sum())
# tensor(3642.)
# print(val_data.edge_label.sum())
# tensor(455.)

# 2. GCN链接预测
# 使用GCN对训练集中的节点进行编码，得到节点的向量表示，
# 然后使用这些向量表示对训练集中的正负样本进行有监督学习，具体来讲就是使用节点向量求得样本中节点对的内积，
# 然后与标签求损失，最后反向传播更新参数。

# 2.1 负采样
# 负采样函数：
def negative_sample():
    # 从训练集中采样与正边相同数量的负边
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    # print('neg_edge_index.size(1):', neg_edge_index.size(1))
    # 3642
    # 3642条负边，即每次采样与训练集中正边数量一致地负边
    # 采样后，将neg_edge_index与训练集中原有的正样本train.edge_label_index进行拼接以得到完整的样本集，
    # 同时也需要在原本的train_data.edge_label后面添加指定数量的0用于表示负样本。
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    return edge_label, edge_label_index

# 2.2 模型搭建
# GCN链接预测模型搭建如下：
class GCN_LP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_LP, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print('src.size():', src.size())
        # torch.Size([7284, 64])
        # print('dst.size():', dst.size())
        # torch.Size([7284, 64])
        r = (src * dst).sum(dim=-1)
        # print('r.size():', r.size())
        # torch.Size([7284])
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
# 编码器由一个两层GCN组成，用于得到训练集中节点的向量表示，解码器用于得到训练集中节点对向量间的内积。
# 由前面可知训练集中的正样本数量为3642，经过负采样函数negative_sample得到3642个负样本，一共7284个样本，最终解码器返回7284个节点对间的内积。

# 2.3 模型训练/测试
# 评价指标采用AUC
def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    # 评价指标采用AUC：
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

model = GCN_LP(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
# 损失函数采用BCEWithLogitsLoss，要想弄懂BCEWithLogitsLoss，就要先了解BCELoss。
# BCELoss是一种二元交叉熵损失：
# 而BCEWithLogitsLoss则是在BCELoss的基础上增加了Sigmoid选项，即先把输入经过一个Sigmoid，然后再计算BCELoss。
criterion = torch.nn.BCEWithLogitsLoss().to(device)
# print(model)
# GCN_LP(
#   (conv1): GCNConv(3703, 128)
#   (conv2): GCNConv(128, 64)
# )

def train():
    min_epochs = 10
    best_model = None
    best_val_auc = 0
    final_test_auc = 0
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_auc = test(model, val_data)
        test_auc = test(model, test_data)
        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            # best_model = model
        print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} test_auc {:.4f}'
              .format(epoch, loss.item(), val_auc, test_auc))
        # epoch 093 train_loss 0.44373870 val_auc 0.8889 test_auc 0.8928
    return final_test_auc

final_test_auc = train()
print('final best auc:', final_test_auc)
# final best auc: 0.9159956526989494
# final best auc: 0.8920130419031518
