
# 利用节点特征进行节点分类 - https://ifwind.github.io/2021/06/20/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8A%A11-%E5%88%A9%E7%94%A8%E8%8A%82%E7%82%B9%E7%89%B9%E5%BE%81%E8%BF%9B%E8%A1%8C%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB/

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from torch_geometric.datasets import Planetoid

# Load the dataset
# 加载数据
# 包括数据集的下载，若root路径存在数据集，则直接加载数据集
# dataset = Planetoid(root=root_path + '/dataset', name='Cora')
dataset = Planetoid(root=root_path + '/dataset', name='Citeseer')
# print(len(dataset))
# 1
# 该数据集只有一个图
data = dataset[0]
# print(data)
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
# 2708 篇科学出版物（节点），节点的初始特征向量（data.x，维度为[2708,1433]）
# 该词典由 1433 个独特的词组成，相对于一个one hot编码的词袋向量


# 实验
# 节点分类任务是根据已知类别标签的节点和节点特征的映射，对未知类别标签节点进行类别标签标注。

# 基于GCNConv的模型
# 设计这个GCN网络为两个GCNConv层、一个ReLU非线性层和一个dropout操作。
# 第一个GCNConv层将1433维的特征向量嵌入（embedding）到低维空间中（hidden_channels=16），经过ReLU层激活，再经过dropout操作，
# 输入第二个GCNConv层——将低维节点表征嵌入到类别空间中（num_classes=7）。
# 值得注意的是，在forward()函数中输出的是节点特征，维度为[2708,7]，而不是输出经softmax层的分类概率。

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv

class GCN(torch.nn.Module):
    # 初始化
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    # 前向传播
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # 注意这里输出的是节点的特征，维度为[节点数, 类别数]
        return x
# 实例化模型
# model = GCN(dataset.num_features, 16, dataset.num_classes)
# print(model)
# Cora
# GCN(
#   (conv1): GCNConv(1433, 16)
#   (conv2): GCNConv(16, 7)
# )
# Citeseer
# GCN(
#   (conv1): GCNConv(3703, 16)
#   (conv2): GCNConv(16, 6)
# )


# 基于TransformerConv的模型
# 特别地，通过更换卷积层，我们可以得到基于其他卷积层的模型，卷积层API可参考：torch_geometric.nn-convolutional-layers
class Transformer(torch.nn.Module):
    # 初始化
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Transformer, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = TransformerConv(in_channels, hidden_channels, dropout=0.5)
        self.conv2 = TransformerConv(hidden_channels, out_channels, dropout=0.5)

    # 前向传播
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # 注意这里输出的是节点的特征，维度为[节点数, 类别数]
        return x
# 实例化模型
model = Transformer(dataset.num_features, 16, dataset.num_classes)
print(model)
# Cora
# Transformer(
#   (conv1): TransformerConv(1433, 16, heads=1)
#   (conv2): TransformerConv(16, 7, heads=1)
# )
# Citeseer
# Transformer(
#   (conv1): TransformerConv(3703, 16, heads=1)
#   (conv2): TransformerConv(16, 6, heads=1)
# )

# 选择优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# 选择loss function损失函数
# 选择交叉熵(CrossEntropy)作为loss function，其他loss function见torch.nn-loss function，
# 如何选择loss function参考深度学习中常见的激活函数与损失函数的选择与介绍：
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
# 训练函数
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # 梯度置零
    out = model(data.x, data.edge_index)  # 模型前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 优化器梯度下降
    return loss
# 训练
for epoch in range(200):
    loss = train(model, data, optimizer, criterion)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

# 测试
# model.eval()开启模型的测试模式，利用训练好模型中的各层权重矩阵聚合各层邻居节点的消息，预测目标结点的特征。
# 测试函数
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 使用最大概率的类别作为预测结果
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())  # 获取正确标记的节点
    acc = correct / int(data.test_mask.sum())  # 计算正确率
    return acc
# 精度评价
test_acc = test(model, data)
print(f'Test Accuracy: {test_acc:.4f}')
# Cora
# GCN
# Epoch: 200, Loss: 0.0190
# Test Accuracy: 0.8100
# Transformer
# Epoch: 200, Loss: 0.0430
# Test Accuracy: 0.7720
# Citeseer
# GCN
# Epoch: 200, Loss: 0.0406
# Test Accuracy: 0.6920
# Transformer
# Epoch: 200, Loss: 0.1468
# Test Accuracy: 0.6740

from util.visualize import visualize

# 可视化
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
