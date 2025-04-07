
# PyG搭建GCN实现节点分类(GCNConv参数详解) - https://blog.csdn.net/Cyril_KI/article/details/123457698

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load the dataset
# dataset = Planetoid(root=root_path + '/dataset', name='Cora')
dataset = Planetoid(root=root_path + '/dataset', name='CiteSeer')
data = dataset[0]

# 分割数据
# 为了训练和验证，数据集被分成70%用于训练和30%用于测试。
# Calculate no. of train nodes
num_nodes = data.num_nodes
train_percentage = 0.7
num_train_nodes = int(train_percentage * num_nodes)
# Create a boolean mask for train mask
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[: num_train_nodes] = True
# Add train mask to data object
data.train_mask = train_mask
# Create a boolean mask for test mask
test_mask = ~data.train_mask
data.test_mask = test_mask
# 使用mask来标识训练和验证集
print('>>>', data)
# Cora
# >>> Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
# CiteSeer
# >>> Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.norm = torch.nn.BatchNorm1d(hidden_channels)

    # 1. 前向传播
    # def forward(self, x, edge_index):
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        # 此时的x一共3327行，每一行表示一个节点经过第一层卷积更新后的状态向量。
        # tensor([[ 0.0146,  0.1161,  0.0268,  ...,  0.0149,  0.1371,  0.0287],
        # ...,
        # grad_fn=<AddBackward0>)
        # torch.Size([3327, 16])
        x = self.norm(x)
        # tensor([[-0.7357, -0.6132,  1.3611,  ..., -0.4276,  0.0101,  0.1687],
        # grad_fn=<NativeBatchNormBackward0>)
        x = F.relu(x)
        # tensor([[0.0000, 0.0000, 1.0837,  ..., 0.0000, 0.0000, 0.0528],
        # grad_fn=<ReluBackward0>)
        x = F.dropout(x, training=self.training)
        # print(x)
        # print(x.size())
        # tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0773, 0.0000, 0.0000],
        # grad_fn=<MulBackward0>)
        # torch.Size([3327, 16])
        x = self.conv2(x, edge_index)
        # print(x)
        # print(x.size())
        # tensor([[-1.1116, -0.2082,  1.6402, -0.0081,  1.7150, -0.6582],
        # grad_fn=<AddBackward0>)
        # torch.Size([3327, 6])
        # 每个节点的维度为6的状态向量，表示各个类别的概率。
        return x

# Instantiate the model
model = GCN(dataset.num_node_features, 16, dataset.num_classes)
print('>>>', model)
# CiteSeer
# >>> GCN(
#   (conv1): GCNConv(3703, 32)
#   (conv2): GCNConv(32, 6)
#   (norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Cora
# >>> GCN(
#   (conv1): GCNConv(1433, 32)
#   (conv2): GCNConv(32, 7)
#   (norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_function = torch.nn.CrossEntropyLoss().to(device)

# 3. 训练
# Train the model
def train():
    model.train()
    optimizer.zero_grad()  # Reset gradients
    # 在训练时，我们首先利用前向传播计算出输出：
    out = model(data.x, data.edge_index)  # Forward pass
    # 2. 反向传播
    # out即为最终得到的每个节点的6个概率值，但在实际训练中，我们只需要计算出训练集的损失，所以损失函数这样写：
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])  # Compute loss
    # 然后计算梯度，反向更新！
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    return loss

# 4. 测试
# Test the model
@torch.no_grad()
def test():
    model.eval()
    # 首先需要算出模型对所有节点的预测值：
    out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)
    # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum())
    # acc = correct / int(data.test_mask.sum())
    # 此时得到的是每个节点的6个概率值，我们需要在每一行上取其最大值：
    _, pred = out.max(dim=1)
    # 计算预测精度：
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

import hiddenlayer as hl
import time

# A History object to store metrics
history1 = hl.History()
# A Canvas object to draw the metrics
canvas1 = hl.Canvas()
# New history and canvas objects
canvas2 = hl.Canvas()

# Run training and testing
for epoch in range(200):
# for epoch in range(20):
    loss = train()
    accuracy = test()
    print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    # Cora
    # Epoch: 200, Loss: 0.0352, Accuracy: 0.8339
    # CiteSeer
    # Epoch: 200, Loss: 0.1102, Accuracy: 0.7257

    # Log metrics and display them at certain intervals
    # if epoch % 10 == 0:
    #     # Store metrics in the history object
    #     history1.log(epoch, loss=loss, accuracy=accuracy)
    history1.log(epoch, loss=loss, accuracy=accuracy)

# Plot the two metrics in one graph
canvas1.draw_plot([history1["loss"]], labels=["Loss"])
canvas2.draw_plot([history1["accuracy"]], labels=["Accuracy"])
