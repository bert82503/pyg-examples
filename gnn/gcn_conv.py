
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
dataset = Planetoid(root=root_path + '/dataset', name='Cora')
# dataset = Planetoid(root='../dataset', name='CiteSeer')
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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm = torch.nn.BatchNorm1d(hidden_channels)

    # def forward(self, x, edge_index):
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model
model = GCN(dataset.num_features, 32, dataset.num_classes)
print('>>>', model)
# >>> GCN(
#   (conv1): GCNConv(1433, 32)
#   (conv2): GCNConv(32, 7)
#   (norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_function = torch.nn.CrossEntropyLoss()

# Train the model
def train():
    model.train()
    out = model(data.x, data.edge_index)
    optimizer.zero_grad()
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Test the model
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)
    # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum())
    # acc = correct / int(data.test_mask.sum())
    _, pred = out.max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

# Run training and testing
for epoch in range(200):
# for epoch in range(20):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    # Cora
    # Epoch: 200, Loss: 0.0352, Accuracy: 0.8339
    # CiteSeer
    # Epoch: 200, Loss: 0.1102, Accuracy: 0.7257
