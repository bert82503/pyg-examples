
# PyG搭建GCN实现节点分类(GCNConv参数详解) - https://blog.csdn.net/Cyril_KI/article/details/123457698

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, NELL
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, num_classes)
        self.norm = torch.nn.BatchNorm1d(32)

    # 节点特征向量、节点状态向量
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_data(name):
    if name == 'NELL':
        print(root_path + '/dataset/' + name + '/')
        dataset = NELL(root=root_path + '/dataset/' + name + '/')
        # CUDA out of memory
        _device = torch.device('cpu')
    else:
        dataset = Planetoid(root=root_path + '/dataset', name=name)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(_device)
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = dataset.num_node_features
    return data, num_node_features, dataset.num_classes

def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        # 训练集
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))

def test(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    # 测试集
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))

def main():
    # names = ['CiteSeer', 'Cora', 'PubMed', 'NELL']
    names = ['CiteSeer', 'Cora', 'PubMed']
    for name in names:
        print(name + '...')
        data, num_node_features, num_classes = load_data(name)
        print(data, num_node_features, num_classes)
        # Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327]) 3703 6
        # _device = 'cpu' if name == 'NELL' else 'cuda'
        _device = 'cpu'
        device = torch.device(_device)
        model = GCN(num_node_features, num_classes).to(device)
        train(model, data, device)
        test(model, data)

if __name__ == '__main__':
    main()

# CiteSeer...
# Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327]) 3703 6
# Epoch 199 loss 0.0024
# GCN Accuracy: 0.5890

# Cora...
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708]) 1433 7
# Epoch 199 loss 0.0011
# GCN Accuracy: 0.7340

# PubMed...
# Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717]) 500 3
# Epoch 199 loss 0.0003
# GCN Accuracy: 0.7340
