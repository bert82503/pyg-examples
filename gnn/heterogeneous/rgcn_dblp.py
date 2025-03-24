import copy
# PyG搭建R-GCN实现节点分类 - https://blog.csdn.net/Cyril_KI/article/details/126048682

import os.path as osp

from gnn.gcn_conv import loss_function

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../', 'dataset', 'DBLP')

import torch
from torch_geometric.datasets import DBLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据处理
# 导入数据：
dataset = DBLP(root=path)
graph = dataset[0]
print(graph)
# HeteroData(
#   author={
#     x=[4057, 334],
#     y=[4057],
#     train_mask=[4057],
#     val_mask=[4057],
#     test_mask=[4057],
#   },
#   paper={ x=[14328, 4231] },
#   term={ x=[7723, 50] },
#   conference={ num_nodes=20 },
#   (author, to, paper)={ edge_index=[2, 19645] },
#   (paper, to, author)={ edge_index=[2, 19645] },
#   (paper, to, term)={ edge_index=[2, 85810] },
#   (paper, to, conference)={ edge_index=[2, 14328] },
#   (term, to, paper)={ edge_index=[2, 85810] },
#   (conference, to, paper)={ edge_index=[2, 14328] }
# )
# 可以发现，DBLP数据集中有作者(author)、论文(paper)、术语(term)以及会议(conference)四种类型的节点。
# DBLP中包含14328篇论文(paper)， 4057位作者(author)， 20个会议(conference)， 7723个术语(term)。
# 作者分为四个领域：数据库、数据挖掘、机器学习、信息检索。
# 任务：对author节点进行分类，一共4类。

# 由于conference节点没有特征，因此需要预先设置特征：
graph['conference'].x = torch.randn((graph['conference'].num_nodes, 50))
# 所有conference节点的特征都随机初始化。

# 获取一些有用的数据：
graph = graph.to(device)
# 节点分类数量
num_classes = torch.max(graph['author'].y).item() + 1
train_mask, val_mask, test_mask = graph['author'].train_mask, graph['author'].val_mask, graph['author'].test_mask
y = graph['author'].y

node_types, edge_types = graph.metadata()
num_nodes = graph['author'].x.shape[0]
num_relations = len(edge_types)
init_sizes = [graph[x].x.shape[1] for x in node_types]
# homogeneous_graph = graph.to_homogeneous()
# print(homogeneous_graph)
in_feats, hidden_feats = 128, 64

# print(graph)
# HeteroData(
#   author={
#     x=[4057, 334],
#     y=[4057],
#     train_mask=[4057],
#     val_mask=[4057],
#     test_mask=[4057],
#   },
#   paper={ x=[14328, 4231] },
#   term={ x=[7723, 50] },
#   conference={
#     num_nodes=20,
#     x=[20, 50],
#   },
#   (author, to, paper)={ edge_index=[2, 19645] },
#   (paper, to, author)={ edge_index=[2, 19645] },
#   (paper, to, term)={ edge_index=[2, 85810] },
#   (paper, to, conference)={ edge_index=[2, 14328] },
#   (term, to, paper)={ edge_index=[2, 85810] },
#   (conference, to, paper)={ edge_index=[2, 14328] }
# )

from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import RGCNConv
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

# 模型搭建
class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=30)
        self.lins = torch.nn.ModuleList()
        for i in range(len(node_types)):
            lin = nn.Linear(init_sizes[i], in_channels)
            self.lins.append(lin)

    def trans_dimensions(self, g):
        data = copy.deepcopy(g)
        for node_type, lin in zip(node_types, self.lins):
            data[node_type].x = lin(data[node_type].x)
        return data

    # 1. 前向传播
    def forward(self, data):
        # 转换后的data中所有类型节点的特征维度都为128，然后再将其转为同质图：
        data = self.trans_dimensions(data)
        homogeneous_data = data.to_homogeneous()
        # print(homogeneous_data)
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = x[:num_nodes]
        return x

# model = RGCN(in_feats, hidden_feats, num_classes).to(device)
# 输出一下模型：
# print(model)
# RGCN(
#   (conv1): RGCNConv(128, 64, num_relations=6)
#   (conv2): RGCNConv(64, 4, num_relations=6)
#   (lins): ModuleList(
#     (0): Linear(in_features=334, out_features=128, bias=True)
#     (1): Linear(in_features=4231, out_features=128, bias=True)
#     (2-3): 2 x Linear(in_features=50, out_features=128, bias=True)
#   )
# )

# 3. 训练
# 训练时返回验证集上表现最优的模型：
def train():
    model = RGCN(in_feats, hidden_feats, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
    model.train()
    # for epoch in tqdm(range(100)):
    for epoch in range(100):
        f = model(graph)
        loss = loss_function(f[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # validation
        val_acc, val_loss = test(model, val_mask)
        test_acc, test_loss = test(model, test_mask)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        print('Epoch{:3d} train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'.
              format(epoch, loss.item(), val_acc, test_acc))
        # tqdm.write('Epoch{:3d} train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'.
        #            format(epoch, loss.item(), val_acc, test_acc))
    return final_best_acc

# 4. 测试
@torch.no_grad()
def test(model, mask):
    model.eval()
    out = model(graph)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[mask], y[mask])
    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(test_mask.sum())
    return acc, loss.item()

def main():
    final_best_acc = train()
    print('RGCN Accuracy:', final_best_acc)
    # RGCN Accuracy: 0.921400061406202
    # RGCN Accuracy: 0.9306109917101627

if __name__ == '__main__':
    main()
