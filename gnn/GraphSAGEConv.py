import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

# Load the dataset
dataset = Planetoid(root='../dataset', name='Cora')
data = dataset[0]

# 分割数据
# 为了训练和验证，数据集被分成70%用于训练和30%用于测试，前15,728个节点用于训练，最后6,742个节点用于测试集。
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
# >>> Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node feature matrix
        # edge_index: Graph connectivity matrix
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the model
model = GraphSAGE(dataset.num_features, 16, dataset.num_classes)
print('>>>', model)
# >>> GraphSAGE(
#   (conv1): SAGEConv(1433, 16, aggr=mean)
#   (conv2): SAGEConv(16, 7, aggr=mean)
# )

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# Test the model
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum())
    test_acc = test_correct / int(data.test_mask.sum())
    return test_acc

# Run training and testing
# for epoch in range(200):
for epoch in range(20):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    # Epoch: 200, Loss: 0.0005, Accuracy: 0.8462
    # Epoch: 200, Loss: 0.0005, Accuracy: 0.8647
