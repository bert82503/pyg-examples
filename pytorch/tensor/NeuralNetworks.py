import torch
import torch.nn as nn

# Building Neural Networks in PyTorch
# 构建神经网络

# In PyTorch, neural networks are built using the torch.nn module, where:
# nn.Linear(in_features, out_features) defines a fully connected (dense) layer.
# Activation functions like torch.relu, torch.sigmoid, or torch.softmax are applied between layers.
# forward() method defines how data moves through the network.

# 神经网络是使用 torch.nn 模块构建的，其中：
# nn.Linear(in_features, out_features) 定义完全连接（密集）层。
# 在层之间应用激活函数，如 torch.relu、torch.sigmoid 或 torch.softmax。
# forward() 方法定义数据如何在网络中移动。

# To build a neural network in PyTorch, we create a class that inherits from torch.nn.Module and defines its layers and forward pass.
# 构建神经网络，我们创建一个从 torch.nn.Module 继承的类并定义其层和前向传递。

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 16)  # First layer
        self.fc2 = nn.Linear(16, 8)   # Second layer
        self.fc3 = nn.Linear(8, 1)    # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 神经网络
model = NeuralNetwork()
# print(model)
# NeuralNetwork(
#   (fc1): Linear(in_features=10, out_features=16, bias=True)
#   (fc2): Linear(in_features=16, out_features=8, bias=True)
#   (fc3): Linear(in_features=8, out_features=1, bias=True)
# )

# 应用
# 5x10矩阵向量
# x = torch.randn(5, 10)
# y = model(x)
# print(y)
# tensor([[0.5370],
#         [0.5463],
#         [0.5262],
#         [0.5413],
#         [0.5358]], grad_fn=<SigmoidBackward0>)
