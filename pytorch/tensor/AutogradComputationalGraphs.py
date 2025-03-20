import torch

# Autograd and Computational Graphs
# 自动梯度和计算图
# The autograd module automates gradient calculation for backpropagation.
# This is crucia in training deep neural networks.
# 自动梯度模块自动计算反向传播的梯度。
# 这对于训练深度神经网络至关重要。

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  #(dy/dx = 2x = 4 when x=2)
# tensor(4.)

# PyTorch dynamically creates a computational graph that tracks operations and gradients for backpropagation.
# PyTorch 动态创建一个计算图，用于跟踪反向传播的操作和梯度。
