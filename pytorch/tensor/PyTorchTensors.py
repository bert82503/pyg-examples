import torch

# What is PyTorch ?
# https://www.geeksforgeeks.org/getting-started-with-pytorch/

# PyTorch Tensors
# 张量
# Tensors are the fundamental data structures in PyTorch, similar to NumPy arrays but with GPU acceleration capabilities.
# PyTorch tensors support automatic differentiation, making them suitable for deep learning tasks.
# 张量是 PyTorch 中的基本数据结构，类似于 NumPy 数组，但具有 GPU 加速功能。
# PyTorch 张量支持自动微分，使其适用于深度学习任务。

# Creating a 1D tensor
# 1x3矩阵向量
x = torch.tensor([1.0, 2.0, 3.0])
print('1D Tensor: \n', x)
# tensor([1., 2., 3.])

# Creating a 2D tensor
# 3x3矩阵向量
y = torch.zeros((3, 3))
print('2D Tensor: \n', y)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
