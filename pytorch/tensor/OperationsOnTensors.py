import torch

# Operations on Tensors
# 张量上的运算

# 1x2矩阵向量
a = torch.tensor([1.0, 2.0])
# 1x2矩阵向量
b = torch.tensor([3.0, 4.0])

# Element-wise addition
# 向量加法
print('Element Wise Addition of a & b: \n', a + b)
# tensor([4., 6.])

# Matrix multiplication
# 2行1列
# a.view(2, 1)
# 1行2列
# b.view(1, 2)
# 矩阵乘法
print('Matrix Multiplication of a & b: \n',
      torch.matmul(a.view(2, 1), b.view(1, 2)))
# tensor([[3., 4.],
#         [6., 8.]])
