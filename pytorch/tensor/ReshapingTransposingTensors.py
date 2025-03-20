import torch

# Reshaping and Transposing Tensors
# 重塑和转置张量

# 3x4矩阵向量
t = torch.tensor([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

# Reshaping
print("Reshaping")
# 6行2列
print(t.reshape(6, 2))
# tensor([[ 1,  2],
#         [ 3,  4],
#         [ 5,  6],
#         [ 7,  8],
#         [ 9, 10],
#         [11, 12]])

# Resizing (deprecated, use reshape)
print("\nResizing")
# 2行6列
print(t.view(2, 6))
# tensor([[ 1,  2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11, 12]])

# Transposing
print("\nTransposing")
# 行列互换
# 4x3矩阵向量
print(t.transpose(0, 1))
# tensor([[ 1,  5,  9],
#         [ 2,  6, 10],
#         [ 3,  7, 11],
#         [ 4,  8, 12]])
