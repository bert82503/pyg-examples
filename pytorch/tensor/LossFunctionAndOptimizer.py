import torch
import torch.nn as nn
from pytorch.tensor.NeuralNetworks import NeuralNetwork as NeuralNetwork

# Define Loss Function and Optimizer
# 定义损失函数和优化器

# Once we define our model, we need to specify:
# A loss function to measure the error.
# An optimizer to update the weights based on computed gradients.
# 一旦定义了模型，我们就需要指定：
# 用于测量误差的损失函数。
# 根据计算的梯度更新权重的优化器。

# We use nn.BCELoss() for binary cross-entropy loss and used optim.Adam() for Adam optimizer to combine the benefits of momentum and adaptive learning rates.
# 我们使用 nn.BCELoss() 作为二元交叉熵损失，并使用 optim.Adam() 作为 Adam 优化器，以结合动量和自适应学习率的优点。

model = NeuralNetwork()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train the Model
# 训练模型

# The training involves:
# 1. Generating dummy data (100 samples, each with 10 features).
# 2. Running a training loop where we:
#     optimizer.zero_grad() clears the accumulated gradients from the previous step.
#     Forward Pass (model(inputs)) passes inputs through the model to generate predictions.
#     Loss Computation (criterion(outputs, targets)) computes the difference between predictions and actual labels.
#     Backpropagation (loss.backward()) computes gradients for all weights.
#     Optimizer Step (optimizer.step()) updates the weights based on the computed gradients.
# 训练内容包括：
# 1. 生成虚拟数据（100 个样本，每个样本有 10 个特征）。
# 2. 运行训练循环，我们：

# 100x10矩阵向量
inputs = torch.randn((100, 10))
targets = torch.randint(0, 2, (100, 1)).float()
epochs = 20

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, targets)  # Compute loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Epoch [5/20], Loss: 0.6684
# Epoch [10/20], Loss: 0.6533
# Epoch [15/20], Loss: 0.6355
# Epoch [20/20], Loss: 0.6085
