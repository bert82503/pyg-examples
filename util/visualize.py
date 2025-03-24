
# 可视化
# 可以直接利用模型输出的节点特征降维进行可视化，用TSNE降维方法将节点特征降至2维，在坐标系中可视化。

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 可视化
def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
