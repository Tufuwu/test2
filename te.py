import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 西瓜数据集 4.0
data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]
])

# 设置不同的 k 值和初始化中心点
k_values = [2, 3, 4]
initial_centers = [
    np.array([[0.6, 0.3], [0.4, 0.4]]),         # k=2 时的初始中心点
    np.array([[0.6, 0.3], [0.4, 0.4], [0.7, 0.2]]),  # k=3 时的初始中心点
    np.array([[0.6, 0.3], [0.4, 0.4], [0.7, 0.2], [0.3, 0.1]])  # k=4 时的初始中心点
]

# 绘制不同 k 值下的聚类结果
for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, init=initial_centers[i], n_init=1)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 绘制聚类结果
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    plt.title(f'k-Means Clustering (k={k})')
    plt.xlabel('Density')
    plt.ylabel('Sugar Content')
    plt.legend()
    plt.grid()
    plt.show()
