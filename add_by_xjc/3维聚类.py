import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# 生成3维数据集
X = np.random.rand(100, 3)  # 生成100个3维随机数据点

# 可视化3维数据集
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 使用K-Means进行聚类
k = 3  # 设定聚类簇数为3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)  # 拟合K-Means模型
labels = kmeans.labels_  # 获取聚类标签
centroids = kmeans.cluster_centers_  # 获取聚类中心

# 可视化聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='*', s=300)  # 显示聚类中心
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
