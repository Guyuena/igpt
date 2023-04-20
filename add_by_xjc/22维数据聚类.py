import numpy as np
from sklearn.cluster import KMeans

# 生成22维数据集
X = np.random.rand(100, 22)  # 生成100个22维随机数据点

# 使用K-Means进行聚类
k = 3  # 设定聚类簇数为3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)  # 拟合K-Means模型
labels = kmeans.labels_  # 获取聚类标签
centroids = kmeans.cluster_centers_  # 获取聚类中心

# 输出聚类结果
print("聚类标签：", labels)
print("聚类中心：", centroids)
print("聚类标签：", labels.shape)
print("聚类中心：", centroids.shape)
