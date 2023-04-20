
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
data = np.load('Go_data.npy')
print(data.shape)

X_train = data.reshape(-1,data.shape[-1])

# K-means模型
# 获取模型，设置要分多少类
# n_clusters 比如说你说数据非常复杂，那么类别数就可能很大
#  比如彩色图片，类别数可以就是它的色域空间
kmeans = KMeans(
                n_clusters=361,
                max_iter=10,
                verbose=1
)
#
# # ）训练并聚类
y_ = kmeans.fit_predict(X_train)

# kmeans = MiniBatchKMeans(
#     # n_clusters: 即我们的k值，和KMeans类的n_clusters意义一样。
#     n_clusters=361,
#     random_state=0,
#     verbose=1,
#    # max_no_improvement：即连续多少个Mini Batch没有改善聚类效果的话，就停止算法，
#     max_no_improvement=100,
#     # batch_size：即用来跑MiniBatch KMeans算法的采样集的大小，
#     # 如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果
#     batch_size=8000
# ).fit(X_train)








# 聚类后的目标值是16类
# res = np.sort(pd.Series(y_).unique())
# print(res)



# 4）寻找聚类中心
# 其实就是每一类像素点中的代表
centers = kmeans.cluster_centers_   # 按照各组顺序的 聚类中心
np.save("GO_result_mini2.npy",centers)


# inertia，统计学中 “和方差”、“簇内离差平方和”（SSE）在这里指同一意思
# 而将一个数据集中的所有簇的簇内平方和相加，就得到了整体平方和(Total Cluster Sum of Square)，
# 又叫做total inertia，TSSE。Total Inertia越小，代表着每个簇内样本越相似，聚类的效果就越好。
# 因此KMeans追求的是，求解能够让Inertia最小化的质心。
