from turtle import pd

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


img = plt.imread('img.png')  # 加载图片 转换成多维数组
# print(img)
print(img.shape)

# 随机生成一张图片
# img = np.random.randint(low=0,high=255,size=(100,100,4))
# print(img.shape)


# 提取所有的像素点 每一个像素点作为一个样本 像素点这个样本有三个特征
X_train = img.reshape(-1,img.shape[-1])

# K-means模型
# 获取模型，设置要分多少类
# n_clusters 比如说你说数据非常复杂，那么类别数就可能很大
#  比如彩色图片，类别数可以就是它的色域空间
kmeans = KMeans(n_clusters=20)
# ）训练并聚类
y_ = kmeans.fit_predict(X_train)


# 聚类后的目标值是16类
# res = np.sort(pd.Series(y_).unique())
# print(res)



# 4）寻找聚类中心
# 其实就是每一类像素点中的代表
centers = kmeans.cluster_centers_   # 按照各组顺序的 聚类中心
np.save("result.npy",centers)


# 5）将原图片像素点替换为分类后的像素点
# 原理： 将原图数据的像素点替换为分为聚类后的像素点，这些像素点其实就是将原数据的像素点分为16种后的像素点
result_img = centers[y_]

# 6）将聚类后的图片绘出来比较
# 注意：要将图片的数据进行形状还原 (有进行shape处理的话)
# print(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(result_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
plt.show()

# 比如上面进行了一张3D彩色图片的聚类，类别数是16，那么聚类结果result就是一个 [16,3]的数组  [类别数，通道数]
# 加入你的数据是 [H,W,C]  C >> 3 那么聚类结果也是  [类别数，通道数]