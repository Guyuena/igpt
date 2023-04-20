

import numpy as np



# loadData = np.load('cifar10_centroids.npy')
loadData = np.load('mnist_centroids.npy')
# loadData = np.load('kmeans_centers.npy')
print("----type----")
print(type(loadData))
print("----shape----")
print(loadData.shape)
print("----data----")
print(loadData)