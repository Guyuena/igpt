import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import PIL.Image as image


# X = np.ones([19,19,22])
# print(X.shape)
#读取原始图像
paths ="img.png"
X = plt.imread(paths)
print("输入图像 X.shape ",X.shape)
X = np.array(X)
print("输入图像 X.shape ",X.shape)
#print(X.shape)
shape = row ,col ,dim =X.shape
X_ = X.reshape(-1,3)#(将矩阵化为2维，才能使用聚类)
#print(X_.shape)
def kmeans(X, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    Y = kmeans.predict(X)
    return Y

plt.figure(1)  # 图像窗口名称
plt.subplot(2,3,1)
plt.imshow(X)
plt.axis('off')  # 关掉坐标轴为 off
plt.xticks([])
plt.yticks([])
plt.title("Picture")
for t in range(2, 7):
    index = '23' + str(t)
    plt.subplot(int(index))
    label = kmeans(X_,t)
    print("label.shape=",label.shape)
    # get the label of each pixel
    label = label.reshape(row,col)
    print("label new  shaep= ",label.shape)
    # create a new image to save the result of K-Means
    pic_new = image.new("RGB", (col, row))#定义的是图像大小为y*x*3的图像，这里列在前面行在后面
    for i in range(col):
        for j in range(row):
                if label[j][i] == 0:
                    pic_new.putpixel((i, j), (0, 0, 255))#填写的是位置为（j,i）位置的像素，列和行也是反的
                elif label[j][i] == 1:
                    pic_new.putpixel((i, j), (255, 0, 0))
                elif label[j][i] == 2:
                    pic_new.putpixel((i, j), (0, 255, 0))
                elif label[j][i] == 3:
                    pic_new.putpixel((i, j), (60, 0, 220))
                elif label[j][i] == 4:
                    pic_new.putpixel((i, j), (249, 219, 87))
                elif label[j][i] == 5:
                    pic_new.putpixel((i, j), (167, 255, 167))
                elif label[j][i] == 6:
                    pic_new.putpixel((i, j), (216, 109, 216))
    title = "k="+str(t)
    plt.title(title)
    plt.imshow(pic_new)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.xticks([])
    plt.yticks([])

plt.show()


# kmeans步骤共可分为以下步骤：
#
#    1 随机初始化k个聚类中心。
#
#    2 计算所有像素点到聚类中心的距离。
#
#    3 选择最近的聚类中心作为像素点的聚类种类。
#
#    4 根据像素点的聚类种类更新聚类中心。
#
#    5 重复步骤2-4直至聚类中心收敛。