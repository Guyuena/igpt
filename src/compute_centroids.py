import numpy as np
from pathlib import Path
import argparse

from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans, MiniBatchKMeans

from data import DATASETS


def download(dataset, datapath):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=ToTensor()
    )
    train_x = np.stack([x.numpy() for x, _ in train_ds])
    train_x = train_x.transpose(0, 2, 3, 1)  # put channel dimension last
    return train_x

# 对于图像来说：
#     image.shape[0]——图片高
#     image.shape[1]——图片长
#     image.shape[2]——图片通道数
# 而对于矩阵来说：
#     shape[0]：表示矩阵的行数
#     shape[1]：表示矩阵的列数



def find_centroids(train_x, num_clusters=16, batch_size=1024):
    # 先将其转为1维的向量。直接对其展平：
    print("train_x.shape= " ,train_x.shape)
    # 输入 (B,H,W,C)
    pixels = train_x.reshape(-1, train_x.shape[-1])
    print("处理后 pixels.shape= ", pixels.shape)
    # 展平： (B*H*W, C)
    if batch_size:
        kmeans = MiniBatchKMeans(
            # n_clusters: 即我们的k值，和KMeans类的n_clusters意义一样。
            n_clusters=num_clusters,
            random_state=0,
            # batch_size：即用来跑MiniBatch KMeans算法的采样集的大小，
            # 如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果
            batch_size=batch_size
        ).fit(pixels)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)

    # print("kmeans.shape= ", kmeans.shape)
    print("kmeans.cluster_centers_= ",kmeans.cluster_centers_)
    print("kmeans= ", kmeans)
    return kmeans.cluster_centers_
    # 计算出类别和簇的的中心



#  聚类质心计算
def main(args):
    datapath = Path("data")
    datapath.mkdir(exist_ok=True)

    train_x = download(args.dataset, datapath)
    centroids = find_centroids(train_x, args.num_clusters, args.batch_size)
    np.save(datapath / f"{args.dataset}_centroids.npy", centroids)


if __name__ == "__main__":
    print("compute_centroids.py被调用")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="mnist") # 使用哪个数据集
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="cifar10") # 使用哪个数据集
    parser.add_argument("--num_clusters", default=16, type=int)  # 组别数
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="batch size for mini batch kmeans to quantize images",
    )
    args = parser.parse_args()
    main(args)

# python src/compute_centroids.py --dataset mnist --num_clusters=8