import torch


# 欧氏距离（Euclidean Distance）
def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2   # A^2 - 2ab + B^2
    return d


def quantize(x, centroids):
    b, c, h, w = x.shape
    # [B, C, H, W] => [B, H, W, C]
    # permute: 维度交换
    # contiguous：连续
    # is_contiguous直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致
    x = x.permute(0, 2, 3, 1).contiguous()
    # 进行维度交换后要进行contiguous()，避免在后续的view()出错
    # x.view()就是对tensor进行reshape：
    x = x.view(-1, c)  # flatten to pixels  shape = [b*h*w,c]
    d = squared_euclidean_distance(x, centroids)
    # 返回扁平化张量的最小值或沿某一维度的最小值的索引
    x = torch.argmin(d, 1)
    # 距离越小就是证明越是这个簇的，所以要选小的
    x = x.view(b, h, w)
    return x

# pytorch的 contiguous()
# https://zhuanlan.zhihu.com/p/64551412


def unquantize(x, centroids):
    return centroids[x]


# a = torch.arange(0, 120)  # a's shape is (16,)
#
# a = a.view(4,2,3, 5)  # output below  b c h w
#
#
# a = a.permute(0, 2, 3, 1).contiguous()
# print(a)
# b, c, h, w = a.shape
# a = a.view(-1, c)  # flatten to pixels
# print(a)
# print(a.shape)



