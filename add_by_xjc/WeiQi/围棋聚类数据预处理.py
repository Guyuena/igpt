import numpy as np
import tensorflow as tf
from PIL import Image
import math
data = np.load('data1.npz')
print(data.files)
print(data["binaryInputNCHWPacked"].shape)
print(data["policyTargetsNCMove"].shape)

binchwp = data["binaryInputNCHWPacked"]  # shape= [batch-size, num_plane ,packedBoardArea ]  [batch_size,num_bin_input_features,(pos_len*pos_len+7)//8])
bitmasks = tf.reshape(tf.constant([128, 64, 32, 16, 8, 4, 2, 1], dtype=tf.uint8), [1, 1, 1, 8])
# bitmasks结果： tf.Tensor([[[[128  64  32  16   8   4   2   1]]]], shape=(1, 1, 1, 8), dtype=uint8)
# print(bitmasks)
binchw = tf.reshape(tf.bitwise.bitwise_and(tf.expand_dims(binchwp, axis=3), bitmasks),
                    [-1, 22, ((19 * 19 + 7) // 8) * 8])
binchw = binchw[:, :, :19 * 19]
print("binchw= ",binchw.shape)
binhwc = tf.cast(tf.transpose(binchw, [0, 2, 1]), tf.float32)  #变为 [batch-size,22,361]  ---> [batch-size,361,22]
# print("binchwc= ",binhwc.shape)
# print("binchwc= ",binhwc)
binhwc = tf.math.minimum(binhwc, tf.constant(1.0))  # 取出真正表示棋子的0-1数据
# print("binchwc= ",binhwc)
print("binchwc= ",binhwc.shape)
binchwc_2 = tf.reshape(binhwc,[8353,19,19,22])
print("binchwc= ",binchwc_2.shape)
np.save('Go_data.npy',binchwc_2)
# print(binhwc[0])