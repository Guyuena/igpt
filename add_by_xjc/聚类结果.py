
import numpy as np

f = np.load('result.npy')
print(f)
print(f.shape)

data = np.random.randint(low=0,high=255,size=(100,100,3))
print(data.shape)