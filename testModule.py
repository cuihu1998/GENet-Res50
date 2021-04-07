
import mindspore
import numpy as np
from src.GENet import *
input = mindspore.Tensor(np.ones([128,3,224,224]).astype(np.float32))
net = GE_resnet50()
out = net(input)
print(out.shape)