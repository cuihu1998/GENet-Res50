# import moxing as mox
import argparse
import os
import numpy as np
import mindspore


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')

args_opt = parser.parse_args()

t1 = np.random.randn(3,4)
t2 = np.random.randn(3,4)

print(t1)
print(t2)

t1 = mindspore.Tensor(t1).astype(np.float32)
t2 = mindspore.Tensor(t2).astype(np.float32)

add_operation = mindspore.ops.TensorAdd()
t3 = add_operation(t1,t2)

print("-----------------")
print(t3)
# print(add_operation)