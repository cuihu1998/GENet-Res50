import moxing as mox
import argparse
import os
import mindspore


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')

args_opt = parser.parse_args()

mul = mindspore.ops.Mul()

print("--------------")
print(mul)