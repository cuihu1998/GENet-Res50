import moxing as mox
import argparse
import os
import mindspore
from src.AddOp import Add


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')

args_opt = parser.parse_args()

add_operation = Add()

print("--------------")
print(add_operation)