import moxing as mox
import argparse
import os

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')

args_opt = parser.parse_args()

local_data_url = "/cache/data"
local_zipfolder_url = "/cache/tarzip"

mox.file.make_dirs(local_zipfolder_url)
mox.file.make_dirs(local_data_url)

filename = "imagenet_original.tar.gz"
local_zip_path = os.path.join(local_zipfolder_url, filename)
obs_zip_path = os.path.join(args_opt.data_url,filename)
mox.file.copy(obs_zip_path, local_zip_path)

unzip_command = "tar -xvf %s -C %s" % (local_zip_path,local_data_url)
os.system(unzip_command)
print("++++++++++++++ unzip success ++++++++++++++++++")
mox.file.copy_parallel(local_data_url, args_opt.data_url)
print("++++++++++++++ transfer success ++++++++++++++++++")