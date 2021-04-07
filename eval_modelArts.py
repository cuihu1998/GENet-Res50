# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train resnet."""
import os
import argparse
import moxing as mox
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')


parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
args_opt = parser.parse_args()

set_seed(1)


from src.GENet import GE_resnet50 as Net

from src.config import config2 as config
from src.dataset import create_dataset2 as create_dataset


if __name__ == '__main__':
    target = args_opt.device_target

    device_id = int(os.getenv('DEVICE_ID'))

    local_data_url = "/cache/data"

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        context.set_context(device_id=device_id)

    mox.file.copy_parallel(args_opt.data_url, local_data_url)

    # create dataset
    dataset = create_dataset(dataset_path=local_data_url, do_train=False, batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = Net(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
