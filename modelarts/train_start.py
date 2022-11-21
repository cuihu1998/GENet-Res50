# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train GENet."""
import os
import argparse
import glob
import numpy as np
import moxing as mox
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.CrossEntropySmooth import CrossEntropySmooth
from src.GENet import GE_resnet50 as GENet
from src.lr_generator import get_lr
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='Image classification')

parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--extra', type=str, default="False",
                    help='whether to use Depth-wise conv to down sample')
parser.add_argument('--mlp', type=str, default="True", help='bottleneck . whether to use 1*1 conv')
parser.add_argument('--num_classes', default="1000", type=int, help="classes")
parser.add_argument('--epoch_size', default="1", type=int, help="epoch_size")
parser.add_argument('--batch_size', default="128", type=int, help="batch_size")

args_opt = parser.parse_args()

if args_opt.extra.lower() == "false":
    from src.config import config3 as config
else:
    if args_opt.mlp.lower() == "false":
        from src.config import config2 as config
    else:
        from src.config import config1 as config

set_seed(1)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def frozen_to_air(network, args):
    param_dict_t = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(network, param_dict_t)
    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[args.get("batch_size"), 3, args.get("width"), \
        args.get("height")]), mindspore.float32)
    export(network, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))

def trans_char_to_bool(str_):
    """
    Args:
        str_: string

    Returns:
        bool
    """
    result = False
    if str_.lower() == "true":
        result = True
    return result

if __name__ == '__main__':

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv("RANK_SIZE"))

    local_summary_dir = "/cache/summary"
    local_data_url = "/cache/data"
    local_train_url = "/cache/ckpt"
    local_pretrained_path = "/cache/pretrained"

    ckpt_save_dir = local_train_url
    mox.file.make_dirs(local_train_url)
    mox.file.make_dirs(local_summary_dir)
    mox.file.make_dirs(local_pretrained_path)
    filename = "pretrained.ckpt"
    local_pretrained_url = os.path.join(local_pretrained_path, filename)

    if (args_opt.pre_trained is not None) and args_opt.pre_trained != "":
        mox.file.copy(args_opt.pre_trained, local_pretrained_url)
    mox.file.copy_parallel(args_opt.data_url, local_data_url)

    target = args_opt.device_target
    if target != 'Ascend':
        raise ValueError("Unsupported device target.")

    run_distribute = False

    if device_num > 1:
        run_distribute = True

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if run_distribute:

        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()

    # create dataset
    dataset = create_dataset(dataset_path=local_data_url, do_train=True, repeat_num=1,
                             batch_size=args_opt.batch_size, target=target, distribute=run_distribute)
    step_size = dataset.get_dataset_size()

    # define net
    mlp = trans_char_to_bool(args_opt.mlp)
    extra = trans_char_to_bool(args_opt.extra)

    net = GENet(class_num=args_opt.num_classes, extra=extra, mlp=mlp)

    # init weight
    if (args_opt.pre_trained is not None) and args_opt.pre_trained != "":
        param_dict = load_checkpoint(local_pretrained_url)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    lr = get_lr(config.lr_init, config.lr_end, args_opt.epoch_size, step_size, config.decay_mode)

    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if target == "Ascend":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0

        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=args_opt.num_classes)

        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                      metrics={'acc'}, amp_level="O2", keep_batchnorm_fp32=False)
    else:
        raise ValueError("Unsupported device target.")

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    rank_id = int(os.getenv("RANK_ID"))

    cb = [time_cb, loss_cb]

    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs*step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="GENet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    dataset_sink_mode = target != "CPU"
    model.train(args_opt.epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    ckpt_list = glob.glob(ckpt_save_dir + "/*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")
    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)
    net = GENet(class_num=args_opt.num_classes, extra=extra, mlp=mlp)

    frozen_to_air_args = {'ckpt_file': ckpt_model, 'batch_size': 1, 'height': 224, 'width': 224,
                          'file_name': ckpt_save_dir + '/GENet', 'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)

    mox.file.copy_parallel(ckpt_save_dir, args_opt.train_url)
