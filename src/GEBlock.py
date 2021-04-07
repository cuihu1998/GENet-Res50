


# Copyright 2020 Huawei Technologies Co., Ltd
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
"""ResNet."""
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P
from src.AddOp import Add



class GEBlock(nn.Cell):
    """
    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        spatial = output_size of block
        extra_params : whether to use Depth-wise Conv to downsample
        mlp: whether to use 1*1 conv to
    Returns:
        Tensor, output tensor.
    Examples:
        >>>
    """

    def __init__(self, in_channel, out_channel, stride=1, spatial=0, extra_params=False, mlp=True):
        super().__init__()
        self.expansion = 4

        self.extra_params = extra_params

        channel = out_channel // self.expansion
        self.conv1 = nn.Conv2dBnAct(in_channel, channel, kernel_size=1, stride=1,
                                    has_bn=True, pad_mode="same", activation='relu')

        self.conv2 = nn.Conv2dBnAct(channel, channel, kernel_size=3, stride=stride,
                                    has_bn=True,  pad_mode="same", activation='relu')

        self.conv3 = nn.Conv2dBnAct(channel, out_channel, kernel_size=1, stride=1, pad_mode='same',
                                    has_bn=True)

        # whether down-sample identity
        self.res_down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.res_down_sample = True

        self.res_down_sample_layer = None
        if self.res_down_sample:
            self.res_down_sample_layer = nn.Conv2dBnAct(in_channel, out_channel,
                                                        kernel_size=1, stride=stride,
                                                        pad_mode='same', has_bn=True)

        if extra_params:
            self.downop = self.downop = nn.SequentialCell([nn.Conv2d(in_channels=out_channel,
                                                                     out_channels=out_channel,
                                                                     kernel_size=spatial,
                                                                     stride=1,group=out_channel,
                                                                     padding=0,
                                                                     pad_mode="pad"),
                                                           nn.BatchNorm2d(out_channel)])
            # self.downop = self.downop = nn.SequentialCell([nn.Conv2d(in_channels=out_channel,
            #                                                          out_channels=out_channel,
            #                                                          kernel_size=spatial,
            #                                                          stride=1,
            #                                                          padding=0,
            #                                                          pad_mode="pad"),
            #                                                nn.BatchNorm2d(out_channel)])
        else:
            # self.downop = nn.AvgPool2d(kernel_size = spatial)
            self.downop = P.ReduceMean(keep_dims=True)

        self.mlp = mlp
        if mlp:
            mlpLayer = []
            mlpLayer.append(nn.Conv2d(in_channels=out_channel,
                                      out_channels=out_channel//16,
                                      kernel_size=1))
            mlpLayer.append(nn.ReLU())
            mlpLayer.append(nn.Conv2d(in_channels=out_channel//16,
                                      out_channels=out_channel,
                                      kernel_size=1))
            self.mlpLayer = nn.SequentialCell(mlpLayer)

        self.sigmoid = nn.Sigmoid()
        self.add_op = Add()
        self.relu = nn.ReLU()
        self.mul = ms.ops.Mul()

    def construct(self, x):

        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.res_down_sample:
            identity = self.res_down_sample_layer(identity)

        # gather operation
        # input = (C,spatial,spatial)
        # output = (C,1,1)
        if self.extra_params:
            out_ge = self.downop(out)
        else:
            out_ge = self.downop(out, (2, 3))

        if self.mlp:
            out_ge = self.sigmoid(self.mlpLayer(out_ge))

        out = self.mul(out, out_ge)
        out = self.add_op.add(out, identity)
        out = self.relu(out)

        return out

