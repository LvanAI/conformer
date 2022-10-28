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
"""
Functions of cells
"""
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import ops

from mindspore._checkparam import Validator as validator
from mindspore.ops import functional as F

from src.args import args


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def cast_amp(net):
    """cast network amp_level"""
    if args.amp_level == "O1":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
        cell_types = (nn.GELU, nn.Softmax,nn.Conv1d, nn.BatchNorm2d, nn.LayerNorm) # nn.Conv2d, 
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O2":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
        cell_types = (nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O3":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {args.amp_level}")
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {args.loss_scale}")


class WithEvalCell(nn.Cell):
    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = validator.check_value_type("add_cast_fp32", add_cast_fp32, [bool], self.cls_name)

    def construct(self, data, label):
        outputs = self._network(data)
        labels =  ops.Concat(axis = 0)((label, label))
        if self.add_cast_fp32:
            labels = F.mixed_precision_cast(mstype.float32, labels)
            outputs = F.cast(outputs, mstype.float32)

        loss = self._loss_fn(outputs, labels)
        return loss, outputs, labels