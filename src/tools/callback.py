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
"""callback function"""

import math
import time
from copy import deepcopy


import mindspore.ops as ops
from mindspore import nn
from mindspore.communication import get_group_size

from mindspore.train.callback import Callback
from mindspore._checkparam import Validator
import numpy as np
from mindspore.common.tensor import Tensor

from src.args import args


class EvaluateCallBack(Callback):
    """
    EvaluateCallBack
    """

    def __init__(self, model, eval_dataset,  n_parameters):
        super(EvaluateCallBack, self).__init__()

        self.model = model
        self.eval_dataset = eval_dataset
        self.n_parameters = n_parameters
        self.train_loss = 0

    def eval_print(self, epoch):
        eval_start_time = time.time()
        result = self.model.eval(self.eval_dataset, dataset_sink_mode = False)
        eval_end_time = time.time()

        print("epoch: %s top-1: %s, top-5: %s  test-loss: %s n_parameters: %s eval_time: %s" %
            (epoch, result["top_1_accuracy"], result["top_5_accuracy"],  result["loss"], self.n_parameters, eval_end_time - eval_start_time), flush=True)


    def step_end(self, run_context):

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))
        self.train_loss += loss


    def epoch_end(self, run_context):
        """
        Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num

        if epoch < 200:
            if epoch % 10 == 0:
                self.eval_print(epoch)
        elif 200 <= epoch < 250:
            if epoch % 5 == 0:
                self.eval_print(epoch)
        else:
            self.eval_print(epoch)
            
        self.train_loss = 0


class VarMonitor(Callback):
    def __init__(self,step_size, per_print_times, lr):
        super(VarMonitor, self).__init__()
        Validator.check_non_negative_int(per_print_times)

        self.lr = lr
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.step_one_epoch = step_size
        self.step_time = time.time()
        self.time_cost = 0
        self.count = 1

    def step_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        if self.count == 1:
            self.per_print = time.time()
            self.count += 1


    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num

            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            time_cost = time.time() - self.per_print
            print("time: %s epoch: %s step: %s, steps_one_epoch: %d, loss: %s,lr: %s step time: %.3fs" \
                % (time_str, cb_params.cur_epoch_num, cur_step_in_epoch,self.step_one_epoch, loss, self.lr[cb_params.cur_step_num], time_cost), flush=True)
            
            self.count = 1


