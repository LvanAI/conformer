import os

from copy import deepcopy
import math
from collections import OrderedDict

from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator
from mindspore.train.serialization import _get_merged_param_data, _exec_save

from threading import Thread, Lock
import time

_ckpt_mutex = Lock()

class ModelEMA(Callback):
    """ 
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    """

    def __init__(self, model, decay, eval_dataset, n_parameters, updates = 0):

        # Create EMA
        # # decay exponential ramp (to help early epochs)
        # self.decay = lambda x: decay * (1 - math.exp(-x / 2000)) 
        self.decay = decay
        
        self.updates = updates  # number of EMA updates
        self.model = model
        self.ema = deepcopy(self.model.eval_network())

        # for eval
        self.eval_dataset = eval_dataset
        self.n_parameters = n_parameters


    def step_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.update(self.model.eval_network())

        
    def update(self, network):

        self.updates += 1
        # d = self.decay(self.updates)

        d = self.decay

        msd = network.parameters_dict()
        for k, v in self.ema.parameters_dict.items():
            v *= d
            v += (1. - d) * msd[k]
            self.ema[k] = v
    

    def eval_print(self, epoch):
        eval_start_time = time.time()
        result = self.model.eval(self.eval_dataset, dataset_sink_mode = True)
        eval_end_time = time.time()

        print("[Test] epoch: %s top-1: %s, top-5: %s  test-loss: %s n_parameters: %s eval_time: %s" %
            (epoch, result["top_1_accuracy"], result["top_5_accuracy"],  result["loss"], self.n_parameters, eval_end_time - eval_start_time), flush=True)
        self.model._eval_network = deepcopy(self.ema)
        

        eval_start_time = time.time()
        result = self.model.eval(self.eval_dataset, dataset_sink_mode = True)
        eval_end_time = time.time()
        print("[Test EMA] epoch: %s top-1: %s, top-5: %s  test-loss: %s n_parameters: %s eval_time: %s" %
            (epoch, result["top_1_accuracy"], result["top_5_accuracy"],  result["loss"], self.n_parameters, eval_end_time - eval_start_time), flush=True)

    
    def epoch_end(self, run_context):
        """
        Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        if epoch < 200:
            if epoch % 2 == 0:
                self.eval_print(epoch)

        elif 200 < epoch < 250:
            if (epoch + 1) % 5 == 0:
                self.eval_print(epoch)
        else:
            self.eval_print(epoch)
            
        self.train_loss = 0



