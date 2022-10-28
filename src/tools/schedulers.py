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
"""LearningRate scheduler functions"""
import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy", "exp_lr", "multiply_lr"]


def get_policy(name):
    """get lr policy from name"""
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "exp_lr": exp_lr,
        "multiply_lr": multiply_lr,
    }

    return out_dict[name]


def constant_lr(args, batch_num):
    """Get constant lr"""
    learning_rate = []

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, args.base_lr, args.warmup_length, epoch)
        else:
            lr = args.base_lr

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def exp_lr(args, batch_num):
    """Get exp lr """
    learning_rate = []

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, args.base_lr, args.warmup_length, epoch)
        else:
            lr = args.base_lr * args.lr_gamma ** epoch

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def cosine_lr(args, batch_num):
    """Get cosine lr"""

    import os
    learning_rate = []

    base_lr = args.base_lr * args.batch_size * int(os.getenv("DEVICE_NUM", args.device_num)) / 512

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, base_lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def cosine_lr2(args, batch_num):
    """Get cosine lr timm"""
    import os
    learning_rate = []
    base_lr = args.base_lr * args.batch_size * int(os.getenv("DEVICE_NUM", args.device_num)) / 512
    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, base_lr, args.warmup_length, epoch)
        else:
            lr = args.min_lr + 0.5 * (base_lr - args.min_lr) * (1 + np.cos(np.pi * epoch / args.epochs))

        return lr

    for epoch in range(args.epochs):
        for _ in range(batch_num):
            lr = _lr_adjuster(epoch)
            if epoch < args.warmup_length:
                lr = max(lr, args.warmup_lr)
            else:
                lr = max(lr, args.min_lr)
            learning_rate.append(lr)

    return np.array(learning_rate)


def multistep_lr(args, batch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = []

    def _lr_adjuster(epoch):
        lr = args.base_lr * (args.lr_gamma ** (epoch / args.lr_adjust))
        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))

    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate

def multiply_lr(args, batch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    import os
    learning_rate = []

    base_lr = args.base_lr * args.batch_size * int(os.getenv("DEVICE_NUM", args.device_num)) / 512

    def _lr_adjuster(epoch):
        lr = base_lr * (1 - epoch / args.epochs)
        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))

    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))

    return learning_rate


def _warmup_lr(warmup_lr, base_lr, warmup_length, epoch):
    """Linear warmup"""
    return epoch / warmup_length * (base_lr - warmup_lr) + warmup_lr

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--warmup_length', type = int, default=5, help='none')
    argparser.add_argument('--base_lr', type = float, default=0.08, help='none')
    argparser.add_argument('--warmup_lr', type= float, default=1e-6, help='')
    argparser.add_argument('--epochs', type= int, default=300, help='')
    argparser.add_argument('--min_lr', type= float, default=1e-5, help='')
    argparser.add_argument('--batch_size', type=int, default=64, help='')
    argparser.add_argument('--device_num', type=int, default=8, help='')
    args = argparser.parse_args()
    learning_rate = multiply_lr(args, 1)
    print(len(learning_rate))
    print('learning_rate', learning_rate)