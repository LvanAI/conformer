# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.dataset.vision.utils import Inter

from src.data.augment.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from src.data.augment.mixup import Mixup

from .data_utils.moxing_adapter import sync_data


class ImageNet:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads = 256)
            print('Create train and evaluate dataset.')
            train_dir = os.path.join(local_data_path,"imagenet", "train")
            val_ir = os.path.join(local_data_path,"imagenet", "val")
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)
        else:
            train_dir = os.path.join(args.data_url, "train")
            val_ir = os.path.join(args.data_url, "val")
            if training:
                self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)


def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True):
    """
    create a train or eval imagenet2012 dataset for SwinTransformer

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """
    ds.config.set_seed(58)

    device_num, rank_id = _get_rank_info()
    print("device_num %d rank_id %d" %(device_num, rank_id))
    if training:
        print("*"*10, "create train dataset", "*"*10)
    else:
        print("*"*10, "create val dataset", "*"*10)
    shuffle = bool(training)
    if device_num == 1 or not training:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers,
                                         shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = args.image_size

    interpolation = args.interpolation
    if training:
        auto_augment = args.auto_augment
        transform_img = transforms_imagenet_train(img_size= image_size,
            scale=None,
            ratio=None,
            hflip=0.5,
            vflip=0.,
            color_jitter=0.4,
            auto_augment= auto_augment,
            interpolation= interpolation,
            re_prob= args.re_prob,
            re_mode= args.re_mode,
            re_count= args.re_count
        )
        
    else:
        transform_img = transforms_imagenet_eval(img_size = image_size, crop_pct = args.crop_pct)

    transform_label = C2.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mix_up > 0. or args.cutmix > 0.)  and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C2.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder= True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)


    if training:
        data_set = data_set.shuffle(10)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
