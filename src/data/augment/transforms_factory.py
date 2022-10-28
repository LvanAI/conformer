
import math

import mindspore.dataset.vision.py_transforms as PT
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.vision.utils import Inter

from src.data.augment.auto_augment import _pil_interp, rand_augment_transform
from src.data.augment.random_erasing import RandomErasing
from src.data.augment.constant import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,DEFAULT_CROP_PCT


DEFAULT_CROP_PCT = 0.875

def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        re_prob=0.,
        re_mode='const',
        re_count=1
):
    """

    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        C.RandomCropDecodeResize(img_size, scale= scale, ratio= ratio,
                                          interpolation=Inter.BICUBIC),]
    primary_tfl += [PT.ToPIL()]
    if hflip > 0.:
        primary_tfl += [PT.RandomHorizontalFlip(hflip)]
    if vflip > 0.:
        primary_tfl += [PT.RandomVerticalFlip(vflip)]
    
    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )

        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)

        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [C.RandomColorAdjust(*color_jitter)]

    final_tfl = []

        
    final_tfl += [
        PT.ToTensor(),
        PT.Normalize(mean = mean, std = std),
    ]
    if re_prob > 0.:
        final_tfl.append(
            RandomErasing(re_prob, mode= re_mode, max_count= re_count))

    return primary_tfl + secondary_tfl + final_tfl


def transforms_imagenet_eval(
        img_size=224,
        crop_pct = None
        ):
    
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    mean = [0.485, 0.456 , 0.406]
    std =  [0.229 , 0.224 , 0.225]

    if crop_pct:
        tfl = [C.Decode(),
            C.Resize([scale_size, scale_size], interpolation = Inter.BICUBIC),
            C.CenterCrop(img_size)
        ]
    else:
        tfl = [C.Decode(),
            C.Resize([img_size, img_size], interpolation = Inter.BICUBIC)
        ]
    tfl += [
        PT.ToTensor(),
        PT.Normalize( mean = mean, std = std),
    ]
    return tfl
