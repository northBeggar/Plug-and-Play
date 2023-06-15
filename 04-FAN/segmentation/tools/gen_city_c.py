# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# Copyright (c) Open-MMLab. All rights reserved.

import numpy as np
from imagecorruptions import corrupt
import random
import os
import mmcv


random.seed(8) # for reproducibility
np.random.seed(8)


corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

def perturb(i, p, s):
    img = corrupt(i, corruption_name=p, severity=s)
    return img


def convert_img_path(ori_path, suffix):
    new_path = ori_path.replace('clean', suffix)
    assert new_path != ori_path
    return new_path

def main():
    img_dir = '../ade20k_c/clean/'
    severity = [1, 2, 3, 4, 5]
    num_imgs = 5000
    for p in corruptions:
        print("\n ### gen corruption:{} ###".format(p))
        prog_bar = mmcv.ProgressBar(num_imgs)
        for img_path in mmcv.scandir(img_dir, suffix='jpg', recursive=True):
            img_path = os.path.join(img_dir, img_path)
            img = mmcv.imread(img_path)
            prog_bar.update()
            for s in severity:
                perturbed_img = perturb(img, p, s)
                img_suffix = p+"/"+str(s)
                perturbed_img_path = convert_img_path(img_path, img_suffix)
                mmcv.imwrite(perturbed_img, perturbed_img_path, auto_mkdir=True)



if __name__ == '__main__':
    main()
