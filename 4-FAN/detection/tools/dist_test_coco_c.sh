#!/usr/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

set -e
set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
CORRUPT=$4
SEVERITY=$5
PORT=${PORT:-29501}


# corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
#               'defocus_blur', 'glass_blur', 'motion_blur', 'gaussian_blur',
#                'snow', 'frost', 'fog', 'spatter',
#                 'brightness', 'contrast', 'saturate', 'jpeg_compression']


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_cocoC.py $CONFIG $CHECKPOINT \
    --corrupt $CORRUPT --severity $SEVERITY \
    --launcher pytorch --eval bbox segm ${@:6}
