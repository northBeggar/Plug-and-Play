#!/usr/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE


set -e
set -x

CONFIG=$1
NODE_NUM=$2
NODE_RANK=$3
MASTER_ADDR=$4


PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    --nnodes=$NODE_NUM  --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:5}
