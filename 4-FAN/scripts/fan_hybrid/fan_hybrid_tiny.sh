#!/user/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

set -e
set -x
export NCCL_LL_THRESHOLD=0


rank_num=$1
rank_index=$2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=$rank_num \
	--node_rank=$rank_index --master_addr="10.192.26.27" --master_port=3349 \
	main.py  ../imagenet_raw/ --model fan_tiny_8_p4_hybrid -b 32 --sched cosine --epochs 300 \
	--opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
	--model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel \
	--reprob 0.3 --lr 10e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
	--drop-path .1 --img-size 224 --mixup 0.8 --cutmix 1.0 \
	--smoothing 0.1 \
	--output ../fan_tiny_8_p4_hybrid/ \
	--amp --model-ema \
