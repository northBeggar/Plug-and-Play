#!/usr/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

set -e
set -x




CONFIG=/PATH/TO/CONFIG/FILE
CKP=/PATH/TO/CHECKPOINT/FILE
WORKDIR='output/'



#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_noise 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_noise 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_noise 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_noise 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_noise 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 shot_noise 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 shot_noise 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 shot_noise 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 shot_noise 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 shot_noise 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 impulse_noise 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 impulse_noise 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 impulse_noise 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 impulse_noise 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 impulse_noise 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 speckle_noise 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 speckle_noise 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 speckle_noise 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 speckle_noise 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 speckle_noise 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 defocus_blur 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 defocus_blur 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 defocus_blur 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 defocus_blur 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 defocus_blur 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 motion_blur 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 motion_blur 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 motion_blur 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 motion_blur 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 motion_blur 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_blur 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_blur 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_blur 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_blur 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 gaussian_blur 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 glass_blur 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 glass_blur 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 glass_blur 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 glass_blur 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 glass_blur 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 snow 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 snow 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 snow 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 snow 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 snow 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 frost 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 frost 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 frost 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 frost 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 frost 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 fog 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 fog 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 fog 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 fog 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 fog 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 spatter 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 spatter 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 spatter 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 spatter 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 spatter 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 brightness 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 brightness 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 brightness 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 brightness 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 brightness 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 contrast 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 contrast 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 contrast 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 contrast 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 contrast 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 saturate 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 saturate 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 saturate 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 saturate 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 saturate 5 --work-dir $WORKDIR
#-------------------------------
sh dist_test_cocoC.sh $CONFIG $CKP 8 jpeg_compression 1 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 jpeg_compression 2 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 jpeg_compression 3 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 jpeg_compression 4 --work-dir $WORKDIR
sh dist_test_cocoC.sh $CONFIG $CKP 8 jpeg_compression 5 --work-dir $WORKDIR
-------------------------------
