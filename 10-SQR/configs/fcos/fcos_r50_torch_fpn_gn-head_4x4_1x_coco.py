# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

data = dict(samples_per_gpu=4, workers_per_gpu=4)

model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
)
