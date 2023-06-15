import torch
from mmdet.models.dense_heads import query_generator

from mmdet.models.roi_heads.bbox_heads.adaptive_mixing_operator import AdaptiveMixing
from mmdet.models.dense_heads.query_generator import InitialQueryGenerator
from mmdet.models.detectors import QueryBased

num_stages = 6
num_proposals = 100
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048

# P_in for spatial mixing in the paper.
in_points_list = [32, ] * num_stages

# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns_list = [128, ] * num_stages

# G for the mixer grouping in the paper. Also named as n_head in this codebase.
n_group_list = [4, ] * num_stages

detector = QueryBased(
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
    neck=dict(
        type='ChannelMapping',
        in_channels=[256, 512, 1024, 2048],
        out_channels=FEAT_DIM,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    rpn_head=dict(
        type='InitialQueryGenerator',
        num_query=num_proposals,
        content_dim=QUERY_DIM),
    roi_head=dict(
        type='AdaMixerDecoder',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=QUERY_DIM,
        bbox_head=[
            dict(
                type='AdaMixerDecoderStage',
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=FF_DIM,
                content_dim=QUERY_DIM,
                feat_channels=FEAT_DIM,
                dropout=0.0,
                in_points=in_points_list[stage_idx],
                out_points=out_patterns_list[stage_idx],
                n_groups=n_group_list[stage_idx],
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for stage_idx in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals))
)
