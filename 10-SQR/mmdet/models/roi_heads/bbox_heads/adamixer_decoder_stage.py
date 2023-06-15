import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead

from .sampling_3d_operator import sampling_3d
from .adaptive_mixing_operator import AdaptiveMixing

from mmdet.core import bbox_overlaps

import os


DEBUG = 'DEBUG' in os.environ


def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)


def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi


def make_sample_points(offset, num_group, xyzr):
    '''
        offset_yx: [B, L, num_group*3], normalized by stride

        return: [B, H, W, num_group, 3]
        '''
    B, L, _ = offset.shape

    offset = offset.view(B, L, 1, num_group, 3)

    roi_cc = xyzr[..., :2]
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                               xyzr[..., 3:4] * 0.5], dim=-1)
    roi_wh = scale * ratio

    roi_lvl = xyzr[..., 2:3].view(B, L, 1, 1, 1)

    offset_yx = offset[..., :2] * roi_wh.view(B, L, 1, 1, 2)
    sample_yx = roi_cc.contiguous().view(B, L, 1, 1, 2) \
        + offset_yx

    sample_lvl = roi_lvl + offset[..., 2:3]

    return torch.cat([sample_yx, sample_lvl], dim=-1)


class AdaptiveSamplingMixing(nn.Module):
    _DEBUG = 0

    def __init__(self,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 content_dim=256,
                 feat_channels=None
                 ):
        super(AdaptiveSamplingMixing, self).__init__()
        self.in_points = in_points
        self.out_points = out_points
        self.n_groups = n_groups
        self.content_dim = content_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.content_dim

        self.sampling_offset_generator = nn.Sequential(
            nn.Linear(content_dim, in_points * n_groups * 3)
        )

        self.norm = nn.LayerNorm(content_dim)

        self.adaptive_mixing = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.content_dim,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups,
        )

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator[-1].weight)
        nn.init.zeros_(self.sampling_offset_generator[-1].bias)

        bias = self.sampling_offset_generator[-1].bias.data.view(
            self.n_groups, self.in_points, 3)

        # if in_points are squared number, then initialize
        # to sampling on grids regularly, not used in most
        # of our experiments.
        if int(self.in_points ** 0.5) ** 2 == self.in_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)

        # initialize sampling delta z
        nn.init.constant_(bias[:, :, 2:3], -1.0)

        self.adaptive_mixing.init_weights()

    def forward(self, x, query_feat, query_xyzr, featmap_strides):
        offset = self.sampling_offset_generator(query_feat)

        sample_points_xyz = make_sample_points(
            offset, self.n_groups * self.in_points,
            query_xyzr,
        )

        if DEBUG:
            torch.save(
                sample_points_xyz, 'demo/sample_xy_{}.pth'.format(AdaptiveSamplingMixing._DEBUG))

        sampled_feature, _ = sampling_3d(sample_points_xyz, x,
                                         featmap_strides=featmap_strides,
                                         n_points=self.in_points,
                                         )

        if DEBUG:
            torch.save(
                sampled_feature, 'demo/sample_feature_{}.pth'.format(AdaptiveSamplingMixing._DEBUG))
            AdaptiveSamplingMixing._DEBUG += 1

        query_feat = self.adaptive_mixing(sampled_feature, query_feat)
        query_feat = self.norm(query_feat)

        return query_feat


def position_embedding(token_xyzr, num_feats, temperature=10000):
    assert token_xyzr.size(-1) == 4
    term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=4).flatten(2)
    return pos_x


@HEADS.register_module()
class AdaMixerDecoderStage(BBoxHead):
    _DEBUG = -1

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(AdaMixerDecoderStage, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 4)

        self.in_points = in_points
        self.n_groups = n_groups
        self.out_points = out_points

        self.sampling_n_mixing = AdaptiveSamplingMixing(
            content_dim=content_dim,  # query dim
            feat_channels=feat_channels,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups
        )

        self.iof_tau = nn.Parameter(torch.ones(self.attention.num_heads, ))

    @torch.no_grad()
    def init_weights(self):
        super(AdaMixerDecoderStage, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                nn.init.xavier_uniform_(m.weight)

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        nn.init.uniform_(self.iof_tau, 0.0, 4.0)

        self.sampling_n_mixing.init_weights()

    @auto_fp16()
    def forward(self,
                x,
                query_xyzr,
                query_content,
                featmap_strides):
        N, n_query = query_content.shape[:2]

        AdaMixerDecoderStage._DEBUG += 1

        with torch.no_grad():
            rois = decode_box(query_xyzr)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(query_xyzr, query_content.size(-1) // 4)

        '''IoF'''
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)

        query_content = query_content.permute(1, 0, 2)
        pe = pe.permute(1, 0, 2)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe
        query_content = self.attention(
            query_content_attn,
            attn_mask=attn_bias,
        )
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)

        ''' adaptive 3D sampling and mixing '''
        query_content = self.sampling_n_mixing(
            x, query_content, query_xyzr, featmap_strides)

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content))

        cls_feat = query_content
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(N, n_query, -1)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        return cls_score, xyzr_delta, query_content.view(N, n_query, -1)

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes

        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
