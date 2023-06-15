import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy


@HEADS.register_module()
class InitialQueryGenerator(BaseModule):
    """
    This module produces initial content vector $\mathbf{q}$ and positional vector $(x, y, z, r)$.
    Note that the initial positional vector is **not** learnable.
    """

    def __init__(self,
                 num_query=100,
                 content_dim=256,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(InitialQueryGenerator, self).__init__(init_cfg)
        self.num_query = num_query
        self.content_dim = content_dim
        self._init_layers()

    def _init_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_query, 4)
        self.init_content_features = nn.Embedding(
            self.num_query, self.content_dim)

    def init_weights(self):
        super(InitialQueryGenerator, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas):
        """
        Hacks based on 'sparse_roi_head.py'.
        For the positional vector, we first compute (x, y, z, r) that fully covers an image. 
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        proposals = proposals * imgs_whwh

        xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
        wh = proposals[..., 2:4] - proposals[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()

        # NOTE: xyzr **not** learnable
        xyzr = torch.cat([xy, z, r], dim=-1).detach()

        init_content_features = self.init_content_features.weight.clone()
        init_content_features = init_content_features[None].expand(
            num_imgs, *init_content_features.size())

        init_content_features = torch.layer_norm(
            init_content_features, normalized_shape=[init_content_features.size(-1)])

        return xyzr, init_content_features, imgs_whwh

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, img, img_metas):
        """Forward function in training stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)
