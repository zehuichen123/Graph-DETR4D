import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from torchvision import transforms
import torch.nn.functional as F
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

ed = lambda x: x.unsqueeze(-1)
ed2 = lambda x: ed(ed(x))
ed3 = lambda x: ed(ed2(x))


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@HEADS.register_module()
class HDetr3DHeadPE(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 depth_step=0.8,
                 depth_num=64,
                 depth_start=1,
                 scale_pred=True,
                 with_detach=True,
                 k_one2many,
                 lambda_one2many,
                 num_queries_one2one,
                 **kwargs):

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        # --------
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.position_dim = 3 * self.depth_num

        # self.num_query = num_query
        self.num_queries_one2one = num_queries_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many


        super(HDetr3DHeadPE, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self.scale_pred = scale_pred
        self.with_detach = with_detach



    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)


        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.fpe = SELayer(self.embed_dims)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        num_level = len(img_feats)
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        coords_position_embedings = []
        coords_masks = []


        for level_id in range(num_level):

            B, N, C, H, W = img_feats[level_id].shape
            coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
            coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

            # if self.LID:
            #     index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            #     index_1 = index + 1
            #     bin_size = (self.pc_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            #     coords_d = self.depth_start + bin_size * index * index_1
            # else:
            #     index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            #     bin_size = (self.pc_range[3] - self.depth_start) / self.depth_num
            #     coords_d = self.depth_start + bin_size * index

            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.pc_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1

            D = coords_d.shape[0]
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
            coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
            coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

            img2lidars = []
            for img_meta in img_metas:
                img2lidar = []
                for i in range(len(img_meta['lidar2img'])):
                    img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
                img2lidars.append(np.asarray(img2lidar))
            img2lidars = np.asarray(img2lidars)
            img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)
            # [1,12,W,H,D,4,4]
            coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
            img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
            coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]   # [1,12,W,H,D,3]
            coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
            coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
            coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])

            coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
            coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
            coords_mask = masks[level_id] | coords_mask.permute(0, 1, 3, 2)
            coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
            coords3d = inverse_sigmoid(coords3d)
            coords_position_embeding = self.position_encoder(coords3d)

            coords_position_embedings.append(coords_position_embeding.view(B, N, self.embed_dims, H, W))
            coords_masks.append(coords_mask)

        return coords_position_embedings,coords_masks

        # return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # for feat in mlvl_feats:
        #     print(feat.size())

        if self.with_detach:
            # for idx in range(len(mlvl_feats)):
            current_frame = mlvl_feats[0][:, :6]
            past_frame = mlvl_feats[0][:, 6:]
            mlvl_feats[0] = torch.cat([current_frame, past_frame.detach()], 1)

        #--------------------position embeding----------------------
        num_level = len(mlvl_feats)
        x = mlvl_feats[0]
        batch_size, num_cams = mlvl_feats[0].size(0), mlvl_feats[0].size(1)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = []
        for nl in range(num_level):
            mask = x.new_ones(
                (batch_size, num_cams, input_img_h, input_img_w))
            masks.append(mask)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                for level_id in range(num_level):
                    masks[level_id][img_id, cam_id, :img_h, :img_w] = 0

        # interpolate masks to have the same spatial shape with x
        for level_id in range(num_level):
            masks[level_id] = F.interpolate(
                masks[level_id], size=mlvl_feats[level_id].shape[-2:]).to(torch.bool)

        coords_position_embedings, _ = self.position_embeding(mlvl_feats, img_metas, masks)

        for level_id in range(num_level):
            # 新增加fpe
            coords_position_embedings[level_id] = self.fpe(coords_position_embedings[level_id].flatten(0,1),mlvl_feats[level_id].flatten(0,1),).view(
                                    mlvl_feats[level_id].size())


            pos_embed = coords_position_embedings[level_id]
            sin_embed = self.positional_encoding(masks[level_id])
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(mlvl_feats[level_id].size())
            pos_embed = pos_embed + sin_embed
            mlvl_feats[level_id] = mlvl_feats[level_id]+pos_embed

        # --------------------------------------------------------------------------
        query_embeds = self.query_embedding.weight

        # attn mask
        self_attn_mask = (
            torch.zeros([self.num_query, self.num_query, ]).bool().to(mlvl_feats[0].device)
        )
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:] = True

        # 从这里进入Detr3DTransformer
        # hs[6,1,900,256] init_reference=[1,900,3] inter_references[6,1,900,3]
        #  inter_references是每一层的rf points
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            # pos_embed,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501 reg_banches是回归分支
            decoder_self_attn_mask=[self_attn_mask, None],
            img_metas=img_metas,
        )
        # print(weights[0].size())
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # 分类 [bs,900,10]
            outputs_class = self.cls_branches[lvl](hs[lvl])
            # 回归  [bs,900,10]
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            if self.scale_pred:

                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]) * img_metas[0]['depth_factors'][0]
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]) * img_metas[0]['depth_factors'][0]
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]) * img_metas[0]['depth_factors'][0]

            else:
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outputs_classes_one2one = outputs_classes[:, :, 0 : self.num_queries_one2one, :]
        outputs_coords_one2one = outputs_coords[:, :, 0 : self.num_queries_one2one, :]
        outputs_classes_one2many = outputs_classes[:, :, self.num_queries_one2one :, :]
        outputs_coords_one2many = outputs_coords[:, :, self.num_queries_one2one :, :]

        outs = {
            'all_cls_scores': outputs_classes_one2one,
            'all_bbox_preds': outputs_coords_one2one,
            'all_cls_scores_one2many': outputs_classes_one2many,
            'all_bbox_preds_one2many': outputs_coords_one2many,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        # outs = {
        #     'all_cls_scores': outputs_classes,
        #     'all_bbox_preds': outputs_coords,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None,
        # }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # [1,900,10]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]  # [1,900,10]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_cls_scores_one2many = preds_dicts['all_cls_scores_one2many']
        all_bbox_preds_one2many = preds_dicts['all_bbox_preds_one2many']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # for one2one
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # for one2many
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []
        for gt_bboxes in gt_bboxes_list:
            one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))

        for gt_labels in gt_labels_list:
            one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))

        all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list

        # one2one loss
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        # one2many loss
        losses_cls_one2many, losses_bbox_one2many = multi_apply(
            self.loss_single, all_cls_scores_one2many, all_bbox_preds_one2many,
            all_gt_bboxes_list_one2many, all_gt_labels_list_one2many,
            all_gt_bboxes_ignore_list_one2many)


        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        # loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_cls'] = losses_cls[-1] + losses_cls_one2many[-1] * self.lambda_one2many
        loss_dict['loss_bbox'] = losses_bbox[-1] + losses_bbox_one2many[-1] * self.lambda_one2many

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_cls_i_one2many, loss_bbox_i_one2many in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_cls_one2many[:-1],
                                           losses_bbox_one2many[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i + loss_cls_i_one2many * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i + loss_bbox_i_one2many * self.lambda_one2many
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            # print(bboxes.size())
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
