import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
import warnings
from mmcv.runner.base_module import BaseModule
import math
import warnings
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps, max=1)
    x2 = (1 - x).clamp(min=eps, max=1)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class Deform3DCrossAttnMP(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 fix_offset=False,
                 depth_encode=False,
                 multi_points=True):
        super(Deform3DCrossAttnMP, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fix_offset = fix_offset
        self.depth_encode = depth_encode
        self.multi_points = multi_points
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.cam_attention_weights = nn.Linear(embed_dims,
                                               num_cams)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(4 if self.depth_encode else 3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        # NOTE Deform Weights
        # TODO check if need to differentiate offset in different levels
        self.deform_sampling_offsets = nn.Linear(
            embed_dims, num_heads * 1 * num_points * 3)

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * self.num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        if self.multi_points:
            self.attention_weights_neighbor = nn.Linear(embed_dims, num_heads * num_levels * self.num_points * 8//4)

            self.deform_sampling_offsets_neighbor = nn.Linear(embed_dims, num_heads * 1 * num_points * 3)
            self.output_weight = nn.Linear(embed_dims*2, 2)

        self.init_weight()

        if self.fix_offset:
            self.deform_sampling_offsets.weight.requires_grad = False
            self.deform_sampling_offsets.bias.requires_grad = False



    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.cam_attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

        constant_init(self.deform_sampling_offsets, 0.)
        ### TODO initilize offset bias for DCN
        # print("NEED TO initialize DCN offset")
        # time.sleep(1)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            3).repeat(1, 1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.deform_sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)

        # if self.multi_points:
        #     constant_init(self.deform_sampling_offsets_neighbor, 0.)
        #     self.deform_sampling_offsets_neighbor.bias.data = grid_init.view(-1)
        #     constant_init(self.attention_weights_neighbor, val=0., bias=0.)


    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # print('forward', torch.sum(self.deform_sampling_offsets.weight))
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        # if query_pos is not None:
        #     query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        # [bs,12,900,1]
        cam_attention_weights = self.cam_attention_weights(query).view(
            bs, self.num_cams, num_query, 1)

        # decompose reference points
        reference_points_center = reference_points[:, :num_query]

        # prepare for deformable attention
        lidar2img = [];
        img_metas = kwargs['img_metas'];
        pc_range = self.pc_range
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)  # (B,12,4,4)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points_center = reference_points_center.clone()
        reference_points_center_3d = reference_points_center.clone()

        reference_points_center[..., 0:1] = reference_points_center[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points_center[..., 1:2] = reference_points_center[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points_center[..., 2:3] = reference_points_center[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # add offset before projecting them onto 2d plane
        # self.deform_sampling_offsets 为 Linear(in_features=256, out_features=96, bias=True)  96 = self.num_heads*self.num_points*3
        # [1,900,256]  -->  [1, 900, 8, 4, 4, 3]
        sampling_offsets = self.deform_sampling_offsets(query).view(
            bs, num_query, self.num_heads, 1, self.num_points, 3).repeat(1, 1, 1, self.num_levels, 1, 1)
        # [1,900,8,4,4,3]
        reference_points_center = reference_points_center.view(bs, num_query, 1, 1, 1, 3) + sampling_offsets
        reference_points_center = reference_points_center.view(bs, num_query * self.num_heads * self.num_levels * self.num_points,
                                                 3)  # [1,115200,3] 115200=900*8*4*4
        # reference_points (B, num_queries, 4)
        reference_points_center = torch.cat((reference_points_center, torch.ones_like(reference_points_center[..., :1])), -1)
        B, num_query_fake = reference_points_center.size()[:2]
        num_cam = lidar2img.size(1)  # 12
        reference_points_center = reference_points_center.view(B, 1, num_query_fake, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        center_lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query_fake, 1, 1)
        # [1,12,115200,2]  [1,12,115200,4]??
        reference_points_cam = torch.matmul(center_lidar2img, reference_points_center).squeeze(-1)
        eps = 1e-5
        mask_center = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        # reference_points_cam = (reference_points_cam - 0.5) * 2
        # mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
        #             & (reference_points_cam[..., 0:1] < 1.0)
        #             & (reference_points_cam[..., 1:2] > -1.0)
        #             & (reference_points_cam[..., 1:2] < 1.0))
        mask_center = (mask_center & (reference_points_cam[..., 0:1] > 0.)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 1:2] > 0.)
                & (reference_points_cam[..., 1:2] < 1.0))
        # mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
        # mask = mask.view(B, num_cam, num_query_fake, 1)
        # mask = torch.nan_to_num(mask)
        # HARD CODE HERE NOTE
        nan_mask = torch.isnan(mask_center)
        mask_center[nan_mask] = 0.
        # NOTE
        # valid_masks = [];
        src_flattens = [];
        spatial_shapes = []
        # mask_flattens = [];
        lvl_pos_embed_flatten = []
        # 逐特征层
        for i in range(len(value)):
            bs, n, c, h, w = value[i].shape  # n=12
            spatial_shapes.append((h, w))  # 每一层的h和w
            # [12,h*w,256]
            flatten_feat = value[i].view(bs * n, c, h, w).flatten(2).transpose(1, 2)
            src_flattens.append(flatten_feat)

        value_flatten = torch.cat(src_flattens, 1)
        # mask_flattens = torch.cat(mask_flattens, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=flatten_feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points_cam = reference_points_cam.view(B * self.num_cams, num_query, self.num_heads, self.num_levels,
                                                         self.num_points, 2)

        # print(reference_points_cam.shape)
        query_cam = query.repeat(self.num_cams, 1, 1)  # [12,900,256]
        value_flatten = self.value_proj(value_flatten)
        _, num_value, _ = value_flatten.size()
        # [12,num_value,8,32]
        value_flatten = value_flatten.view(bs * self.num_cams, num_value, self.num_heads, -1)
        attention_weights = self.attention_weights(query_cam).view(
            bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_points)
        mask_center = mask_center.view(bs * self.num_cams, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1) * mask_center

        # if reference_points_cam.shape[-1] == 2:
        #     offset_normalizer = torch.stack(
        #         [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        #     sampling_locations = reference_points_cam[:, :, None, :, None, :] \
        #         + sampling_offsets \
        #         / offset_normalizer[None, None, None, :, None, :]
        # elif reference_points_cam.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #         + sampling_offsets / self.num_points \
        #         * reference_points_cam[:, :, None, :, None, 2:] \
        #         * 0.5
        # else:
        #     raise ValueError(
        #         f'Last dim of reference_points must be'
        #         f' 2 or 4, but get {reference_points_cam.shape[-1]} instead.')

        # MultiScaleDeformableAttnFunction 在mmcv ops下
        if torch.cuda.is_available() and value_flatten.is_cuda:
            # output [bs,num_queries,embed_dims] [12,900,256]
            output = MultiScaleDeformableAttnFunction.apply(
                value_flatten, spatial_shapes, level_start_index, reference_points_cam,
                attention_weights, self.im2col_step)
        else:
            # WON'T REACH HERE
            print("Won't Reach Here")
            output = multi_scale_deformable_attn_pytorch(
                value_flatten, spatial_shapes, sampling_locations, attention_weights)

        # output = torch.nan_to_num(output)
        # mask = torch.nan_to_num(mask)

        # nan_mask = torch.isnan(output)
        # output[nan_mask] = 0.

        # nan_mask = torch.isnan(mask)
        # mask[nan_mask] = 0.

        cam_attention_weights = cam_attention_weights.sigmoid()
        # attention_weights = attention_weights.permute(0, 2, 3, 1)
        output = output.view(bs, self.num_cams, num_query, -1)
        output = output * cam_attention_weights
        # 把所有相机上的都相加
        output = output.sum(1)

        # output = self.output_proj(output)
        # output = output.permute(1, 0, 2)
        # (num_query, bs, embed_dims) (900,bs,256)


        if self.multi_points:
            # 这些padding的点就不需要再取offset了，直接投在img上取feature就好
            reference_points_neighbor = reference_points[:, num_query:]   # [1,num_query*8,3]
            # [12,num_query*8,8,16]
            attention_weights_neighbor = self.attention_weights_neighbor(query_cam).view(
            bs * self.num_cams, num_query*8, self.num_heads, self.num_levels * self.num_points//4)

            reference_points_neighbor[..., 0:1] = reference_points_neighbor[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points_neighbor[..., 1:2] = reference_points_neighbor[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points_neighbor[..., 2:3] = reference_points_neighbor[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

            query_neighbor = query.repeat(1,8,1)
            # add offset before projecting them onto 2d plane  [1,num_query*8,8,4,4,3]
            # sampling_offsets_neighbor = self.deform_sampling_offsets_neighbor(query_neighbor).view(
            #     bs, num_query*8, self.num_heads, 1, self.num_points, 3).repeat(1, 1, 1, self.num_levels, 1, 1)
            # reference_points_neighbor = reference_points_neighbor.view(bs, num_query*8, 1, 1, 1, 3) + sampling_offsets_neighbor   #[1,num_query*8,8,4,4,3]

            # 这里不取做deformable形式,因此num_points 为1
            reference_points_neighbor = reference_points_neighbor.view(bs,num_query*8,1,1,1,3).repeat(1,1,self.num_heads,self.num_levels,self.num_points//4,1)
            reference_points_neighbor = reference_points_neighbor.view(bs, num_query*8 * self.num_heads * self.num_levels * self.num_points//4,3)

            reference_points_neighbor = torch.cat((reference_points_neighbor, torch.ones_like(reference_points_neighbor[..., :1])), -1)
            B, num_query_fake = reference_points_neighbor.size()[:2]

            reference_points_neighbor = reference_points_neighbor.view(B, 1, num_query_fake, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
            neighbor_lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query_fake, 1, 1)
            reference_points_neighbor_cam = torch.matmul(neighbor_lidar2img, reference_points_neighbor).squeeze(-1)

            mask_neighbor = (reference_points_neighbor_cam[..., 2:3] > eps)
            reference_points_neighbor_cam = reference_points_neighbor_cam[..., 0:2] / torch.max(
                reference_points_neighbor_cam[..., 2:3], torch.ones_like(reference_points_neighbor_cam[..., 2:3]) * eps)
            reference_points_neighbor_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
            reference_points_neighbor_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

            mask_neighbor = (mask_neighbor & (reference_points_neighbor_cam[..., 0:1] > 0.)
                    & (reference_points_neighbor_cam[..., 0:1] < 1.0)
                    & (reference_points_neighbor_cam[..., 1:2] > 0.)
                    & (reference_points_neighbor_cam[..., 1:2] < 1.0))
            nan_mask = torch.isnan(mask_neighbor)
            mask_neighbor[nan_mask] = 0.
            reference_points_neighbor_cam = reference_points_neighbor_cam.view(B * self.num_cams, num_query*8, self.num_heads,
                                                             self.num_levels, self.num_points//4, 2)
            mask_neighbor = mask_neighbor.view(bs * self.num_cams, num_query*8, self.num_heads, self.num_levels * self.num_points//4)
            attention_weights_neighbor = attention_weights_neighbor.softmax(-1) * mask_neighbor

            if torch.cuda.is_available() and value_flatten.is_cuda:
                output_neighbor = MultiScaleDeformableAttnFunction.apply(
                    value_flatten, spatial_shapes, level_start_index, reference_points_neighbor_cam,
                    attention_weights_neighbor, self.im2col_step)
            else:
                # WON'T REACH HERE
                print("Won't Reach Here")
                output = multi_scale_deformable_attn_pytorch(
                    value_flatten, spatial_shapes, sampling_locations, attention_weights)

            cam_attention_weights = self.cam_attention_weights(query_neighbor).view(
                bs, self.num_cams, num_query*8, 1)
            # cam_attention_weights = cam_attention_weights.repeat(1,1,8,1)


            output_neighbor = output_neighbor.view(bs, self.num_cams, num_query*8, -1)
            output_neighbor = output_neighbor * cam_attention_weights

            output_neighbor = output_neighbor.view(bs,self.num_cams,8,num_query,-1)
            # 把所有相机上的都相加
            output_neighbor = output_neighbor.sum(1).sum(1)   # [1,num_query,256]

            output_tmp = torch.cat([output, output_neighbor], dim=-1)  # [1,num_query,512]
            output_weight = self.output_weight(output_tmp)
            output_weight = output_weight.sum(1).softmax(-1)
            output = output*output_weight[0][0]+output_neighbor*output_weight[0][1]

        output = self.output_proj(output)  # [1,300,256]
        output = output.permute(1, 0, 2)
        # add depth into positional embedding
        if self.depth_encode:  # false
            depth_data = (reference_points_center_3d[..., 0:1] ** 2 + reference_points_center_3d[..., 1:2] ** 2) ** 0.5
            reference_points_3d = torch.cat([reference_points_center_3d, depth_data], dim=-1)

        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_center_3d)).permute(1, 0, 2)
        # print(output.shape, inp_residual.shape, pos_feat.shape)
        res = self.dropout(output) + inp_residual + pos_feat
        # print(res.shape)  [bs,900,256]

        return res

def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    # mask = torch.nan_to_num(mask)
    # HARD CODE HERE NOTE
    nan_mask = torch.isnan(mask)
    mask[nan_mask] = 0.
    # NOTE
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask