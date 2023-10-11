import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from torchvision import transforms
from mmdet.core import (multi_apply, multi_apply, reduce_mean, build_sampler, build_assigner)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d.models.builder import build_loss

ed = lambda x: x.unsqueeze(-1)
ed2 = lambda x: ed(ed(x))
ed3 = lambda x: ed(ed2(x))

class AddCAMCoords(nn.Module):

    def __init__(self,coord_maps,centered_coord,norm_coord_maps,with_r,bord_dist,scale_centered_coord,fov_maps,
            data_format='channels_last',
            resize_policy=None):

        self.coord_maps = coord_maps
        self.centered_coord = centered_coord
        self.norm_coord_maps = norm_coord_maps
        self.with_r = with_r
        self.bord_dist = bord_dist
        self.scale_centered_coord = scale_centered_coord
        self.fov_maps = fov_maps
        self.data_format=data_format
        self.resize_policy = resize_policy
        super(AddCAMCoords,self).__init__()


    def additional_channels(self):
        return self.coord_maps*2 + self.centered_coord*2 + self.norm_coord_maps*2 + self.with_r*1 + self.bord_dist*4 + self.fov_maps*2

    def _resize_map_(self,data,w,h):
        if self.data_format == 'channels_first':
            # 这里和tensorflow刚好相反
            return F.interpolate(data, size=(h, w))
            # resize = transforms.Resize([h, w])
            # return resize(data)
        else:
            # data_cl = convert_NCHW_to_NHWC(data) # data to channels last
            # torchvision.transforms.Resize([H,W])的作用是把最后两个维度resize成[H,W].
            # 所以，这对图像的通道顺序有要求
            data_cl = data.permute(0, 3, 2, 1)
            resize = transforms.Resize([w, h])
            data_cl_r = resize(data_cl)
            # return convert_NHWC_to_NCHW(data_cl_r)
            return data_cl_r.permute(0, 3, 2, 1)

    def __define_coord_channels__(self, bs, x_dim, y_dim):
        """
        Returns coord x and y channels from 0 to x_dim-1 and from 0 to y_dim -1
        """
        # [batch_size,ydim]
        xx_ones = torch.ones((bs,y_dim),dtype=torch.int32)
        # [batch_size,ydim,1]
        xx_ones = xx_ones.unsqueeze(-1)
        # [batch_size,x_dim]
        xx_range = torch.range(0,x_dim-1,dtype=torch.int32).unsqueeze(0).repeat(bs,1)
        # [bs,h,w]
        xx_channel = torch.matmul(xx_ones, xx_range)


        yy_ones = torch.ones((bs, x_dim), dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.range(0, y_dim - 1, dtype=torch.int32).unsqueeze(0).repeat(bs, 1)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones)

        if self.data_format == 'channels_last':
            xx_channel = xx_channel.unsqueeze(-1)
            yy_channel = yy_channel.unsqueeze(-1)
        else:
            xx_channel = xx_channel.unsqueeze(1)
            yy_channel = yy_channel.unsqueeze(1)

        xx_channel = xx_channel.float()
        yy_channel = yy_channel.float()

        return xx_channel, yy_channel


    def forward(self,input_tensor,h=0,w=0,cx=0,cy=0,fx=0,fy=0):
        """
           input_tensor: Tensor
               (N,H,W,C) if channels_last or (N,C,H,W) if channels_first
        """
        if self.additional_channels()==0:
            return input_tensor

        batch_size_tensor = input_tensor.shape[0]

        if self.data_format == 'channels_first':
            x_dim_tensor = input_tensor.shape[3]
            y_dim_tensor = input_tensor.shape[2]
            ax_concat = 1
        else:
            x_dim_tensor = input_tensor.shape[2]
            y_dim_tensor = input_tensor.shape[1]
            ax_concat = -1
        xx_channel,yy_channel = self.__define_coord_channels__(batch_size_tensor,w,h)
        xx_channel = xx_channel.to(cx.device)
        yy_channel = yy_channel.to(cx.device)


        extra_channels = []
        # 1) Normalized coordinates
        if self.norm_coord_maps:
            norm_xx_channel = (xx_channel / (w - 1)) * 2.0 - 1.0
            norm_yy_channel = (yy_channel / (h - 1)) * 2.0 - 1.0
            if self.with_r:
                norm_rr_channel = torch.sqrt(torch.square(norm_xx_channel - 0.5) + torch.square(norm_yy_channel - 0.5))
                extra_channels = extra_channels + [norm_xx_channel, norm_yy_channel, norm_rr_channel]
            else:
                extra_channels = extra_channels + [norm_xx_channel, norm_yy_channel]

        if self.centered_coord or self.fov_maps:
            # 2) Calculate Centered Coord
            # ed2 is equal to extend_dims twice
            cent_xx_channel = (xx_channel-ed2(cx)+0.5)
            cent_yy_channel = (yy_channel-ed2(cy)+0.5)

            if self.fov_maps:
                fov_xx_channel = torch.atan(cent_xx_channel/ed2(fx))
                fov_yy_channel = torch.atan(cent_yy_channel/ed2(fy))
                extra_channels = extra_channels + [fov_xx_channel,fov_yy_channel]
            # 4) Scaled Centered  coordinates
            if self.centered_coord:
                extra_channels = extra_channels + [cent_xx_channel/self.scale_centered_coord,cent_yy_channel/self.scale_centered_coord]

        # 5) Coord Maps (Unormalized, uncentered and unscaled)
        if self.coord_maps:
            extra_channels = extra_channels + [xx_channel, yy_channel]

        # Concat and resize
        if len(extra_channels)>0:
            extra_channels = torch.cat(extra_channels,axis=ax_concat)  # maybe
            extra_channels = self._resize_map_(extra_channels,x_dim_tensor,y_dim_tensor)
            extra_channels = [extra_channels]

        # 6) Distance to border in pixels in feature space.
        if self.bord_dist:
            t_xx_channel,t_yy_channel = self.__define_coord_channels__(batch_size_tensor,x_dim_tensor,y_dim_tensor)
            l_dist = t_xx_channel
            r_dist = x_dim_tensor.float() - t_xx_channel-1
            t_dist = t_yy_channel
            b_dist = y_dim_tensor.float() - t_yy_channel-1
            extra_channels = extra_channels + [l_dist,r_dist,t_dist,b_dist]

        # extra_channels = [tf.stop_gradient(e) for e in extra_channels] # Stop Gradients
        extra_channels = [e.detach() for e in extra_channels]  # Stop Gradients
        extra_channels = [e.to(input_tensor.device) for e in extra_channels]
        output_tensor = torch.cat(extra_channels+[input_tensor],axis=ax_concat)
        return output_tensor


class CAMConv(nn.Conv2d):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1,1),
                padding=(0,0),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
               coord_maps=False,
               centered_coord = True,
               scale_centered_coord = 320,
               norm_coord_maps = True,
               with_r = False,
               bord_dist = False,
               fov_maps = True,
               data_format='channels_first',
                add_channel = True,
               **kwargs):
        super(CAMConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_channel = add_channel
        self.add_cam_coords = AddCAMCoords(coord_maps, centered_coord, norm_coord_maps, with_r, bord_dist,
                                           scale_centered_coord, fov_maps, data_format=data_format)

    def forward(self,input_tensor,intrinsics,h,w,training = None,*args,**kwargs):
        """
            intrinsics:  Tensor
                intrinsic.shape = (N,4) where N is the batch size and 4 represents the 4 values in order (fx,fy,cx,cy)
            image_shape : TensorShape
                shape of the input image. One of: (N,H,W,C) if channels_last or (N,C,H,W) if channels_first or (N,H,W) or (H,W)
        """

        # bs,c,h,w = input_tensor.size()
        # # fx,fy,cx,cy = tf.split(intrinsics,num_or_size_splits=4,axis=-1)
        fx = torch.tensor(intrinsics[0][0],device=input_tensor.device)
        fy = torch.tensor(intrinsics[1][1],device=input_tensor.device)
        cx = torch.tensor(intrinsics[0][2],device=input_tensor.device)
        cy = torch.tensor(intrinsics[1][2],device=input_tensor.device)
        h = torch.tensor(h,device=input_tensor.device)
        w = torch.tensor(w,device=input_tensor.device)
        if self.add_channel:
            new_input_tensor = self.add_cam_coords(input_tensor,h=h,w=w,cx=cx,cy=cy,fx=fx,fy=fy)

            # 输入卷积中tensor的顺序为应该为 [bs,c,h,w]
            # new_input_tensor = new_input_tensor.permute(0,3,1,2)
            new_input_tensor = new_input_tensor.float()
        else:
            new_input_tensor = input_tensor.float()

        return super(CAMConv,self).forward(new_input_tensor.float(),*args,**kwargs)


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


class SELayerCAMConv(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid,
                 coord_maps=True,centered_coord=True,norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_last'):
        super().__init__()
        # self.conv_reduce = CAMConv(channels+8, channels+8, kernel_size=1, bias=True,coord_maps=True,centered_coord=True,
        #                            norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_first',add_channel=True)
        self.conv_reduce = nn.Conv2d(channels+8, channels+8, 1, bias=True)       
        self.act1 = act_layer()
        # self.conv_expand = CAMConv(channels+8, channels, kernel_size=1, bias=True,coord_maps=True,centered_coord=True,
        #                            norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_first',add_channel=False)
        self.conv_expand = nn.Conv2d(channels+8, channels, 1, bias=True)       
        self.gate = gate_layer()

    def forward(self, x, x_se,all_intrinsics,h,w):
        add_camcoords = AddCAMCoords(coord_maps=True,centered_coord=True,
                                   norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_first')
        new_x_se = []
        for idx in range(len(all_intrinsics)):
            fx = torch.tensor(all_intrinsics[idx][0][0])
            fy = torch.tensor(all_intrinsics[idx][1][1])
            cx = torch.tensor(all_intrinsics[idx][0][2])
            cy = torch.tensor(all_intrinsics[idx][1][2])
            new_x_se.append(add_camcoords(x_se[idx].unsqueeze(0),h=h,w=w,cx=cx,cy=cy,fx=fx,fy=fy))
        new_x_se = torch.stack(new_x_se,dim=0).squeeze(1)

        new_x_se = self.conv_reduce(new_x_se.float())           
        new_x_se = self.act1(new_x_se.float())                  
        new_x_se = self.conv_expand(new_x_se.float())           
        return x * self.gate(new_x_se.float())                  


class SELayerCAMConvV1(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid,
                 coord_maps=True,centered_coord=True,norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_last'):
        super().__init__()
        self.conv_reduce = CAMConv(channels+8, channels, kernel_size=1, bias=True,coord_maps=True,centered_coord=True,
                                   norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_first',add_channel=True)
        self.act1 = act_layer()
        self.conv_expand = CAMConv(channels+8, channels, kernel_size=1, bias=True,coord_maps=True,centered_coord=True,
                                   norm_coord_maps=True,with_r=False,bord_dist=False,scale_centered_coord=320,fov_maps=True,data_format='channels_first',add_channel=True)
        self.gate = gate_layer()

    def forward(self, x, x_se,all_intrinsics,h,w):
        all_x_se = []
        for idx in range(len(all_intrinsics)):
            x_se_tmp1 = self.conv_reduce(x_se[idx].unsqueeze(0),all_intrinsics[idx],h,w).squeeze(0)
            x_se_tmp2 = self.act1(x_se_tmp1)
            x_se_tmp3 = self.conv_expand(x_se_tmp2.unsqueeze(0),all_intrinsics[idx],h,w).squeeze(0)
            x_se[idx] = x_se_tmp3
            # all_x_se = [x_se_tmp3] + all_x_se
            all_x_se.append(x_se_tmp3)
        x_se = torch.stack(all_x_se,dim=0)
        return x * self.gate(x_se)



@HEADS.register_module()
class Detr3DHeadPE(DETRHead):
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

        super(Detr3DHeadPE, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        
        self.scale_pred = scale_pred
        self.with_detach = with_detach

    def _build_distill_modules(self, **kwargs):
        distill_assigner = kwargs['distill_assigner']
        self.distill_assigner = build_assigner(distill_assigner) if distill_assigner is not None else None
        distill_sampler_cfg = dict(type='PseudoSampler')
        self.distill_sampler = build_sampler(distill_sampler_cfg, context=self)
        self.loss_cls_distill = build_loss(kwargs['loss_cls_distill']) if kwargs['loss_cls_distill'] is not None else None
        self.loss_reg_distill = build_loss(kwargs['loss_reg_distill']) if kwargs['loss_reg_distill'] is not None else None

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

    def forward(self, mlvl_feats, img_metas, teacher_queries=None):
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

            coords_position_embedings[level_id] = self.fpe(coords_position_embedings[level_id].flatten(0,1),mlvl_feats[level_id].flatten(0,1)).view(
                                    mlvl_feats[level_id].size())

            pos_embed = coords_position_embedings[level_id]
            sin_embed = self.positional_encoding(masks[level_id])
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(mlvl_feats[level_id].size())
            pos_embed = pos_embed + sin_embed
            mlvl_feats[level_id] = mlvl_feats[level_id]+pos_embed

        # --------------------------------------------------------------------------
        query_embeds = self.query_embedding.weight
        # 从这里进入Detr3DTransformer
        # hs[6,1,900,256] init_reference=[1,900,3] inter_references[6,1,900,3]
        # hs是sample的feature      inter_references是每一层的rf points
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            # pos_embed,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501 reg_banches是回归分支
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

        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        if teacher_queries is not None:
            query_embeds = teacher_queries
            hs, init_reference, inter_references = self.transformer(
                mlvl_feats,
                query_embeds,
                # pos_embed,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501 reg_banches是回归分支
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

            outs['guided_cls_scores'] = outputs_classes
            outs['guided_bbox_preds'] = outputs_coords

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

    def loss_distill_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    reweight_score=False):
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
        cls_reg_targets = self.get_distill_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0).unsqueeze(-1)
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
        loss_cls = self.loss_cls_distill(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        if reweight_score:
            reweight_score = torch.max(labels, dim=-1, keepdims=True)[0]
            bbox_weights = bbox_weights * reweight_score
            teacher_fg_idx = labels[:, 0] != 10
            num_total_pos = torch.sum(reweight_score[teacher_fg_idx])

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_reg_distill(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def get_distill_targets(self,
                            cls_scores_list,
                            bbox_preds_list,
                            gt_bboxes_list,
                            gt_labels_list,
                            gt_bboxes_ignore_list=None,
                            img_metas=None):
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        img_meta_list = [
            img_metas for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_distill_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, img_meta_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    
    @force_fp32()
    def _get_distill_target_single(self,
                                   cls_score,
                                   bbox_pred,
                                   gt_labels,
                                   gt_bboxes,
                                   gt_bboxes_ignore=None,
                                   img_meta=None):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.distill_assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.distill_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, self.num_classes),
                                    self.num_classes,
                                    dtype=torch.float)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # # NOTE NOTE visualize the matching results
        # pos_cls_score = cls_score[pos_inds]
        # pos_bbox_pred = bbox_pred[pos_inds]
        # gt_bbox_pred = sampling_result.pos_gt_bboxes.cpu().detach().numpy()

        # bboxes = denormalize_bbox(pos_bbox_pred, self.pc_range)[:, :self.code_size - 1].cpu().detach().numpy()
        # # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        # # bboxes = LiDARInstance3DBoxes(bboxes, self.code_size - 1)

        # # gt_bbox_pred[:, 2] = gt_bbox_pred[:, 2] - gt_bbox_pred[:, 5] * 0.5
        # # gt_bboxes = LiDARInstance3DBoxes(gt_bbox_pred, self.code_size - 1)
        
        # from mmdet3d.core.visualizer.show_result import _write_oriented_bbox
        # import os

        # filename = img_meta['sample_idx']
        # result_path = 'distill_code/data/' + filename
        # os.makedirs(result_path, exist_ok=True)
        # # bottom center to gravity center
        # # gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
        # gt_bbox_pred[:, 6] *= -1
        # _write_oriented_bbox(gt_bbox_pred,
        #                      osp.join(result_path, f'{filename}_gt.obj'))
        
        # # bboxes[..., 2] += bboxes[..., 5] / 2
        # bboxes[:, 6] *= -1
        # _write_oriented_bbox(bboxes,
        #                      osp.join(result_path, f'{filename}_pred.obj'))
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

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
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

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
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
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
