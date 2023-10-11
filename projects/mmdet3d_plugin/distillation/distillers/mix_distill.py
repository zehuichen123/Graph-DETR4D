
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

@DISTILLER.register_module()
class MixDistill(BaseDetector):

    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 teacher_pretrained=None,
                 student_pretrained=None,
                 loss_cls_distill=None,
                 loss_reg_distill=None,
                 loss_feat_distill=None,
                 reweight_score=True,
                 init_student=False):

        super(MixDistill, self).__init__()
        self.use_teacher = True
        self.reweight_score = reweight_score
        if loss_feat_distill is None and loss_cls_distill is None \
                                        and loss_reg_distill is None:
            self.use_teacher = False

        if self.use_teacher:
            self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
            self.init_weights_teacher(teacher_pretrained)
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                      train_cfg=student_cfg.get('train_cfg'),
                                      test_cfg=student_cfg.get('test_cfg'))
        self.init_weights_student(student_pretrained)

        self.loss_cls_distill = loss_cls_distill
        self.loss_reg_distill = loss_reg_distill
        self.loss_feat_distill = loss_feat_distill

        if self.loss_feat_distill is not None:
            self.lateral_convs = nn.ModuleList()
            num_fpn_levels = 4
            for ii in range(num_fpn_levels):
                self.lateral_convs.append(nn.Conv2d(256, 256, 1, 1, 0))

    def base_parameters(self):
        if self.loss_feat_distill is not None:
            return nn.ModuleList([self.student, self.lateral_convs])
        return nn.ModuleList([self.student])

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if path is not None:
            checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        
    def init_weights_student(self, path=None):
        checkpoint = load_checkpoint(self.student, path, map_location='cpu')

    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        B,num_cam,c,h,w = img.size()
        if self.use_teacher:
            with torch.no_grad():
                self.teacher.eval()
                teacher_feats = self.teacher.extract_feat(img,img_metas)
                teacher_outs = self.teacher.pts_bbox_head.forward(teacher_feats,img_metas)

        # 由于extract_feat时候会改变img的size,因此要将img的size变回来
        img = img.view(B,num_cam,c,h,w)
        # 计算stu的前向输出以及loss
        stu_feats = self.student.extract_feat(img, img_metas)
        if self.loss_cls_distill is not None or self.loss_reg_distill is not None:
            student_outs = self.student.pts_bbox_head.forward(stu_feats, img_metas, teacher_queries=self.teacher.pts_bbox_head.query_embedding.weight)
        else:
            student_outs = self.student.pts_bbox_head.forward(stu_feats, img_metas)
        loss_inputs = [kwargs['gt_bboxes_3d'], kwargs['gt_labels_3d'], student_outs]
        student_loss = self.student.pts_bbox_head.loss(*loss_inputs)

        # use teacher query to output student outputs for distillation
        if self.loss_cls_distill is not None or self.loss_reg_distill is not None:
            instance_loss = self.get_instance_distill_loss(teacher_outs, student_outs)
            student_loss.update(instance_loss)

        if self.loss_feat_distill is not None:
            feat_loss = self.get_feat_distill_loss(teacher_feats, stu_feats)
            student_loss.update(feat_loss)
        return student_loss

    def get_feat_distill_loss(self, teacher_feats, student_feats):
        num_levels = len(teacher_feats)
        feat_loss = 0
        for level_id in range(num_levels):
            student_feat = student_feats[level_id]; teacher_feat = teacher_feats[level_id]
            bs, num_cams, num_c, w, h = student_feat.shape
            student_feat = student_feat.reshape(bs * num_cams, num_c, w, h)
            teacher_feat = teacher_feat.reshape(bs * num_cams, num_c, w, h)
            student_feat = self.lateral_convs[level_id](student_feat)
            distill_type = self.loss_feat_distill.get('type', 'vanilla')
            if distill_type == 'vanilla':
                feat_loss += F.mse_loss(student_feat, teacher_feat)
            elif distill_type == 'attention':
                T = 0.5
                g_c = torch.mean(torch.abs(teacher_feat), dim=1, keepdim=True).reshape(bs * num_cams, 1, w * h)
                g_s = torch.mean(torch.abs(teacher_feat), dim=(2, 3), keepdim=True)
                a_c = num_c * F.softmax(g_c / T, dim=2).reshape(bs * num_cams, 1, w, h)
                a_s = w * h * F.softmax(g_s / T, dim=1)
                _feat_loss = F.mse_loss(teacher_feat, student_feat, reduction='none')
                feat_loss += torch.mean(a_c * a_s * _feat_loss)
        return dict(feat_loss=self.loss_feat_distill['loss_weight'] * feat_loss / num_levels)  

    def get_instance_distill_loss(self, teacher_outs, student_outs):
        t_cls_scores = teacher_outs['all_cls_scores']
        t_bbox_preds = teacher_outs['all_bbox_preds']
        s_cls_scores = student_outs['guided_cls_scores']
        s_bbox_preds = student_outs['guided_bbox_preds']

        num_stage = len(t_cls_scores)
        instance_loss = dict()
        for stage_id in range(num_stage):
            t_cls_score = t_cls_scores[stage_id].detach(); t_bbox_pred = t_bbox_preds[stage_id].detach()
            s_cls_score = s_cls_scores[stage_id]; s_bbox_pred = s_bbox_preds[stage_id]

            t_cls_score = t_cls_score.sigmoid()
            q_score = torch.max(t_cls_score, dim=-1, keepdim=True)[0]
            num_class = t_cls_score.shape[-1]

            cls_loss = F.binary_cross_entropy_with_logits(s_cls_score, t_cls_score, reduction='none')
            reg_loss = F.l1_loss(s_bbox_pred, t_bbox_pred, reduction='none')
            
            # NOTE think that the q_score shape is 1,900,1, we broadcast it into the 1,900,10, therefore * num_class
            if self.reweight_score == True:
                cls_loss = torch.sum(q_score * cls_loss) / (torch.sum(q_score) * num_class + 1e-10) 
                reg_loss = torch.sum(q_score * reg_loss) / (torch.sum(q_score) * num_class + 1e-10)
            else:
                cls_loss = torch.mean(cls_loss)
                reg_loss = torch.mean(reg_loss)

            instance_loss['distill_loss_cls.%d' % stage_id] = cls_loss * self.loss_cls_distill['loss_weight']
            instance_loss['distill_loss_reg.%d' % stage_id] = reg_loss * self.loss_reg_distill['loss_weight']
        return instance_loss


    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img=img, img_metas=img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
