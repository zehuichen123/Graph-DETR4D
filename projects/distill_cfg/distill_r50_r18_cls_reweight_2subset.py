_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# model settings
# find_unused_parameters=True

distiller = dict(
    type='Detr4D_Distiller',
    teacher_pretrained = 'work_dirs/detr4d_res50_deform_pe_testaug_320_fullset_ceph/epoch_24.pth',
    student_pretrained = 'data/pretrain_models/fcos3d_r18_new.pth',
    init_student = False,
    loss_reg_distill=dict(type='L1Loss', loss_weight=0.0),
    loss_cls_distill=dict(type='DistillCrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    reweight_score=True,
    # loss_feat_distill=dict(loss_weight=0.0),
    distill_assigner=dict(
        type='DistillHungarianAssigner3D',
        cls_cost=dict(
            type='DistillCrossEntropyLossCost',
            # use_sigmoid=True,
            weight=1.0),
        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
        iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
        pc_range=point_cloud_range),
)
#-------------------------------

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         # '/data/Dataset/nuScenes/': 'cluster1:s3://JjwBucket/nuscenes/',
#         # 'data/nuscenes/': 'cluster1:s3://JjwBucket/nuscenes/',
#         'data/nuscenes/': 'czh:s3://czhBucket/nuscenes/',

#     }))

file_client_args = dict(
    backend='disk')

ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, file_client_args=file_client_args, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d','box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename','intrinsics'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, file_client_args=file_client_args, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d','box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename','intrinsics'))
        ])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_train_tiny.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        load_interval=2,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, data_root=data_root,ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val_tiny.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, data_root=data_root,ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val_tiny.pkl', classes=class_names, modality=input_modality))
#-------------------------------

student_cfg = 'projects/configs/detr4d/detr4d_res18_deform_pe_testaug_320_fullset_ceph.py'
teacher_cfg = 'projects/configs/detr4d/detr4d_res50_deform_pe_testaug_320_fullset_ceph.py'


dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=6, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])