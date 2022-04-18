point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = ['car']
voxel_size = [0.2, 0.2, 8]
# out_size_factor = 4
# evaluation = dict(interval=1)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
# input_modality = dict(
#     use_lidar=True,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=False)
# img_scale = (800, 448)
# num_views = 6
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),
    # dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    # dict(type='MyNormalize', **img_norm_cfg),
    # dict(type='MyPad', size_divisor=32),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=img_scale,
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1.0, 1.0],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D'),
    #         dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    #         dict(type='MyNormalize', **img_norm_cfg),
    #         dict(type='MyPad', size_divisor=32),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points', 'img'])
    #     ])
]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=6,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             num_views=num_views,
#             ann_file=data_root + '/nuscenes_infos_train.pkl',
#             load_interval=1,
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             box_type_3d='LiDAR')),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         num_views=num_views,
#         ann_file=data_root + '/nuscenes_infos_val.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         num_views=num_views,
#         ann_file=data_root + '/nuscenes_infos_val.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'))
model = dict(
    type='BevCameraDetector',
    freeze_img=False,
    point_cloud_range=point_cloud_range,
    voxel_size = voxel_size,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    # pts_voxel_layer=dict(
    #     max_num_points=20,
    #     voxel_size=voxel_size,
    #     max_voxels=(30000, 60000),
    #     point_cloud_range=point_cloud_range),
    # pts_voxel_encoder=dict(
    #     type='PillarFeatureNet',
    #     in_channels=5,
    #     feat_channels=[64],
    #     with_distance=False,
    #     voxel_size=voxel_size,
    #     norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
    #     point_cloud_range=point_cloud_range,
    # ),
    # pts_middle_encoder=dict(
    #     type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)
    # ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 6
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'checkpoints/fusion_pillar02_R50.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
freeze_lidar_components = True
find_unused_parameters = True
