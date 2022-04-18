import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector

import kornia


@DETECTORS.register_module()
class BevCameraDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(BevCameraDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

        self.voxel_size = kwargs.get('voxel_size', True)
        self.point_cloud_range = kwargs.get('point_cloud_range', True)

        self.init_grid_generator()
    
    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(BevCameraDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def init_grid_generator(self):
        # Calculate voxel size
        self.voxel_size = torch.as_tensor(self.voxel_size)
        self.point_cloud_range = torch.as_tensor(self.point_cloud_range).reshape(2, 3)
        self.voxels_size = (self.point_cloud_range[1] - self.point_cloud_range[0]) / self.voxel_size

        # Create voxel grid
        self.width, self.depth, self.height = self.voxels_size.int() # x:width / y:depth / z:height
        self.voxel_grid = kornia.utils.create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False)

        self.voxel_grid = self.voxel_grid.permute(0, 3, 1, 2, 4)  # depth 1Y height 2Z width 3X-> XYZ
        print ('voxel_grid ', self.voxel_grid.shape)

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.trans_grid_to_lidar = self.grid_to_lidar_unproject()
        self.lidar_grid = kornia.transform_points(trans_01=self.trans_grid_to_lidar, points_1=self.voxel_grid)

    def grid_to_lidar_unproject(self):
        x_size, y_size, z_size = self.voxels_size
        x_min, y_min, z_min = self.point_cloud_range[0]
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=torch.float32)  # (4, 4)
        return unproject      

    def create_2D_grid(self, img_feats, img_metas, lidar_grid):
        batch_size = img_feats.shape[0]
        lidar_pts_4d = lidar_grid.repeat_interleave(repeats=batch_size, dim=0)
        print ("lidar_grid.shape ", lidar_grid.shape)
        lidar2img_front = torch.zeros((batch_size, 4, 4), device=img_feats.device, dtype=img_feats.device)
        for sample_idx in range(batch_size):
            lidar2img_front[sample_idx] = torch.as_tensor(img_metas[sample_idx]['lidar2img'][1], device=img_feats.device, dtype=img_feats.device)
        lidar2img_front.reshape(batch_size, 1, 1, 4, 4)   
        pts_2d = lidar2img_front @ lidar_pts_4d
        pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        print ("pts_2d.shape ", pts_2d.shape)
        min_n = -1
        max_n = 1
        img_shape = img_metas[0]['img_shape'][:2]
        # Subtract 1 since pixel indexing from [0, shape - 1]
        pts_2d = pts_2d / (img_shape - 1) * (max_n - min_n) + min_n
        return pts_2d                                

    def trans_img_feats_to_voxel_feats(self, img_feats, img_metas):
        # 1.Generate sampling grid for img features
        grid_2d = self.create_2D_grid(self, img_feats, img_metas, self.lidar_grid)
        # 2.Sample img features to generate voxel features
        voxel_features = F.grid_sample(input=img_feats, grid=grid_2d, mode="bilinear", padding_mode="zeros")
        return voxel_features

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        # voxels, num_points, coors = self.voxelize(pts)
        # voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
        #                                         )
        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.trans_img_feats_to_voxel_feats(img_feats, img_metas)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
