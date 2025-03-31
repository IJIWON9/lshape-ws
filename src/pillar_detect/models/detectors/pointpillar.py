import os

import torch
import torch.nn as nn

from ..backbones_3d.vfe.pillar_vfe import PillarVFE
from ..backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from ..backbones_2d.base_bev_backbone import BaseBEVBackbone
from ..dense_heads.anchor_head_single import AnchorHeadSingle
from ..model_utils import model_nms_utils
from ops.iou3d_nms import iou3d_nms_utils

class PointPillar(nn.Module):
    def __init__(self, model_cfg, num_class, dataset, training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.training = training

        self.model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }

        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.build_network()
        self.module_list = self.model_info_dict['module_list']


    def forward(self, points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size):
        points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features = self.vfe.forward(points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size)
        
        points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features = self.map2bev_module.forward(points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features)

        points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features, spatial_features_2d = self.backbone_2d.forward(points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features)

        points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features, spatial_features_2d, batch_cls_preds, batch_box_preds, cls_preds_normalized = self.dense_head.forward(points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features, spatial_features_2d)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict

        else:
            pred_dicts = self.post_processing(points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features, spatial_features_2d, batch_cls_preds, batch_box_preds, cls_preds_normalized)
            return pred_dicts
        
    
    def build_network(self):
        self.vfe = PillarVFE(model_cfg=self.model_cfg.VFE,
                                    num_point_features=self.model_info_dict['num_rawpoint_features'],
                                    point_cloud_range=self.model_info_dict['point_cloud_range'],
                                    voxel_size=self.model_info_dict['voxel_size'],
                                    grid_size=self.model_info_dict['grid_size'],
                                    depth_downsample_factor=self.model_info_dict['depth_downsample_factor']
                                    ).cuda()
        
        self.map2bev_module = PointPillarScatter(model_cfg=self.model_cfg.MAP_TO_BEV,
                                                 grid_size=self.model_info_dict['grid_size']
                                                 ).cuda()
        
        self.backbone_2d = BaseBEVBackbone(model_cfg=self.model_cfg.BACKBONE_2D,
                                           input_channels=self.map2bev_module.num_bev_features
                                           ).cuda()
        
        self.dense_head = AnchorHeadSingle(self.model_cfg.DENSE_HEAD,
                                           input_channels=self.backbone_2d.num_bev_features,
                                           num_class=self.num_class,
                                           class_names=self.class_names,
                                           grid_size=self.model_info_dict['grid_size'],
                                           point_cloud_range=self.model_info_dict['point_cloud_range'],
                                           predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
                                           voxel_size=self.model_info_dict.get('voxel_size', False)
                                           ).cuda()
        
        self.model_info_dict['module_list'].append(self.vfe)
        self.model_info_dict['module_list'].append(self.map2bev_module)
        self.model_info_dict['module_list'].append(self.backbone_2d)
        self.model_info_dict['module_list'].append(self.dense_head)

        self.module_topology = [
            'vfe', 'map_to_bev_module', 'backbone_2d', 'dense_head'
        ]

        for i in range(4):
            self.add_module(self.module_topology[i], self.model_info_dict['module_list'][i])
            

    def post_processing(self, points, use_lead_xyz, voxels, voxel_coords, voxel_num_points, batch_size, pillar_features, spatial_features, spatial_features_2d, batch_cls_preds, batch_box_preds, cls_preds_normalized, **kwargs):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        # batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if 'batch_index' in kwargs.keys():
                assert batch_box_preds.shape.__len__() == 2
                batch_mask = (kwargs['batch_index'] == index)
            else:
                assert batch_box_preds.shape.__len__() == 3
                batch_mask = index

            box_preds = batch_box_preds[batch_mask]
            src_box_preds = box_preds
            
            if not isinstance(batch_cls_preds, list):
                cls_preds = batch_cls_preds[batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not cls_preds_normalized:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_cls_preds]
                src_cls_preds = cls_preds
                if not cls_preds_normalized:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = kwargs["multihead_label_mapping"]

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if 'has_class_labels' in kwargs.keys():
                    if kwargs['has_class_labels'] == False:
                        label_key = 'roi_labels' if 'roi_labels' in kwargs.keys() else 'batch_pred_labels'
                        label_preds = kwargs[label_key][index]
                else:
                    label_preds = label_preds + 1 
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                    
            # recall_dict = self.generate_recall_record(
            #     box_preds=final_boxes if 'rois' not in kwargs.keys() else src_box_preds,
            #     recall_dict=recall_dict, batch_index=index, data_dict=kwargs,
            #     thresh_list=post_process_cfg.RECALL_THRESH_LIST
            # )        

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts
    
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
    
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.densehead_module.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict