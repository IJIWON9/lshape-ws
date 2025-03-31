import torch.onnx
import onnx
from onnxsim import simplify
import onnx_graphsurgeon as gs
import numpy as np
import os
from pathlib import Path
import onnx.numpy_helper as numpy_helper
import sys

sys.path.append(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent.parent, 'src/custom_pillar'))

from modify_onnx import simplify_preprocess, simplify_postprocess
# from modify_onnx_new import simplify_preprocess, simplify_postprocess
# from modify_onnx_orig import simplify_postprocess, simplify_preprocess

from models.detectors.pointpillar import PointPillar
from configs import cfg_from_yaml_file, cfg
from datasets.dataset import DatasetTemplate
from collections import defaultdict
from utils import common_utils
from models import load_data_to_gpu


def collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        try:
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['roi_boxes']:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['roi_scores', 'roi_labels']:
                max_gt = max([x.shape[1] for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                ret[key] = batch_gt_boxes3d

            elif key in ['gt_boxes2d']:
                max_boxes = 0
                max_boxes = max([len(x) for x in val])
                batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    if val[k].size > 0:
                        batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_boxes2d
            elif key in ["images", "depth_maps"]:
                # Get largest image size (H, W)
                max_h = 0
                max_w = 0
                for image in val:
                    max_h = max(max_h, image.shape[0])
                    max_w = max(max_w, image.shape[1])

                # Change size of images
                images = []
                for image in val:
                    pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                    pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                    pad_width = (pad_h, pad_w)
                    pad_value = 0

                    if key == "images":
                        pad_width = (pad_h, pad_w, (0, 0))
                    elif key == "depth_maps":
                        pad_width = (pad_h, pad_w)

                    image_pad = np.pad(image,
                                        pad_width=pad_width,
                                        mode='constant',
                                        constant_values=pad_value)

                    images.append(image_pad)
                ret[key] = np.stack(images, axis=0)
            elif key in ['calib']:
                ret[key] = val
            elif key in ["points_2d"]:
                max_len = max([len(_val) for _val in val])
                pad_value = 0
                points = []
                for _points in val:
                    pad_width = ((0, max_len-len(_points)), (0,0))
                    points_pad = np.pad(_points,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value)
                    points.append(points_pad)
                ret[key] = np.stack(points, axis=0)
            else:
                ret[key] = np.stack(val, axis=0)
        
        except:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

    ret['batch_size'] = batch_size

    return ret

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, points, training=False, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.points = points
    
    def get_points(self):
        input_dict = {
            'points': self.points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

cur_file_path = os.path.realpath(__file__)
pkg_root_path = Path(cur_file_path).parent.parent

ckpt_rel_path = 'weights/waymo_200.pth'
model_cfg_rel_path = 'configs/waymo_models/pointpillar_waymo.yaml'

ckpt_path = os.path.join(pkg_root_path, ckpt_rel_path)
model_cfg_path = os.path.join(pkg_root_path, model_cfg_rel_path)

pkg_src_root_path = Path(cur_file_path).parent

cfg_from_yaml_file(model_cfg_path, cfg)

dummy_points = np.load(os.path.join(pkg_src_root_path, 'points.npy'))
dummy_points = dummy_points[:,:4]
dummy_use_lead_xyz = np.load(os.path.join(pkg_src_root_path, 'use_lead_xyz.npy'))
dummy_voxels = np.load(os.path.join(pkg_src_root_path, 'voxels.npy'))
dummy_voxel_num_points = np.load(os.path.join(pkg_src_root_path, 'voxel_num_points.npy'))
dummy_voxel_coords = np.load(os.path.join(pkg_src_root_path, 'voxel_coords.npy'))
dummy_batch_size = 1

dummy_batch_dict = {
    'points': dummy_points,
    'use_lead_xyz': dummy_use_lead_xyz,
    'voxels': dummy_voxels,
    'voxel_coords': dummy_voxel_coords,
    'voxel_num_points': dummy_voxel_num_points,
    'batch_size': dummy_batch_size
}

pcl_np = None

demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=None, points=None
)

demo_dataset.points = dummy_points

with torch.no_grad():
    x_dict = demo_dataset.get_points()
    x = collate_batch([x_dict])
    load_data_to_gpu(x)

    model = PointPillar(model_cfg=cfg.MODEL,
                        num_class=len(cfg.CLASS_NAMES),
                        dataset=demo_dataset,
                        training=False).cuda()
    
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.cuda()
    model.eval()

    torch.onnx.export(model, 
                    x, 
                    'pillar_waymo_0415_2t.onnx',
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['points', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size'],
                    output_names=['pred_boxes', 'pred_labels', 'pred_scores'])
    
    onnx_raw = onnx.load(os.path.join(pkg_src_root_path, 'pillar_waymo_0415_2t.onnx'))
    onnx_trim_post = simplify_postprocess(onnx_raw)

    onnx_simp, check = simplify(onnx_trim_post)
    assert check, "Simplified ONNX model could not be validated"

    onnx_final = simplify_preprocess(onnx_simp)
    onnx.save(onnx_final, os.path.join(pkg_src_root_path, "pointpillar_final_channel_modified.onnx"))

