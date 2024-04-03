import torch
import numpy as np
import pickle
import copy
from tqdm import tqdm

from pathlib import Path

from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..models import load_data_to_gpu
from ..datasets import build_dataloader
from ..ops.iou3d_nms import iou3d_nms_utils
import torch.distributed as dist


class GTDataCollector:
    def __init__(self, sampler_cfg, model, dataloader):
        self.sampler_cfg = sampler_cfg
        self.interval = sampler_cfg['INTERVAL']
        self.model = model
        self.dataloader = dataloader
        self.root_path = dataloader.dataset.root_path
        self.class_names = dataloader.dataset.class_names

        if self.sampler_cfg['Dataset'] == 'KITTI':
            self.database_save_path = Path(self.root_path) / 'gt_database_runtime_train'
            self.db_info_save_path = Path(self.root_path) / 'kitti_dbinfos_score_train.pkl'
            imageset_file = self.root_path / 'ImageSets' / 'train.txt'
            self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)
        elif self.sampler_cfg['Dataset'] == 'Waymo':
            self.database_save_path = Path(self.root_path) / 'gt_database_runtime_train'
            self.db_info_save_path = Path(self.root_path) / 'waymo_processed_data_v0_5_0_waymo_dbinfos_score_train_sampled_1.pkl'  
            imageset_file = self.root_path / 'ImageSets' / 'train.txt'
            with open(imageset_file, 'r') as file:
                self.labeled_mask = [line.strip() for line in file.readlines()]
        elif self.sampler_cfg['Dataset'] == 'ONCE':
            self.database_save_path = Path(self.root_path) / 'gt_database_runtime_train'
            self.db_info_save_path = Path(self.root_path) / 'once_dbinfos_score_train.pkl'
            imageset_file = self.root_path / 'ImageSets' / 'train.txt'
            self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)
        
        self.class_names = np.array(self.class_names)
    
    def clear_database(self):
        import shutil
        if dist.is_initialized():
            if dist.get_rank() == 0:
                if self.database_save_path.exists():
                    shutil.rmtree(str(self.database_save_path))
                self.database_save_path.mkdir(parents=False, exist_ok=False)
                if self.db_info_save_path.exists():
                    self.db_info_save_path.unlink()
            dist.barrier()  
        else:
            if self.database_save_path.exists():
                shutil.rmtree(str(self.database_save_path))
            self.database_save_path.mkdir(parents=False, exist_ok=False)
            if self.db_info_save_path.exists():
                self.db_info_save_path.unlink()
                
    def generate_single_db(self, gt_labels, batch_dict, db_infos):
        batch_size = batch_dict['batch_size']
        gt_labels_size = len(gt_labels)
        for batch_idx in range(batch_size):
            if batch_idx >= gt_labels_size:
                break
            gt_boxes = gt_labels[batch_idx]['gt_boxes'].cpu().detach().numpy()
            num_obj = gt_boxes.shape[0]

            sample_idx = batch_dict['frame_id'][batch_idx]
            points_indices = batch_dict['points'][:, 0] == batch_idx
            points = batch_dict['points'][points_indices][:, 1:].cpu().detach().numpy()
            gt_names = np.array(self.class_names[gt_labels[batch_idx]['gt_labels'].cpu().detach().numpy() - 1])
            iou_scores = gt_labels[batch_idx]['gt_scores']
            iou_scores = np.where(iou_scores < 0.1, 0.1, iou_scores)

            bbox = np.zeros([num_obj, 4])
            difficulty = np.zeros_like(gt_names, dtype=np.int32)

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = self.database_save_path / filename
                if filepath.exists():
                    continue
                
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))
                db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i], 'bbox': bbox[i], 'score': -1.0,
                        'pred_score': iou_scores[i], 'cls_score': -1.0}
                if gt_names[i] in db_infos:
                    db_infos[gt_names[i]].append(db_info)
                else:
                    db_infos[gt_names[i]] = [db_info]
        return db_infos


    def save_db_infos(self, db_infos):
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(db_infos, f)