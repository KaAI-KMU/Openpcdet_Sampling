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


class FPDataCollector:
    def __init__(self, sampler_cfg, model, dataloader):
        self.sampler_cfg = sampler_cfg
        self.interval = sampler_cfg['INTERVAL']
        if sampler_cfg['REMOVE_THRESHOLD'] is not None:
            self.remove_threshold = sampler_cfg['REMOVE_THRESHOLD']
        else:
            self.remove_threshold = 0.0
        self.model = model
        self.dataloader = dataloader
        self.root_path = dataloader.dataset.root_path
        self.class_names = dataloader.dataset.class_names

        if self.sampler_cfg['Dataset'] == 'KITTI':
            self.database_save_path = Path(self.root_path) / 'gt_database_runtime'
            self.db_info_save_path = Path(self.root_path) / 'kitti_dbinfos_runtime.pkl'
            imageset_file = self.root_path / 'ImageSets' / 'train.txt'
            self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)
        elif self.sampler_cfg['Dataset'] == 'Waymo':
            self.database_save_path = Path(self.root_path) / 'gt_database_runtime'
            self.db_info_save_path = Path(self.root_path) / 'waymo_processed_data_v0_5_0_waymo_dbinfos_runtime_sampled_1.pkl'  
            imageset_file = self.root_path / 'ImageSets' / 'train.txt'
            with open(imageset_file, 'r') as file:
                self.labeled_mask = [line.strip() for line in file.readlines()]
    
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

    def generate_single_db(self, fp_labels, batch_dict, db_infos):
        batch_size = batch_dict['batch_size']
        fp_labels_size = len(fp_labels)
        for batch_idx in range(batch_size):
            if batch_idx >= fp_labels_size:
                break
            fp_boxes = fp_labels[batch_idx]['pred_boxes'].cpu().detach().numpy()
            num_obj = fp_boxes.shape[0]

            sample_idx = batch_dict['frame_id'][batch_idx]
            points_indices = batch_dict['points'][:, 0] == batch_idx
            points = batch_dict['points'][points_indices][:, 1:].cpu().detach().numpy()
            fp_names = np.array(self.class_names[fp_labels[batch_idx]['pred_labels'].cpu().detach().numpy() - 1])
            iou_scores = fp_labels[batch_idx]['pred_scores'].cpu().detach().numpy()
            #cls_scores = fp_labels[batch_idx]['pred_cls_scores'].cpu().detach().numpy()
            
            valid_indices = iou_scores > self.remove_threshold
            fp_boxes = fp_boxes[valid_indices]
            fp_names = fp_names[valid_indices]
            iou_scores = iou_scores[valid_indices]
            
            num_obj = len(fp_names) 
            bbox = np.zeros([num_obj, 4])
            difficulty = np.zeros_like(fp_names, dtype=np.int32)

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(fp_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, fp_names[i], i)
                filepath = self.database_save_path / filename
                if filepath.exists():
                    continue

                fp_points = points[point_indices[i] > 0]
                fp_points[:, :3] -= fp_boxes[i, :3]
                with open(filepath, 'w') as f:
                    fp_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))
                db_info = {'name': fp_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': fp_boxes[i], 'num_points_in_gt': fp_points.shape[0],
                        'difficulty': difficulty[i], 'bbox': bbox[i], 'score': -1.0,
                        'iou_score': iou_scores[i], 'cls_score': -1.0}
                if fp_names[i] in db_infos:
                    db_infos[fp_names[i]].append(db_info)
                else:
                    db_infos[fp_names[i]] = [db_info]
        return db_infos


    def save_db_infos(self, db_infos):
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(db_infos, f)