import torch
import numpy as np
import pickle
import copy
from tqdm import tqdm

from pathlib import Path
import random


from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..models import load_data_to_gpu
from ..datasets import build_dataloader
from ..ops.iou3d_nms import iou3d_nms_utils
from ..utils.gt_data_collector import GTDataCollector
from ..utils.fp_data_collector import FPDataCollector

class DataCollector:
    def __init__(self, sampler_cfg, model, dataloader):
        self.fp_data_collector = FPDataCollector(sampler_cfg, model, dataloader)
        self.gt_data_collector = GTDataCollector(sampler_cfg, model, dataloader)
        self.model = model
        self.dataloader = dataloader
        self.sampler_cfg = sampler_cfg
        
        self.root_path = dataloader.dataset.root_path
        self.class_names = np.array(dataloader.dataset.class_names)
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
        
    def sample_labels(self):
        self.clear_database()
        self.model.eval()
        all_db_infos_fp = {}
        all_db_infos_gt = {}
        fp_pred_dict = {}  
        gt_pred_dict = {}  
        sample_rate = 0.01

        for batch_dict in tqdm(self.dataloader, desc='labels_generating', leave=True):
            if self.sampler_cfg['Dataset'] == 'Waymo':
                if random.random() > sample_rate:
                    continue
            batch_size = batch_dict['batch_size']
            load_data_to_gpu(batch_dict)
            
            with torch.no_grad():
                pred_dicts, _ = self.model(batch_dict)

            for batch_idx in range(batch_size):
                pred_scores = pred_dicts[batch_idx]['pred_scores']
                gt_boxes = batch_dict['gt_boxes'][batch_idx]
                pred_boxes = pred_dicts[batch_idx]['pred_boxes']
                pred_classes = pred_dicts[batch_idx]['pred_labels']
                if pred_boxes.shape[0] == 0:
                    continue
                
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes[:,:-1])
                class_matrix = torch.eq(pred_classes.unsqueeze(1), gt_boxes[:, -1]).float()
                iou3d = iou3d * class_matrix

                max_ious_gt = iou3d.max(axis=0)[0].cpu().numpy()
                max_ious = iou3d.max(axis=1)[0].cpu().numpy()
                selected_fp = max_ious == 0.0
                
                fp_pred_dict[batch_idx] = {key: val[selected_fp] for key, val in pred_dicts[batch_idx].items()}
                
                gt_pred_dict[batch_idx] = {
                    'gt_boxes': gt_boxes[:, :7],
                    'gt_labels': gt_boxes[:, -1].to(torch.int64),
                    'gt_scores' : max_ious_gt
                }

            fp_label_dict = self.fp_data_collector.generate_single_db(fp_pred_dict, batch_dict, all_db_infos_fp)
            gt_label_dict = self.gt_data_collector.generate_single_db(gt_pred_dict, batch_dict, all_db_infos_gt)

        self.fp_data_collector.save_db_infos(fp_label_dict)
        self.gt_data_collector.save_db_infos(gt_label_dict)

    def clear_database(self):
        self.fp_data_collector.clear_database()
        self.gt_data_collector.clear_database()