import torch
import numpy as np
import pickle
import copy
from tqdm import tqdm

import torch.distributed as dist
from pathlib import Path
import random

from ..models import load_data_to_gpu
from ..ops.iou3d_nms import iou3d_nms_utils
from ..utils.gt_data_collector import GTDataCollector
from ..utils.fp_data_collector import FPDataCollector
from ..utils import common_utils

class DataCollector:
    def __init__(self, sampler_cfg, model, dataloader, use_dist=False):
        self.fp_data_collector = FPDataCollector(sampler_cfg, model, dataloader)
        self.gt_data_collector = GTDataCollector(sampler_cfg, model, dataloader)
        self.model = model
        self.dataloader = dataloader
        self.use_dist = use_dist

        DISABLE_AUG_LIST = sampler_cfg.DISABLE_AUG_LIST
        augmentor_config = self.dataloader.dataset.dataset_cfg.DATA_AUGMENTOR
        augmentor_config.DISABLE_AUG_LIST = DISABLE_AUG_LIST
        self.dataloader.dataset.data_augmentor.disable_augmentation(augmentor_config)

        score_key = sampler_cfg.get('score_key', None)
        if score_key is None:
            self.score_key = 'pred_scores'
        elif score_key == 'iou':
            self.score_key = 'pred_scores'
        elif score_key == 'cls':
            self.score_key = 'pred_cls_scores'
        else:
            raise NotImplementedError
         
    def sample_labels(self):
        self.clear_database()
        if self.use_dist:
            dist.barrier()
            rank, world_size = common_utils.get_dist_info()
        else:
            rank = 0

        fp_db_infos = {}
        gt_db_infos = {}
        self.model.eval()
        if rank == 0:
            progress_bar = tqdm(total=len(self.dataloader), desc='labels_generating', leave=True)
        for i, batch_dict in enumerate(self.dataloader):
            batch_size = batch_dict['batch_size']
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, _ = self.model(batch_dict)
                
            fp_pred_dict_list = []
            gt_pred_dict_list = []
            for batch_idx in range(batch_size):
                pred_scores = pred_dicts[batch_idx]['pred_scores']
                gt_boxes = batch_dict['gt_boxes'][batch_idx][batch_dict['gt_boxes'][batch_idx][:,-1].long() != 0]
                pred_boxes = pred_dicts[batch_idx]['pred_boxes']
                pred_classes = pred_dicts[batch_idx]['pred_labels']
                if pred_boxes.shape[0] == 0:
                    fp_pred_dict_list.append(None)
                    gt_pred_dict_list.append(None)
                    continue
                
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes[:,:-1])
                class_matrix = torch.eq(pred_classes.unsqueeze(1), gt_boxes[:, -1]).float()
                iou3d = iou3d * class_matrix

                max_ious_gt = iou3d.max(axis=0)[0].cpu().numpy()
                max_ious = iou3d.max(axis=1)[0].cpu().numpy()
                selected_fp = max_ious < 0.1
                
                fp_pred_dict_list.append({
                    'boxes': pred_dicts[batch_idx]['pred_boxes'][selected_fp],
                    'labels': pred_dicts[batch_idx]['pred_labels'][selected_fp],
                    'scores': pred_dicts[batch_idx][self.score_key][selected_fp]
                })
                gt_pred_dict_list.append({
                    'boxes': gt_boxes[:, :7],
                    'labels': gt_boxes[:, -1].to(torch.int32),
                    'scores' : max_ious_gt
                })

            if rank == 0:
                progress_bar.update()

            fp_db_infos = self.fp_data_collector.generate_single_db(fp_pred_dict_list, batch_dict, fp_db_infos)
            gt_db_infos = self.gt_data_collector.generate_single_db(gt_pred_dict_list, batch_dict, gt_db_infos)

        if self.use_dist:
            rank, world_size = common_utils.get_dist_info()
            tmpdir = str(self.root_path / 'tmp')
            fp_db_infos = common_utils.merge_dict_dist(fp_db_infos, tmpdir + '_fp')
            gt_db_infos = common_utils.merge_dict_dist(gt_db_infos, tmpdir + '_gt')
        
        if rank == 0:
            progress_bar.close()
            self.fp_data_collector.save_db_infos(fp_db_infos)
            self.gt_data_collector.save_db_infos(gt_db_infos)

    def clear_database(self):
        self.fp_data_collector.clear_database(self.use_dist)
        self.gt_data_collector.clear_database(self.use_dist)
