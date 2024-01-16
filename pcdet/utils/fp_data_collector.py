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



class FPDataCollector:
    def __init__(self, sampler_cfg, model, dataloader):
        self.sampler_cfg = sampler_cfg
        self.interval = sampler_cfg['INTERVAL']
        self.model = model
        self.dataloader = dataloader
        self.root_path = dataloader.dataset.root_path
        self.class_names = dataloader.dataset.class_names

        # split_name을 사용하지 않고, 고정된 경로 사용
        self.database_save_path = Path(self.root_path) / 'gt_database_runtime'
        self.db_info_save_path = Path(self.root_path) / 'kitti_dbinfos_runtime.pkl'

        imageset_file = self.root_path / 'ImageSets' / 'train.txt'
        self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)
    
        self.class_names = np.array(self.class_names)
    
    def clear_database(self):
        import shutil
        if self.database_save_path.exists():
            shutil.rmtree(str(self.database_save_path))
        self.database_save_path.mkdir(parents=False, exist_ok=False)
        if self.db_info_save_path.exists():
            self.db_info_save_path.unlink()

    def sample_fp_labels(self):
        self.clear_database()
        self.model.eval()
        all_db_infos = {}

        for batch_dict in tqdm(self.dataloader, desc='fp_labels_generating', leave=True):
            batch_size = batch_dict['batch_size']
            load_data_to_gpu(batch_dict)
            labeled_indices = [int(batch_dict['frame_id'][batch_idx]) in self.labeled_mask for batch_idx in range(batch_size)]

            with torch.no_grad():
                pred_dicts, _ = self.model(batch_dict)

            for batch_idx in range(batch_size):
                pred_scores = pred_dicts[batch_idx]['pred_scores']
                gt_boxes = batch_dict['gt_boxes'][batch_idx][:, :7]
                pred_boxes = pred_dicts[batch_idx]['pred_boxes']
                pred_classes = pred_dicts[batch_idx]['pred_labels']
                if pred_boxes.shape[0] == 0:
                    continue

                selected = np.array([False] * len(pred_classes))

                if gt_boxes.shape[0] > 0:
                    ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes).cpu().numpy()
                    max_ious = ious.max(axis=1)
                    selected = max_ious < 0.1

                # 클래스별 임계값 적용
                for cls_id, cls_name in enumerate(self.class_names):
                    cls_mask = pred_classes.cpu().numpy() == (cls_id + 1)
                    
                    if cls_name == 'Car':
                        threshold = 0.5
                    elif cls_name == 'Cyclist':
                        threshold = 0.6
                    elif cls_name == 'Pedestrian':
                        threshold = 0.6

                    cls_selected = (pred_scores.cpu().numpy() < threshold) & cls_mask
                    selected = selected | cls_selected

                pred_dicts[batch_idx] = {key: val[selected] for key, val in pred_dicts[batch_idx].items()}

            fp_label_dict = self.generate_single_db(pred_dicts, batch_dict, labeled_indices, all_db_infos)

        self.save_db_infos(fp_label_dict)


    def generate_single_db(self, fp_labels, batch_dict, labeled_mask, db_infos):
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