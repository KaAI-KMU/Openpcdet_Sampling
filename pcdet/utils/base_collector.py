import shutil
import torch
import torch.distributed as dist
import numpy as np
import pickle

from pathlib import Path

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


class BaseCollector:
    def __init__(self, sampler_cfg, model, dataloader, db_type):
        self.sampler_cfg = sampler_cfg
        self.dataset_type = dataloader.dataset.dataset_cfg.DATASET
        
        msg = 'Not Implemented for %s' % self.dataset_type
        assert self.dataset_type in ['KittiDataset', 'WaymoDataset', 'ONCEDataset'], msg
        
        self.model = model
        self.dataloader = dataloader
        self.db_type = db_type
        self.root_path = dataloader.dataset.root_path
        self.class_names = np.array(dataloader.dataset.class_names)

        self.cnt_data = 0
        self.sub_dir_num = 1
        self.max_data_num = 1000000
        self._set_data_path()

    def _get_path_dict(self):
        database_save_path_parent = self.root_path / Path(self.dataset_type + '_%s_database_runtime' % self.db_type)
        db_info_save_path = self.root_path / Path(self.dataset_type + '_%s_dbinfos_runtime.pkl' % self.db_type)
        path_dict = {
            'database_save_path_parent': database_save_path_parent,
            'db_info_save_path': db_info_save_path
        }
        database_save_path = database_save_path_parent / Path('sub_dir_' + str(self.sub_dir_num))
        path_dict['database_save_path'] = database_save_path
        return path_dict

    def _set_data_path(self):
        path_dict = self._get_path_dict()
        self.database_save_path_parent = path_dict['database_save_path_parent']
        self.database_save_path = path_dict['database_save_path']
        self.db_info_save_path = path_dict['db_info_save_path']
        self.database_save_path.mkdir(parents=True, exist_ok=True)
    
    def generate_single_db(self, labels, batch_dict, db_infos):
        raise NotImplementedError

    def clear_database(self, use_dist=False):
        if use_dist:
            raise NotImplementedError
            # assert dist.is_initialized()
            # rank = dist.get_rank()
            # if rank == 0:
            #     self._clear_database()
            # dist.barrier()
        else:
            self._clear_database()
            rank = 0

        if rank == 0:
            self.cnt_data = 0
            self.sub_dir_num = 1
            self._set_data_path()
        if use_dist:
            dist.barrier()
    
    def _clear_database(self):
        path_dict = self._get_path_dict()
        database_save_path_parent = path_dict['database_save_path_parent']
        if database_save_path_parent.exists():
            shutil.rmtree(database_save_path_parent)
        db_info_save_path = path_dict['db_info_save_path']
        if db_info_save_path.exists():
            db_info_save_path.unlink()
    
    def save_db_infos(self, db_infos):
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(db_infos, f)

    def generate_single_db(self, labels, batch_dict, db_infos):
        batch_size = batch_dict['batch_size']
        for batch_idx in range(batch_size):
            boxes = labels[batch_idx]['boxes'].cpu().detach().numpy()
            num_obj = boxes.shape[0]

            sample_idx = batch_dict['frame_id'][batch_idx]
            points_indices = batch_dict['points'][:, 0] == batch_idx
            points = batch_dict['points'][points_indices][:, 1:].cpu().detach().numpy()
            box_names = np.array(self.class_names[labels[batch_idx]['labels'].cpu().detach().numpy() - 1])

            boxes, box_names, box_scores = self.data_post_process(boxes, box_names, labels, batch_idx)
            
            num_obj = len(box_names) 
            bbox_2d = np.zeros([num_obj, 4])
            difficulty = np.zeros_like(box_names, dtype=np.int32)

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, box_names[i], i)
                self.cnt_data += 1
                if self.cnt_data >= self.max_data_num:
                    self.sub_dir_num += 1
                    self.database_save_path = self.database_save_path_parent / ('sub_dir_' + str(self.sub_dir_num))
                    self.database_save_path.mkdir(parent=True, exist_ok=True)
                    self.cnt_data = 0 
                        
                filepath = self.database_save_path / filename
                if filepath.exists():
                    continue
                
                points_in_bbox = points[point_indices[i] > 0]
                points_in_bbox[:, :3] -= boxes[i, :3]
                with open(filepath, 'w') as f:
                    points_in_bbox.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))
                db_info = {'name': box_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': boxes[i], 'num_points_in_gt': points_in_bbox.shape[0],
                        'difficulty': difficulty[i], 'bbox': bbox_2d[i], 'score': -1.0,
                        'pred_score': box_scores[i], 'cls_score': -1.0}
                if box_names[i] in db_infos.keys():
                    db_infos[box_names[i]].extend(db_info)
                else:
                    db_infos[box_names[i]] = [db_info]
        return db_infos

    def data_post_process(self, boxes, names, labels, batch_idx):
        raise NotImplementedError