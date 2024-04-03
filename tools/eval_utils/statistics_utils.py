import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

from pathlib import Path

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


class Statistics:
    def __init__(self,
                 root_path,
                 iou_thresh=[0.5,0.3,0.3],
                 score_keys=['scores'],
                 class_names=['Car', 'Pedestrian', 'Cyclist'],
                 stat_dict_path=None,
                 extra_tag=None):
        self.stat_dict = {}
        self.iou_thresh = iou_thresh
        self.set_score_keys(score_keys)
        self.class_names = class_names
        self.extra_tag = extra_tag

        self.init = True
        if stat_dict_path is not None:
            assert os.path.exists(stat_dict_path), f"Invalid path: {stat_dict_path}"
            stat_dict = pickle.load(open(stat_dict_path, 'rb'))
            self.load_stat_dict(stat_dict)
        self.tpfp_counts = None

        stat_path = Path(root_path) / 'statistics'
        self.save_path = (stat_path / extra_tag) if extra_tag is not None else stat_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.csv_file_path = self.save_path / 'csv'
        self.csv_file_path.mkdir(parents=True, exist_ok=True)
        self.png_file_path = self.save_path / 'png'
        self.png_file_path.mkdir(parents=True, exist_ok=True)
        
        # score sections for visualization
        self.sections = [[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
        self.visualizer = StatVisualizer(self.sections, class_names, self.png_file_path)
    
    def set_score_keys(self, score_keys):
        self.score_keys = score_keys
    
    def save_stat_dict(self, key_name=None):
        k = 'stat_dict' if key_name is None else f'stat_dict_{key_name}'
        stat_dict_path = generate_file_path(self.save_path, k, 'pkl')
        with open(stat_dict_path, 'wb') as f:
            pickle.dump(self.stat_dict, f)
    
    def load_stat_dict(self, stat_dict):
        self.stat_dict = stat_dict
        self.init = False
    
    def export_csv(self, key_name=None):
        if self.tpfp_counts is None:
            self.calc_stats_for_all_keys()

        num_classes = len(self.class_names)
        x_labels = sections_to_str(self.sections)

        for key in self.score_keys:
            tp_counts, fp_counts = self.tpfp_counts[key]['tp'], self.tpfp_counts[key]['fp']

            for i in range(num_classes):
                data = {'tp': tp_counts[i], 'fp': fp_counts[i]}
                df = pd.DataFrame(data, index=x_labels)
                df = df.T

                k = f'{key}_{key_name}_{self.class_names[i]}' if key_name is not None else f'{key}_{self.class_names[i]}'
                path = generate_file_path(self.csv_file_path, k, 'csv')
                df.to_csv(path)
            
    def save_stats_visualization(self, key_name=None, show_image=False):
        if self.tpfp_counts is None:
            self.calc_stats_for_all_keys()

        for key in self.score_keys:
            tp_counts, fp_counts = self.tpfp_counts[key]['tp'], self.tpfp_counts[key]['fp']
            k = f'{key}_{key_name}' if key_name is not None else key
            self.visualizer.save_histogram(tp_counts, fp_counts, show_image=show_image, key_name=k)

        if 'scores' in self.score_keys and 'cls_scores' in self.score_keys:
            self.visualizer.save_scatter(self.stat_dict, show_image=show_image)

    def update_stat_dict(self, batch_dict, pred_dicts):
        batch_size = batch_dict['batch_size']
        for i in range(batch_size):
            pred_boxes = pred_dicts[i]['pred_boxes']
            pred_labels = pred_dicts[i]['pred_labels']
            gt_boxes = batch_dict['gt_boxes'][i]

            # remove dummy gt boxes
            gt_boxes = gt_boxes[gt_boxes[:, -1] > 0]
            gt_num = gt_boxes.shape[0]

            # calculate iou threshold for each gt and pred
            pred_num = pred_boxes.shape[0]
            iou_thresh_per_gt = np.zeros(gt_num, dtype=np.float32)
            iou_thresh_per_pred = np.zeros(pred_num, dtype=np.float32)
            for j in range(gt_num):
                iou_thresh_per_gt[j] = self.iou_thresh[int(gt_boxes[j, -1])-1]
            for j in range(pred_num):
                iou_thresh_per_pred[j] = self.iou_thresh[int(pred_labels[j])-1]

            # calculate iou3d and tp, fp, fn by iou3d and class
            if pred_num == 0:
                tp = np.array([], dtype=np.bool_)
                fn = np.ones(gt_boxes.shape[0], dtype=np.bool_)
            else:
                iou3d = boxes_iou3d_gpu(pred_boxes, gt_boxes[:,:-1]).cpu().numpy()
                class_matrix = np.equal(pred_labels.detach().cpu().numpy().reshape(-1,1), gt_boxes[:,-1].cpu().numpy().reshape(1,-1))
                iou3d = iou3d * class_matrix
                tp = iou3d.max(axis=1) >= iou_thresh_per_pred
                fn = iou3d.max(axis=0) < iou_thresh_per_gt

            # calculate num_points_in_pred and num_points_in_gt
            single_scene_points = batch_dict['points'][batch_dict['points'][:,0] == i][:, 1:4]
            num_points_in_pred = np.zeros(pred_num, dtype=np.int32)
            num_points_in_gt = np.zeros(gt_num, dtype=np.int32)
            num_points_in_pred = points_in_boxes_cpu(single_scene_points.cpu(), pred_dicts[i]['pred_boxes'].cpu()).sum(axis=1)
            num_points_in_gt = points_in_boxes_cpu(single_scene_points.cpu(), gt_boxes[:,:-1].cpu()).sum(axis=1)

            if self.init:
                self.init = False
                self.stat_dict['tp'] = tp
                self.stat_dict['fp'] = ~tp
                self.stat_dict['fn'] = fn
                self.stat_dict['num_points_in_pred'] = num_points_in_pred
                self.stat_dict['num_points_in_gt'] = num_points_in_gt
                for key in self.score_keys:
                    self.stat_dict[key] = pred_dicts[i]['pred_'+key].cpu().numpy()
                self.stat_dict['pred_labels'] = pred_labels.cpu().numpy()
                self.stat_dict['gt_labels'] = gt_boxes[:,-1].cpu().numpy()
            else:
                self.stat_dict['tp'] = np.concatenate((self.stat_dict['tp'], tp))
                self.stat_dict['fp'] = np.concatenate((self.stat_dict['fp'], ~tp))
                self.stat_dict['fn'] = np.concatenate((self.stat_dict['fn'], fn))
                self.stat_dict['num_points_in_pred'] = np.concatenate((self.stat_dict['num_points_in_pred'], num_points_in_pred))
                self.stat_dict['num_points_in_gt'] = np.concatenate((self.stat_dict['num_points_in_gt'], num_points_in_gt))
                for key in self.score_keys:
                    self.stat_dict[key] = np.concatenate((self.stat_dict[key], pred_dicts[i]['pred_'+key].cpu().numpy()))
                self.stat_dict['pred_labels'] = np.concatenate((self.stat_dict['pred_labels'], pred_labels.cpu().numpy()))
                self.stat_dict['gt_labels'] = np.concatenate((self.stat_dict['gt_labels'], gt_boxes[:,-1].cpu().numpy()))

        return self.stat_dict
    
    def calc_stats_for_all_keys(self):
        self.tpfp_counts = {key: self.calculate_pred_stats(key) for key in self.score_keys}

    def calculate_pred_stats(self, key = 'scores'):
        msg = 'key must be one of scores or cls_scores. but got %s' % key
        assert key in ['scores', 'cls_scores'], msg

        tp, fp = self.stat_dict['tp'], self.stat_dict['fp']
        stat_pred = self.stat_dict[key]

        num_classes = len(self.class_names)
        class_mask_pred = np.equal(self.stat_dict['pred_labels'].reshape(1, -1), np.arange(num_classes).reshape(-1, 1)+1)
        tp_mask = tp.reshape(1,-1) & class_mask_pred
        fp_mask = fp.reshape(1,-1) & class_mask_pred

        num_sections = len(self.sections)
        tp_counts = np.zeros((num_classes, num_sections), dtype=np.int32)
        fp_counts = np.zeros((num_classes, num_sections), dtype=np.int32)

        for i in range(num_classes):
            for j in range(num_sections):
                mask_pred = (stat_pred >= self.sections[j][0]) & (stat_pred < self.sections[j][1])
                mask_pred = mask_pred & class_mask_pred[i]
                tp_counts[i, j] = tp_mask[i, mask_pred].sum()
                fp_counts[i, j] = fp_mask[i, mask_pred].sum()

        return {'tp': tp_counts, 'fp': fp_counts}
    

class StatVisualizer:
    def __init__(self,
                 sections,
                 class_names=['Car', 'Pedestrian', 'Cyclist'],
                 save_path=None,):
        self.sections = sections
        self.sections_str = sections_to_str(sections)
        self.class_names = class_names
        self.save_path = str(save_path) if isinstance(save_path, Path) else save_path

    def save_histogram(self, tp_counts, fp_counts, key_name=None, show_image=False):
        num_classes = len(self.class_names)
        num_sections = len(self.sections)

        fig, axs = plt.subplots(1, num_classes, figsize=(20, 20))
        for i in range(num_classes):
            axs[i].bar(np.arange(num_sections), tp_counts[i], width=0.3, label='tp')
            axs[i].bar(np.arange(num_sections)+0.3, fp_counts[i], width=0.3, label='fp')
            axs[i].set_xticks(np.arange(num_sections)+0.3)
            axs[i].set_yscale('log') 
            axs[i].set_title(self.class_names[i])

            for j in range(num_sections):
                axs[i].text(j, tp_counts[i,j], str(tp_counts[i,j]), ha='center', va='top')
                axs[i].text(j+0.3, fp_counts[i,j], str(fp_counts[i,j]), ha='center', va='top')
            
            axs[i].set_xticks(np.arange(num_sections)+0.3)
            axs[i].set_xticklabels(self.sections_str)
            axs[i].set_xlabel('scores')
            axs[i].set_ylabel('# preds')
            axs[i].legend()

        if show_image:
            plt.show()

        key_name = 'tp_fp_counts' if key_name is None else f'tp_fp_counts_{key_name}'
        path = generate_file_path(self.save_path, key_name, 'png')
        fig.savefig(path)

    def save_scatter(self, stat_dict, extra_key=None, show_image=False):
        assert 'scores' in stat_dict.keys() and 'cls_scores' in stat_dict.keys(), 'scores and cls_scores must be in stat_dict'

        tp, fp = stat_dict['tp'], stat_dict['fp']
        iou_scores = stat_dict['scores'] # IOU 점수
        cls_scores = stat_dict['cls_scores'] # 클래스 점수

        num_classes = len(self.class_names)
        class_mask_pred = np.equal(stat_dict['pred_labels'].reshape(1, -1), np.arange(num_classes).reshape(-1, 1) + 1)

        fig, axs = plt.subplots(1, num_classes, figsize=(20, 5))

        for i in range(num_classes):
            # 클래스별로 TP와 FP 필터링
            tp_mask = tp & class_mask_pred[i]
            fp_mask = fp & class_mask_pred[i]

            # TP와 FP의 IOU 점수와 클래스 점수 추출
            tp_iou_scores = iou_scores[tp_mask]
            tp_cls_scores = cls_scores[tp_mask]
            fp_iou_scores = iou_scores[fp_mask]
            fp_cls_scores = cls_scores[fp_mask]

            # TP는 빨간색 점으로, FP는 파란색 점으로 표시
            axs[i].scatter(tp_iou_scores, tp_cls_scores, color='red', label='TP', s=1)
            axs[i].scatter(fp_iou_scores, fp_cls_scores, color='blue', label='FP', s=1)

            axs[i].set_title(self.class_names[i])
            axs[i].set_xlabel('IOU Score')
            axs[i].set_ylabel('Class Score')
            axs[i].legend()

        if show_image:
            plt.show()

        key_name = 'scores_scatter' if extra_key is None else f'scores_scatter_{extra_key}'
        path = generate_file_path(self.save_path, key_name, 'png')
        fig.savefig(path)


def generate_file_path(path, key, ext):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f'{key}.{ext}'

    cnt = 0
    while file_path.exists():
        cnt += 1
        file_path = path / f'{key}_{cnt}.{ext}'

    return file_path


def sections_to_str(sections):
    sections_str = [str(section[0])+'-'+str(section[1]) for section in sections]
    return sections_str