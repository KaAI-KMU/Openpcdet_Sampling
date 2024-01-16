import torch
import numpy as np
import pickle
import copy

from pathlib import Path


import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from ..models import load_data_to_gpu
from tools.eval_utils import statistics_utils

class ThresholdGenerator:
    def __init__(self, model, dataloader, mode='GMM'):
        self.mode = mode
        self.root_path = dataloader.dataset.root_path
        self.stat_dict = {
        'scores': None,
        'labels': None,
        }
        self.threshold = None
        self.dataloader = dataloader
        self.model = model
        self.class_names = dataloader.dataset.class_names
        
        
        imageset_file = self.root_path / 'ImageSets' / 'train.txt'
        self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)
             
    def update_dict(self, desc='pred_scores_generating'):
        if desc == 'pred_scores_generating':
            scores_list = []
            labels_list = []
            boxes_list = []

            for batch_dict in tqdm(self.dataloader, desc=desc, leave=True):
                load_data_to_gpu(batch_dict)
                with torch.no_grad():
                    pred_dicts, _ = self.model(batch_dict)

                for pred_dict in pred_dicts:
                    scores_list.append(pred_dict['pred_scores'].detach().cpu().numpy())
                    labels_list.append(pred_dict['pred_labels'].detach().cpu().numpy())
                    #boxes_list.append(pred_dict['pred_boxes'].detach().cpu().numpy())

            self.stat_dict['scores'] = np.concatenate(scores_list, axis=0)
            self.stat_dict['labels'] = np.concatenate(labels_list, axis=0)
            #self.stat_dict['boxes'] = np.concatenate(boxes_list, axis=0)

    def generate_threshold(self):
        if self.mode == 'GMM':
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(self.stat_dict['scores'].reshape(-1, 1))
            self.threshold = gmm.means_.min()  
        else:
            raise NotImplementedError
        
    def generate_threshold_each_class_kde(self, bandwidth=0.1):
        thresholds = {}
        kdes = {}
        
        # Car
        car_scores = self.stat_dict['scores'][self.stat_dict['labels'] == 1]
        if len(car_scores) >= 2:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(car_scores.reshape(-1, 1))
            kdes['Car'] = kde

            scores_sample = np.linspace(car_scores.min(), car_scores.max(), 1000).reshape(-1, 1)
            log_density = kde.score_samples(scores_sample)
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(scores_sample)
            thresholds[0] = gmm.means_.min()

        # Pedestrian
        ped_scores = self.stat_dict['scores'][self.stat_dict['labels'] == 2]
        if len(ped_scores) >= 2:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(ped_scores.reshape(-1, 1))
            kdes['Pedestrian'] = kde

            scores_sample = np.linspace(ped_scores.min(), ped_scores.max(), 1000).reshape(-1, 1)
            log_density = kde.score_samples(scores_sample)
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(scores_sample)
            thresholds[1] = gmm.means_.min()

        # Cyclist
        cyc_scores = self.stat_dict['scores'][self.stat_dict['labels'] == 3]
        if len(cyc_scores) >= 2:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(cyc_scores.reshape(-1, 1))
            kdes['Cyclist'] = kde

            scores_sample = np.linspace(cyc_scores.min(), cyc_scores.max(), 1000).reshape(-1, 1)
            log_density = kde.score_samples(scores_sample)
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(scores_sample)
            thresholds[2] = gmm.means_.min()

        self.threshold = thresholds

    def generate_threshold_each_class(self):
        if self.mode == 'GMM':
            gmm = GaussianMixture(n_components=8, random_state=0)
            class_scores = [self.stat_dict['scores'][self.stat_dict['labels'] == i] for i in range(1, 4)]  # 1: Car, 2: Pedestrian, 3: Cyclist
            
            self.threshold = []
            for idx, scores in enumerate(class_scores):
                gmm.fit(scores.reshape(-1, 1))
                means = gmm.means_.ravel()
                covariances = gmm.covariances_.ravel()
                weights = gmm.weights_.ravel()

                sorted_indices = np.argsort(means)

                left_indices = sorted_indices[:4]
                right_indices = sorted_indices[4:]

                left_weighted_mean = np.average(means[left_indices], weights=weights[left_indices])
                right_weighted_mean = np.average(means[right_indices], weights=weights[right_indices])
                left_weighted_variance = np.average(covariances[left_indices], weights=weights[left_indices])
                right_weighted_variance = np.average(covariances[right_indices], weights=weights[right_indices])

                threshold = (left_weighted_mean + right_weighted_mean) / 2
                self.threshold.append(threshold)
        else:
            raise NotImplementedError

    def filter_pred_dicts(self, pred_dicts, thresholds):
        filtered_pred_dicts = []

        threshold_mapper = {cls_name: thresholds[i] for i, cls_name in enumerate(self.class_names)}
        
        for pred_dict in pred_dicts:
            pred_classes = pred_dict['pred_labels']
            pred_scores = pred_dict['pred_scores']
            pred_boxes = pred_dict['pred_boxes']
            
            if pred_boxes.shape[0] == 0:
                continue
            
            selected = np.array([False] * len(pred_classes))

            for cls_name, fp_threshold in threshold_mapper.items():
                cls_index = self.class_names.index(cls_name) + 1
                cls_selected = (pred_classes.cpu().numpy() == cls_index) & (pred_scores.cpu().numpy() < fp_threshold)
                selected = selected | cls_selected
            
            filtered_dict = {key: val[selected] for key, val in pred_dict.items()}
            filtered_pred_dicts.append(filtered_dict)

        return filtered_pred_dicts
    
    def get_threshold(self):
        return self.threshold
