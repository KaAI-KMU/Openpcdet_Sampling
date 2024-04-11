import numpy as np

from .base_collector import BaseCollector


class GTDataCollector(BaseCollector):
    def __init__(self, sampler_cfg, model, dataloader):
        super().__init__(sampler_cfg, model, dataloader, db_type='gt')

    def data_post_process(self, boxes, names, labels, batch_idx):
        iou_scores = labels[batch_idx]['scores']
        iou_scores = np.where(iou_scores < 0.1, 0.1, iou_scores)
        return boxes, names, iou_scores
