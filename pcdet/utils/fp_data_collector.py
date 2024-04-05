from .base_collector import BaseCollector


class FPDataCollector(BaseCollector):
    def __init__(self, sampler_cfg, model, dataloader):
        super().__init__(sampler_cfg, model, dataloader, db_type='fp')
        self.remove_threshold = sampler_cfg.REMOVE_THRESHOLD
    
    def data_post_process(self, boxes, names, labels, batch_idx):
        scores = labels[batch_idx]['scores'].cpu().detach().numpy()
        
        valid_indices = scores > self.remove_threshold
        boxes = boxes[valid_indices]
        names = names[valid_indices]
        scores = scores[valid_indices]

        return boxes, names, scores
