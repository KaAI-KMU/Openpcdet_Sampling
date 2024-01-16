import pickle
import time

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

from visual_utils.open3d_vis_utils import draw_batch

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def update_stat_dict(stat_dict, batch_dict, pred_dicts, iou_thresh=[0.5,0.3,0.3], init=False):
    batch_size = batch_dict['batch_size']
    for i in range(batch_size):
        pred_boxes = pred_dicts[i]['pred_boxes']
        pred_labels = pred_dicts[i]['pred_labels']
        gt_boxes = batch_dict['gt_boxes'][i]

        if pred_boxes.shape[0] == 0:
            continue

        # remove dummy gt boxes
        gt_boxes = gt_boxes[gt_boxes[:, -1] > 0]
        gt_num = gt_boxes.shape[0]

        # calculate iou threshold for each gt and pred
        pred_num = pred_boxes.shape[0]
        iou_thresh_per_gt = np.zeros(gt_num, dtype=np.float32)
        iou_thresh_per_pred = np.zeros(pred_num, dtype=np.float32)
        for j in range(gt_num):
            iou_thresh_per_gt[j] = iou_thresh[int(gt_boxes[j, -1])-1]
        for j in range(pred_num):
            iou_thresh_per_pred[j] = iou_thresh[int(pred_labels[j])-1]

        # calculate iou3d and tp, fp, fn by iou3d and class
        iou3d = boxes_iou3d_gpu(pred_boxes, gt_boxes[:,:-1]).cpu().numpy()
        class_matrix = np.equal(pred_labels.detach().cpu().numpy().reshape(-1,1), gt_boxes[:,-1].cpu().numpy().reshape(1,-1))
        iou3d = iou3d * class_matrix
        tp = iou3d.max(axis=1) >= iou_thresh_per_pred
        fn = iou3d.max(axis=0) < iou_thresh_per_gt

        # calculate num_points_in_pred and num_points_in_gt
        single_scene_points = batch_dict['points'][batch_dict['points'][:,0] == i][:, 1:-1]
        num_points_in_pred = np.zeros(pred_num, dtype=np.int32)
        num_points_in_gt = np.zeros(gt_num, dtype=np.int32)
        num_points_in_pred = points_in_boxes_cpu(single_scene_points.cpu(), pred_dicts[i]['pred_boxes'].cpu()).sum(axis=1)
        num_points_in_gt = points_in_boxes_cpu(single_scene_points.cpu(), gt_boxes[:,:-1].cpu()).sum(axis=1)

        if init:
            stat_dict['tp'] = tp
            stat_dict['fp'] = ~tp
            stat_dict['fn'] = fn
            stat_dict['num_points_in_pred'] = num_points_in_pred
            stat_dict['num_points_in_gt'] = num_points_in_gt
            stat_dict['scores'] = pred_dicts[i]['pred_scores'].cpu().numpy()
            stat_dict['pred_labels'] = pred_labels.cpu().numpy()
            stat_dict['gt_labels'] = gt_boxes[:,-1].cpu().numpy()
        else:
            stat_dict['tp'] = np.concatenate((stat_dict['tp'], tp))
            stat_dict['fp'] = np.concatenate((stat_dict['fp'], ~tp))
            stat_dict['fn'] = np.concatenate((stat_dict['fn'], fn))
            stat_dict['num_points_in_pred'] = np.concatenate((stat_dict['num_points_in_pred'], num_points_in_pred))
            stat_dict['num_points_in_gt'] = np.concatenate((stat_dict['num_points_in_gt'], num_points_in_gt))
            stat_dict['scores'] = np.concatenate((stat_dict['scores'], pred_dicts[i]['pred_scores'].cpu().numpy()))
            stat_dict['pred_labels'] = np.concatenate((stat_dict['pred_labels'], pred_labels.cpu().numpy()))
            stat_dict['gt_labels'] = np.concatenate((stat_dict['gt_labels'], gt_boxes[:,-1].cpu().numpy()))

    return stat_dict

def visualize_statistics(stat_dict, class_names = ['Car', 'Pedestrian', 'Cyclist']):
    tp, fp, fn = stat_dict['tp'], stat_dict['fp'], stat_dict['fn']
    num_points_in_pred = stat_dict['num_points_in_pred']
    num_points_in_gt = stat_dict['num_points_in_gt']

    num_classes = len(class_names)
    class_mask_pred = np.equal(stat_dict['pred_labels'].reshape(1, -1), np.arange(num_classes).reshape(-1, 1)+1)
    class_mask_gt = np.equal(stat_dict['gt_labels'].reshape(1, -1), np.arange(num_classes).reshape(-1, 1)+1)
    tp = tp.reshape(1,-1) & class_mask_pred
    fp = fp.reshape(1,-1) & class_mask_pred
    fn = fn.reshape(1,-1) & class_mask_gt

    fig, axs = plt.subplots(1, num_classes, figsize=(20, 20))

    for i in range(num_classes):
        # count the number of true positives, false positives, and false negatives for each number of points in the predicted bounding boxes
        sections = [[0, 10], [10, 100], [100, 1000], [1000, 10000]]
        x_labels = ['0-9', '10-99', '100-999', '1000-9999']
        num_sections = len(sections)
        tp_counts = np.zeros((num_classes,num_sections), dtype=np.int32)
        fp_counts = np.zeros((num_classes,num_sections), dtype=np.int32)
        fn_counts = np.zeros((num_classes,num_sections), dtype=np.int32)
        for j in range(num_sections):
            mask_pred = (num_points_in_pred >= sections[j][0]) & (num_points_in_pred < sections[j][1])
            mask_pred = mask_pred & class_mask_pred[i]
            mask_gt = (num_points_in_gt >= sections[j][0]) & (num_points_in_gt < sections[j][1])
            mask_gt = mask_gt & class_mask_gt[i]

            tp_counts[i,j] = tp[i,mask_pred].sum()
            fp_counts[i,j] = fp[i,mask_pred].sum()
            fn_counts[i,j] = fn[i,mask_gt].sum()

        axs[i].bar(np.arange(num_sections), tp_counts[i], width=0.3, label='tp')
        axs[i].bar(np.arange(num_sections)+0.3, fp_counts[i], width=0.3, label='fp')
        axs[i].bar(np.arange(num_sections)+0.6, fn_counts[i], width=0.3, label='fn')
        axs[i].set_xticks(np.arange(num_sections)+0.3)
        axs[i].set_yscale('log') 
        axs[i].set_title(class_names[i])

        # add text amount to each bar on the top of the bar
        for j in range(num_sections):
            axs[i].text(j, tp_counts[i,j], str(tp_counts[i,j]), ha='center', va='top')
            axs[i].text(j+0.3, fp_counts[i,j], str(fp_counts[i,j]), ha='center', va='top')
            axs[i].text(j+0.6, fn_counts[i,j], str(fn_counts[i,j]), ha='center', va='top')
        
        axs[i].set_xticks(np.arange(num_sections)+0.3)
        axs[i].set_xticklabels(x_labels)

        # add text label
        axs[i].set_xlabel('# points in bbox')
        axs[i].set_ylabel('# bboxes')
        axs[i].legend()

    plt.show()

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None, count_tpfpfn=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    stat_dict = {
        'tp': None,
        'fn': None,
        'num_points_in_pred': None,
        'num_points_in_gt': None,
        'scores': None,
    }

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)


        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        if count_tpfpfn:
            stat_dict = update_stat_dict(stat_dict, batch_dict, pred_dicts, init=(i==0))

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    if count_tpfpfn:
        visualize_statistics(stat_dict, class_names)

    return ret_dict


if __name__ == '__main__':
    pass
