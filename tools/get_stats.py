import argparse
import os

from eval_utils.statistics_utils import Statistics


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--iou_thresholds', nargs='+', default=[0.5, 0.3, 0.3], help='iou thresholds for tp, fp for each class')
    parser.add_argument('--stat_dict_path', type=str, default=None, help='path to stat dict')
    parser.add_argument('--stat_keys', nargs='+', default=['scores', 'cls_scores'], help='stat key for tp, fp')
    parser.add_argument('--class_names', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'], help='class names for tp, fp for each class')
    parser.add_argument('--root_path', type=str, default='..', help='path to save csv file and png file')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    assert args.stat_dict_path is not None, "Please specify a stat dict path"
    assert os.path.exists(args.stat_dict_path), f"Invalid path: {args.stat_dict_path}"

    stats = Statistics(
        iou_thresh=args.iou_thresholds,
        score_keys=args.stat_keys,
        class_names=args.class_names,
        stat_dict_path=args.stat_dict_path,
        root_path=args.root_path,
        extra_tag=args.extra_tag
    )
    stats.save_stats_visualization(key_name='debug', show_image=True)
    stats.export_csv(key_name='debug')


if __name__ == '__main__':
    main()