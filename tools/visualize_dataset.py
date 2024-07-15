import argparse
import glob
from pathlib import Path

import open3d as o3d
from visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

CLASS_COLORS = {
    'Regular_vehicle': [1, 0, 0], # Red
    'Pedestrian': [0, 1, 0], # Green
    'Bicycle': [0, 0, 1],    # Blue
    'Bus': [1, 1, 0],      # Yellow
    'UNKNOW': [1, 0, 1],  # Magenta
}

DEFAULT_COLOR = [0.5, 0.5, 0.5]  # Gray

class DemoDataset(DatasetTemplate):
    def __init__(self, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / 'points' / f'*{self.ext}'), recursive=True) if self.root_path.is_dir() else [self.root_path]
        label_file_list = glob.glob(str(root_path / 'labels' / '*.txt'), recursive=True) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        label_file_list.sort()
        
        SAMPLE_RATE = 1
        self.sample_file_list = data_file_list[::SAMPLE_RATE]
        self.label_file_list = label_file_list[::SAMPLE_RATE]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float64).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            o3d_points = o3d.io.read_point_cloud(self.sample_file_list[index])
            points = np.asarray(o3d_points.points)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        print(self.label_file_list[index])
        gt_boxes_lidar, gt_names = self.get_label(self.label_file_list[index])
        input_dict.update({
            'gt_names': gt_names,
            'gt_boxes': gt_boxes_lidar
        })

        return input_dict
    
    def get_label(self, label_path):
        Path(label_path).exists()
        with open(label_path, 'r') as f:
            lines = f.readlines()

        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.split()
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

def visualize_class_distribution(demo_dataset):
    class_counts = {class_name: 0 for class_name in CLASS_COLORS.keys()}
    positions = {class_name: [] for class_name in CLASS_COLORS.keys()}
    
    for data_dict in demo_dataset:
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']
        
        for i, class_name in enumerate(gt_names):
            class_counts[class_name] += 1
            positions[class_name].append(gt_boxes[i][:3])  # x, y, z positions

    # Plot class distribution
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=[CLASS_COLORS[class_name] for class_name in class_counts.keys()])
    plt.yscale('log')  # Apply logscale
    plt.gca().yaxis.grid(True, linestyle='--', linewidth=0.7)  # Add horizontal grid lines
    plt.title(f'Class Distribution for {len(demo_dataset)} Frames')
    plt.xlabel('Class')
    plt.ylabel('Count (log scale)')
    plt.tight_layout()

    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # va='bottom' aligns text at the top of the bar
    
    plt.show()

    # Scatter plot of positions
    plt.figure(figsize=(20, 20))
    for class_name, pos_list in positions.items():
        if len(pos_list) == 0:
            continue
        pos_array = np.array(pos_list)
        plt.scatter(pos_array[:, 0], pos_array[:, 1], c=[CLASS_COLORS[class_name]], label=class_name, s=1)  # Reduced point size
    plt.title('Class Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.gca().set_aspect('equal', adjustable='box')  # Fix aspect ratio
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_frame(demo_dataset, logger):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 1.0  # Point size
    opt.line_width = 2.0  # Bounding box line width
    opt.light_on = True

    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        
        points = data_dict['points'][:, :3]
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']

        vis.clear_geometries()

        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(points)
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        vis.add_geometry(pts)

        for i in range(gt_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
            color = CLASS_COLORS.get(gt_names[i], DEFAULT_COLOR)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.1)

        vis.poll_events()
        vis.update_renderer()

    logger.info('Visualization done.')
    vis.destroy_window()

def parse_config():
    parser = argparse.ArgumentParser(description='Dataset Visualization')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/rscube_dataset_v4.yaml',
                        help='specify the config for visualization')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    parser.add_argument('--vis_frame', action='store_true', help='Enable frame-wise visualization')
    parser.add_argument('--vis_dist', action='store_true', default=True, help='Enable class distribution visualization')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()  
    logger = common_utils.create_logger()
    logger.info('-----------------Dataset Visualization-------------------------')
    demo_dataset = DemoDataset(
        class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(cfg.DATA_PATH), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    if args.vis_frame:
        visualize_frame(demo_dataset, logger)
    
    if args.vis_dist:
        visualize_class_distribution(demo_dataset)

if __name__ == '__main__':
    main()
