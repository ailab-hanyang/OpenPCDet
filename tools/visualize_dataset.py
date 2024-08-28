import argparse

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance
from process_tools import dataset_utils

CLASS_NAME_COLORS = {
    'Regular_vehicle': [1, 0, 0], # Red
    'Pedestrian': [0, 1, 0], # Green
    'Bicycle': [0, 0, 1],    # Blue
    'Bus': [1, 1, 0],      # Yellow
    'UNKNOW': [1, 0, 1],  # Magenta
    'UNKNOWN': [1, 0, 1],  # Magenta
}
CLASS_IDX_COLORS = {
    1: [1, 0, 0], # Red
    2: [0, 1, 0], # Green
    3: [0, 0, 1], # Blue
    4: [1, 1, 0], # Yellow
    5: [1, 0, 1], # Magenta
    6: [1, 0, 1], # Magenta
}
DEFAULT_COLOR = [0.5, 0.5, 0.5]  # Gray

def visualize_class_distribution(demo_dataset):
    class_counts = {class_name: 0 for class_name in CLASS_IDX_COLORS.keys()}
    positions = {class_name: [] for class_name in CLASS_IDX_COLORS.keys()}
    
    for data_dict in demo_dataset:
        gt_boxes = data_dict['gt_boxes'][:, :7]
        gt_classes = data_dict['gt_boxes'][:, 7]
        
        for i, class_name in enumerate(gt_classes):
            class_counts[class_name] += 1
            positions[class_name].append(gt_boxes[i][:3])  # x, y, z positions

    # Plot class distribution
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=[CLASS_IDX_COLORS[class_name] for class_name in class_counts.keys()])
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
        plt.scatter(pos_array[:, 0], pos_array[:, 1], c=[CLASS_IDX_COLORS[class_name]], label=class_name, s=1)  # Reduced point size
    
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
        gt_boxes = data_dict['gt_boxes'][:, :7]
        gt_classes = data_dict['gt_boxes'][:, 7]

        vis.clear_geometries()

        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(points)
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        vis.add_geometry(pts)

        for i in range(gt_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
            color = CLASS_IDX_COLORS.get(gt_classes[i], DEFAULT_COLOR)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

            # Create and add arrow to represent heading
            heading_arrow = _create_heading_arrow(gt_boxes[i], color)
            vis.add_geometry(heading_arrow)

        ctr = vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.1)

        vis.poll_events()
        vis.update_renderer()

    logger.info('Visualization done.')
    vis.destroy_window()

def _create_heading_arrow(box, color):
    """Create an arrow representing the heading of the bounding box."""
    center = box[:3]
    heading_vector = np.array([np.cos(box[6]), np.sin(box[6]), 0])  # Heading in xy-plane

    arrow_start = center
    arrow_end = center + heading_vector * 2  # Scale arrow length

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.1, cone_radius=0.2,
        cylinder_height=2.0, cone_height=0.5,
        resolution=20, cylinder_split=4, cone_split=1
    )
    
    # Move the arrow to the correct position
    translation = np.eye(4)
    translation[:3, 3] = arrow_start
    arrow.transform(translation)

    # Rotate the arrow to point in the correct direction
    arrow_direction = arrow_end - arrow_start
    arrow_direction /= np.linalg.norm(arrow_direction)
    arrow_up = np.array([0, 0, 1])
    axis = np.cross(arrow_up, arrow_direction)
    angle = np.arccos(np.dot(arrow_up, arrow_direction))

    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    arrow.rotate(rotation, center=arrow_start)
    
    arrow.paint_uniform_color(color)
    
    return arrow


def parse_config():
    parser = argparse.ArgumentParser(description='Dataset Visualization')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/rscube_dataset_v3.yaml',
                        help='specify the config for visualization')
    parser.add_argument('--vis_frame', action='store_true', help='Enable frame-wise visualization')
    parser.add_argument('--vis_dist', action='store_true', help='Enable class distribution visualization')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, dataset_cfg = parse_config()  
    logger = common_utils.create_logger()
    logger.info('-----------------  Dataset Visualization -----------------')

    dataset_cfg['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] = list() # Do not augment data for visualization
    dataset = dataset_utils.build_dataset(dataset_cfg)
    logger.info(f'Total number of samples: \t{len(dataset)}')

    if args.vis_frame:
        visualize_frame(dataset, logger)
    
    if args.vis_dist:
        visualize_class_distribution(dataset)

if __name__ == '__main__':
    main()
