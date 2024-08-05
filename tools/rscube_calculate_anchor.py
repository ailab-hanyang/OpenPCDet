import os
import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import json
from rich.progress import track
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pathlib import Path
from pcdet.datasets import CustomDataset

def create_custom_anchors(dataset_cfg, args, logger):
    num_clusters = {"Bus": 2,
                    "Pedestrian": 1,
                    "Regular_vehicle": 1,
                    "Bicycle": 1}

    custom_dataset = CustomDataset(
        dataset_cfg=dataset_cfg, 
        class_names=cfg.CLASS_NAMES,
        training=True, logger=logger,
        # training=False, logger=logger,
    )

    logger.info(f'Total number of samples: \t{len(custom_dataset)}')
    
    boxes = []
    categories = []

    # Convert gt_boxes into list
    for data_dict in track(custom_dataset, description="Processing Frames..."):
        annotations = data_dict['gt_boxes']

        for annotation in annotations:
            boxes.append([annotation[3], annotation[4], annotation[5]])
            categories.append(int(annotation[7]))  # Ensure that category is int
    
    # Convert categories from numbers to class names
    categories = [cfg.CLASS_NAMES[category-1] for category in categories]

    unique_categories = set(categories)
    anchor_configs = []

    boxes = np.array(boxes)
    categories = np.array(categories)
    
    for category in track(unique_categories, description="Processing categories..."):
        category_boxes = boxes[categories == category]
        if len(category_boxes) == 0:
            continue  # Skip categories with no data
        anchors = calculate_anchors(category_boxes, num_clusters[category])
        config = generate_anchor_config(category, anchors)
        anchor_configs.append(config)
    
    output_file = "tools/anchor/rscube_anchor_config.json"
    save_anchor_configs(anchor_configs, output_file)
    
    logger.info(f"Anchor configuration saved to {output_file}")
    
    output_dir = Path("tools") / "anchor" / "visualizations" / 'rscube'
    visualize_anchors_and_distribution(boxes, categories, anchor_configs, output_dir)

    return boxes, categories


def calculate_anchors(boxes, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    return anchors

def generate_anchor_config(class_name, anchor_sizes, matched_threshold=0.6, unmatched_threshold=0.45, bottom_height=-0.3):
    anchor_sizes = [[round(size, 2) for size in anchor] for anchor in anchor_sizes.tolist()]
    
    config = {
        'class_name': class_name,
        'anchor_sizes': anchor_sizes,
    }
    return config

def save_anchor_configs(anchor_configs, output_file):
    with open(output_file, 'w') as f:
        json.dump(anchor_configs, f, indent=4, separators=(',', ': '))

def visualize_anchors_and_distribution(boxes, categories, anchor_configs, output_dir, sample_size=100):
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(anchor_configs)))
    
    unique_categories = set(categories)

    for config, color in zip(anchor_configs, colors):
        class_name = config['class_name']
        category_boxes = boxes[categories == class_name]
        anchor_sizes = config['anchor_sizes']
        
        if len(category_boxes) > 0:
            plt.figure(figsize=(12, 6))
            if len(category_boxes) > sample_size:
                category_boxes = category_boxes[np.random.choice(len(category_boxes), sample_size, replace=False)]
            
            for length, width, height in category_boxes:
                rect = plt.Rectangle((-length/2, -width/2), length, width, linewidth=0.5, edgecolor=color, facecolor='none', alpha=0.5)
                plt.gca().add_patch(rect)
            
            for size in anchor_sizes:
                l, w, h = size
                rect = plt.Rectangle((-l/2, -w/2), l, w, linewidth=3, edgecolor=color, facecolor='none', alpha=1.0)
                plt.gca().add_patch(rect)
            
            plt.xlabel('Length (L)')
            plt.ylabel('Width (W)')
            plt.title(f'Cuboid Distribution for {class_name}')
            plt.legend([class_name], loc=1)
            plt.grid(True)
            x_range, y_range  = category_boxes[:, 0].max() * 1.5, category_boxes[:, 1].max() * 1.5
            plt.xlim(-x_range/2, x_range/2)
            plt.ylim(-y_range/2, y_range/2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(output_dir / f'{class_name}_distribution.png')
            plt.close()

    fig, ax = plt.subplots(figsize=(18, 12))

    for config, color in zip(anchor_configs, colors):
        class_name = config['class_name']
        anchor_sizes = config['anchor_sizes']
        for size in anchor_sizes:
            l, w, h = size
            rect = plt.Rectangle((-l/2, -w/2), l, w, linewidth=2, edgecolor=color, facecolor='none', label=class_name)
            ax.add_patch(rect)
            ax.text(0, 0, class_name, horizontalalignment='center', verticalalignment='center', color=color, fontsize=8)

    ax.set_xlabel('Length (L)')
    ax.set_ylabel('Width (W)')
    plt.title('Anchor Boxes')
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_dir / 'anchor_boxes.png')
    plt.close()

def parse_config():
    parser = argparse.ArgumentParser(description='Dataset Visualization')
    parser.add_argument('--cfg_file', type=str, default='/home/ailab/Project/05_AICube/OpenPCDet/tools/cfgs/dataset_configs/rscube_dataset_v5.yaml',
                        help='specify the config for visualization')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    parser.add_argument('--vis_frame', action='store_true', help='Enable frame-wise visualization')
    parser.add_argument('--vis_dist', action='store_true', default=True, help='Enable class distribution visualization')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, dataset_cfg = parse_config()  
    logger = common_utils.create_logger()
    logger.info('-----------------Dataset Anchor Calculation-------------------------')

    create_custom_anchors(dataset_cfg, args, logger)

if __name__ == '__main__':
    main()
