import os
import argparse
import json
from rich.progress import track
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from process_tools import dataset_utils

def calculate_anchors(boxes, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    return anchors

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

def parse_config():
    parser = argparse.ArgumentParser(description='RSCube Anchor Calculation')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/rscube_dataset_v3.yaml',
                        help='specify the config for anchor calculation')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def load_boxes(dataset, logger):
    
    boxes = []
    categories = []
    # cnt = 0
    # Convert gt_boxes into list
    for data_dict in track(dataset, description="Processing Frames..."):
        # if cnt > 100: 
            # break
        gt_boxes = data_dict['gt_boxes']

        for gt_boxe in gt_boxes:
            boxes.append([gt_boxe[3], gt_boxe[4], gt_boxe[5]])
            categories.append(int(gt_boxe[7]))  # Ensure that category is int
        # cnt += 1
    # Convert categories from numbers to class names
    categories = [cfg.CLASS_NAMES[category-1] for category in categories]
    
    return np.array(boxes), np.array(categories)

def main():
    args, dataset_cfg = parse_config()  
    logger = common_utils.create_logger()
    logger.info('-----------------  Anchor Calculation -----------------')

    # Build dataset
    dataset_cfg['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] = list() # Do not augment data for anchor calc.
    dataset = dataset_utils.build_dataset(dataset_cfg)
    logger.info(f'Total number of samples: \t{len(dataset)}')
    
    # Create dataset-specific configuration
    if dataset_cfg.DATASET == 'Argo2Dataset':
        num_clusters = {category: 1 for category in dataset_cfg.CLASS_NAMES}
        dataset_name = 'argo2'

    elif dataset_cfg.DATASET == 'CustomDataset':
        num_clusters = { # Only for RSCube Dataset
            "Bus": 1,
            "Pedestrian": 1,
            "Regular_vehicle": 1,
            "Bicycle": 1
        }
        dataset_name = 'rscube'
    
    # Load boxes and categories
    boxes, categories = load_boxes(dataset, logger)
    unique_categories = set(categories)
    
    # Calculate anchors for each category
    anchor_configs = []
    for category in unique_categories:
        category_boxes = boxes[categories == category]
        if len(category_boxes) == 0:
            continue  # Skip categories with no data
        anchors = calculate_anchors(category_boxes, num_clusters[category])
        config = generate_anchor_config(category, anchors)
        anchor_configs.append(config)
    

    # Save anchor configuration
    output_dir = Path(f"anchor/{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / 'anchor_config.json'
    logger.info(f"Anchor visualization saved to {output_file}")
    save_anchor_configs(anchor_configs, output_file)
    
    # Save anchor visualization
    logger.info(f"Anchor visualization saved to {output_dir}")
    visualize_anchors_and_distribution(boxes, categories, anchor_configs, output_dir)

if __name__ == '__main__':
    main()