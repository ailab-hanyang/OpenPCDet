import logging
from pathlib import Path
from typing import Final
import numpy as np
from sklearn.cluster import KMeans
import json
from rich.progress import track
import matplotlib.pyplot as plt
import os

import av2.utils.io as io_utils
from av2.utils.synchronization_database import SynchronizationDB
from av2.structures.cuboid import CuboidList
from av2.datasets.sensor.constants import AnnotationCategories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOME_DIR: Final = Path.home()

def load_data(dataset_dir: Path, log_ids: list):
    all_boxes = []
    all_categories = []

    for log_id in track(log_ids, description="Processing logs..."):
        annotations_feather_path = Path(dataset_dir) / log_id / "annotations.feather"
        annotations = CuboidList.from_feather(annotations_feather_path)

        for cuboid in annotations.cuboids:
            length = cuboid.length_m
            width = cuboid.width_m
            height = cuboid.height_m
            category = cuboid.category

            all_boxes.append([length, width, height])
            all_categories.append(category)

    return np.vstack(all_boxes), all_categories

def calculate_anchors(boxes, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    return anchors

def generate_anchor_config(class_name, anchor_sizes, matched_threshold=0.6, unmatched_threshold=0.45, bottom_height=-0.3):
    config = {
        'class_name': class_name,
        'anchor_sizes': anchor_sizes.tolist(),
        'anchor_rotations': [0, 1.57],
        'anchor_bottom_heights': [bottom_height],
        'align_center': False,
        'feature_map_stride': 4,
        'matched_threshold': matched_threshold,
        'unmatched_threshold': unmatched_threshold
    }
    return config

def visualize_anchors_and_distribution(boxes, categories, anchor_configs, output_dir, sample_size=100):
    output_dir.mkdirs(parents=True, exist_ok=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(anchor_configs)))
    
    unique_categories = set(categories)

    # 개별 클래스 분포 시각화 및 저장
    for config, color in zip(anchor_configs, colors):
        class_name = config['class_name']
        category_boxes = boxes[np.array(categories) == class_name]
        anchor_sizes = config['anchor_sizes']
        
        if len(category_boxes) > 0:
            plt.figure(figsize=(12, 6))
            # 랜덤 샘플링
            if len(category_boxes) > sample_size:
                category_boxes = category_boxes[np.random.choice(len(category_boxes), sample_size, replace=False)]
            
            for length, width, height in category_boxes:
                rect = plt.Rectangle((-length/2, -width/2), length, width, linewidth=0.5, edgecolor=color, facecolor='none', alpha=0.5)
                plt.gca().add_patch(rect)
            
            # 결정된 Anchor box 시각화
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
    
    # 모든 클래스에 대한 앵커 박스 시각화 및 저장
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

def main():
    data_root = HOME_DIR / "AILabDataset" / "01_Open_Dataset" / "24_Argoverse2" / "sensor"
    split_name = "train"
    # split_name = "val"  # For debug
    _sdb = SynchronizationDB(data_root / split_name)
    log_ids = _sdb.get_valid_logs()

    num_clusters = 2  # Adjust the number of clusters per class

    logger.info("Loading data from Argoverse2 dataset ...")
    boxes, categories = load_data(data_root / split_name, log_ids)
    
    unique_categories = set(categories)
    anchor_configs = []
    
    for category in track(unique_categories, description="Processing categories..."):
        category_boxes = boxes[np.array(categories) == category]
        if len(category_boxes) == 0:
            continue  # Skip categories with no data
        anchors = calculate_anchors(category_boxes, num_clusters)
        config = generate_anchor_config(category, anchors)
        anchor_configs.append(config)
    
    output_file = "tools/anchor/anchor_config.json"
    with open(output_file, 'w') as f:
        json.dump(anchor_configs, f, indent=4)
    
    logger.info(f"Anchor configuration saved to {output_file}")
    
    output_dir = "tools" / "anchor" / Path("visualizations") / split_name
    visualize_anchors_and_distribution(boxes, categories, anchor_configs, output_dir)

if __name__ == "__main__":
    main()
