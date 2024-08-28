import argparse
from pathlib import Path
import logging

from pcdet.config import cfg, cfg_from_yaml_file
import pcdet.datasets as datasets

ARGO2_CLASSES=['Regular_vehicle', 'Pedestrian', 'Bicyclist', 'Motorcyclist', 'Wheeled_rider',
               'Bollard', 'Construction_cone', 'Sign', 'Construction_barrel', 'Stop_sign', 'Mobile_pedestrian_crossing_sign', 
               'Large_vehicle', 'Bus', 'Box_truck', 'Truck', 'Vehicular_trailer', 'Truck_cab', 'School_bus', 'Articulated_bus',
               'Message_board_trailer', 'Bicycle', 'Motorcycle', 'Wheeled_device', 'Wheelchair', 'Stroller', 'Dog']
# ARGO2_CLASSES_SIMPLE=['Regular_vehicle', 'Large_vehicle', 'Pedestrian', 'Bicyclist']
RSCUBE_CLASSES=['Regular_vehicle', 'Bus', 'Pedestrian', 'Bicycle']

def build_dataset(dataset_cfg):
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set class names
    if dataset_cfg.DATASET == 'Argo2Dataset':
        dataset_cfg.CLASS_NAMES = ARGO2_CLASSES
    elif dataset_cfg.DATASET == 'CustomDataset':
        dataset_cfg.CLASS_NAMES = RSCUBE_CLASSES

    dataset = datasets.__all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=dataset_cfg.CLASS_NAMES,
        root_path=Path(dataset_cfg.DATA_PATH),
        training=True,
        # training=False, logger=logger, # Debugging (small dataset)
        logger=logger,
    )
    
    return dataset
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Utils')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/rscube_dataset_v3.yaml', \
                        help='specify the dataset configuration file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    dataset = build_dataset(cfg)
