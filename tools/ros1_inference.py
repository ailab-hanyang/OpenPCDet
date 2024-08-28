#!/usr/bin/env python3
# Copyright (c) 2023 AI Lab.

# ROS Library
import rospy
import ros_numpy

# Messages
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Python libraries
import numpy as np
import torch
import time 
import glob
from pathlib import Path
from pyquaternion import Quaternion
import yaml
from easydict import EasyDict

# OpenPCDet Library
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config_custom(config=config, new_config=new_config)

    return config

def merge_new_config_custom(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config_custom(config[key], val)

    return config


class OnlineDetection:
    def __init__(self):
        rospy.init_node('online_detection')
        '''
        # Model Parameter
        '''
        self.points = None
        self.device = None
        self.net = None
        
        '''
        # Parameter
        '''
        self.remove_nans = True
        self.pointcloud_flag = False
        self.inferencing = False
        
        '''
        # ROS Parameter
        '''
        # self.model_path = "../output/rscube_models/pointpillar_voxel0016/default/ckpt/latest_model.pth"
        # self.config_path = "../output/rscube_models/pointpillar_voxel0016/default/pointpillar_voxel0016.yaml"
        self.model_path = "/home/ailab/AILabDataset/03_Shared_Repository/jinsu/10_AICube/1st_training/custom_models/pp_v020/default/ckpt/checkpoint_epoch_100.pth"
        self.config_path = "/home/ailab/AILabDataset/03_Shared_Repository/jinsu/10_AICube/1st_training/custom_models/pp_v020/default/pp_v020.yaml"
        self.score_thresh = 0.6
        self.sub_lidar_topic = "/velodyne_points"
        self.z_offset = 1.65 # meter
        self.pub_jsk_objects_topic = "/openpcdet/jsk_bbox"
        self.pub_rviz_objects_topic = "/openpcdet/marker_bbox"
        self.pub_delayed_lidar_topic = "/velodyne_points/delayed"

        '''
        # Publisher (change topic and var name)
        '''
        self.pub_jsk_objects = rospy.Publisher(self.pub_jsk_objects_topic, BoundingBoxArray, queue_size=1)
        self.pub_rviz_objects = rospy.Publisher(self.pub_rviz_objects_topic, MarkerArray, queue_size=1) 
        self.pub_delayed_lidar_points = rospy.Publisher(self.pub_delayed_lidar_topic, PointCloud2, queue_size=1) 
        '''
        # Subscriber (change topic and var name)
        ''' 
        self.sub_lidar_points_ = rospy.Subscriber(self.sub_lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        
        '''
        # Load configuration
        '''
        self.read_config()

        '''
        # Member Variable
        '''         
        self.header = Header()
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.gpu_time_elapsed = 0

    def lidar_callback(self, msg):
        if self.pointcloud_flag == True:
            return
        self.recevied_raw_pc = msg
        self.header = msg.header
        self.pointcloud_flag = True
        
    def read_config(self):
        config_path = self.config_path
        
        cfg_from_yaml_file(config_path, cfg)
        
        # Create POINT_FEATURE_ENCODING to remove dataset_config dependency
        cfg['DATA_CONFIG']['POINT_FEATURE_ENCODING'] = EasyDict(
            {
            'encoding_type': 'absolute_coordinates_encoding',
            'used_feature_list': ['x', 'y', 'z', 'intensity'],
            'src_feature_list': ['x', 'y', 'z', 'intensity']
            }
        )

        self.logger = common_utils.create_logger()
        self.class_names = cfg.CLASS_NAMES
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/ailab/Dataset/000000.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

        # print("Using GPU: %s" %(torch.cuda.get_device_name(torch.cuda.current_device())))


    def inference(self, points):
        num_features = 4 # x,y,z,i
        self.points = points.reshape([-1, num_features])
        # self.points = np.concatenate((self.points, np.zeros((self.points.shape[0], 1))), axis=1) # x,y,z,i,t

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        
        with torch.no_grad(): 
            torch.cuda.synchronize()
            self.starter.record()
            pred_dicts, _ = self.net.forward(data_dict)
            self.ender.record()
            torch.cuda.synchronize()
            self.gpu_time_elapsed = self.starter.elapsed_time(self.ender)

        pred = pred_dicts[0] # w/o confidence thresholding
        boxes_lidar = pred["pred_boxes"].detach().cpu().numpy()
        scores = pred["pred_scores"].detach().cpu().numpy()
        types = pred["pred_labels"].detach().cpu().numpy()

        return scores, boxes_lidar, types

    def pointcloud_preprocessing(self, received_raw_pc):
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(received_raw_pc)
        self.delayed_pc = received_raw_pc
        
        if self.remove_nans == True:
            mask = np.isfinite(msg_cloud['x']) & np.isfinite(msg_cloud['y']) & np.isfinite(msg_cloud['z'])
            msg_cloud = msg_cloud[mask]
            
        points = np.zeros(msg_cloud.shape + (4,), dtype=np.float64)
        points[...,0] = msg_cloud['x']
        points[...,1] = msg_cloud['y']
        points[...,2] = msg_cloud['z'] + self.z_offset
        points[...,3] = msg_cloud['intensity']

        return points

    def update_objects(self, scores, pred_bbox, types, header):
        jsk_objects = BoundingBoxArray()
        jsk_objects.header = header
        rviz_objects = MarkerArray()
        
        # Visualize the ego detection removal area
        ego_length_half = 0.5
        ego_width_half = 0.3
        ego_removal_marker = Marker()
        ego_removal_marker.header = header
        ego_removal_marker.ns = "ego_area"
        ego_removal_marker.id = 0
        ego_removal_marker.type = Marker.LINE_STRIP
        ego_removal_marker.action = Marker.ADD
        ego_removal_marker.scale.x = 0.1  # Line width
        points = [
            (-ego_length_half, -ego_width_half, 0), (-ego_length_half, ego_width_half, 0),
            (ego_length_half, ego_width_half, 0), (ego_length_half, -ego_width_half, 0),
            (-ego_length_half, -ego_width_half, 0)  # Close the loop
        ]
        for point in points:
            p = Point()
            p.x, p.y, p.z = point
            ego_removal_marker.points.append(p)

        ego_removal_marker.color.r = 1.0
        ego_removal_marker.color.g = 1.0
        ego_removal_marker.color.b = 0.0
        ego_removal_marker.color.a = 1.0

        rviz_objects.markers.append(ego_removal_marker)
        
        if pred_bbox.size != 0:
            for i in range(pred_bbox.shape[0]):
                # Remove Ego detection
                if abs(pred_bbox[i][0]) < ego_length_half and abs(pred_bbox[i][1]) < ego_width_half:
                    continue

                classification = int(types[i]) - 1 # Map to 0 ~ 3

                # Marker: Confidence
                rviz_object_confidence = Marker()
                rviz_object_confidence.header = header
                rviz_object_confidence.pose.position.x = float(pred_bbox[i][0])
                rviz_object_confidence.pose.position.y = float(pred_bbox[i][1])
                rviz_object_confidence.pose.position.z = float(pred_bbox[i][2]) + 1.8
                rviz_object_confidence.id = i
                rviz_object_confidence.ns = "Info"
                rviz_object_confidence.action = Marker.ADD
                rviz_object_confidence.type = Marker.TEXT_VIEW_FACING
                rviz_object_confidence.lifetime = rospy.Duration(0.1)
                rviz_object_confidence.color.r = 1.0
                rviz_object_confidence.color.g = 1.0
                rviz_object_confidence.color.b = 1.0
                rviz_object_confidence.color.a = 1.0
                rviz_object_confidence.scale.z = 1.0
                rviz_object_confidence.text = "<{}|{:.2f}>".format(self.class_names[classification], scores[i])

                # Marker: Location
                rviz_object_location = Marker()
                rviz_object_location.header = header
                rviz_object_location.pose.position.x = float(pred_bbox[i][0])
                rviz_object_location.pose.position.y = float(pred_bbox[i][1])
                rviz_object_location.pose.position.z = float(pred_bbox[i][2]) - self.z_offset
                rviz_object_location.scale.x = float(pred_bbox[i][3])
                rviz_object_location.scale.y = float(pred_bbox[i][4])
                rviz_object_location.scale.z = float(pred_bbox[i][5])
                q = yaw2quaternion(float(pred_bbox[i][6]))
                rviz_object_location.pose.orientation.x = q[1]
                rviz_object_location.pose.orientation.y = q[2]
                rviz_object_location.pose.orientation.z = q[3]
                rviz_object_location.pose.orientation.w = q[0]
                rviz_object_location.id = i
                rviz_object_location.ns = "Location"
                rviz_object_location.action = Marker.ADD
                rviz_object_location.type = Marker.CUBE
                rviz_object_location.lifetime = rospy.Duration(0.1)
                rviz_object_location.color.r = 1.0
                rviz_object_location.color.g = 0.0
                rviz_object_location.color.b = 0.0
                rviz_object_location.color.a = 0.5

                ## JSK BoundingBox
                jsk_bbox = BoundingBox()
                jsk_bbox.header = header
                jsk_bbox.pose.position.x = float(pred_bbox[i][0])
                jsk_bbox.pose.position.y = float(pred_bbox[i][1])
                jsk_bbox.pose.position.z = float(pred_bbox[i][2]) - self.z_offset
                q = yaw2quaternion(float(pred_bbox[i][6]))
                jsk_bbox.pose.orientation.x = q[1]
                jsk_bbox.pose.orientation.y = q[2]
                jsk_bbox.pose.orientation.z = q[3]
                jsk_bbox.pose.orientation.w = q[0]
                jsk_bbox.dimensions.x = float(pred_bbox[i][3])
                jsk_bbox.dimensions.y = float(pred_bbox[i][4])
                jsk_bbox.dimensions.z = float(pred_bbox[i][5])
                jsk_bbox.label = int(types[i])
                jsk_bbox.value = float(scores[i])

                # Visualize object with confidence above threshold
                if scores[i] > self.score_thresh:
                    jsk_objects.boxes.append(jsk_bbox)
                    rviz_objects.markers.append(rviz_object_confidence)
                    rviz_objects.markers.append(rviz_object_location)

                i+=1
                        
        return jsk_objects, rviz_objects
        
    def Publish(self, jsk_objects, rviz_objects):
        if len(jsk_objects.boxes) != 0:
            self.pub_jsk_objects.publish(jsk_objects)
            self.pub_rviz_objects.publish(rviz_objects)
            self.pub_delayed_lidar_points.publish(self.delayed_pc)
            rviz_objects.markers.clear()
            jsk_objects.boxes.clear()
            
        else: # Clear objects
            rviz_objects.markers.clear()
            jsk_objects.boxes.clear()
            self.pub_jsk_objects.publish(jsk_objects)
            self.pub_rviz_objects.publish(rviz_objects)

    def Terminate(self):
        # self.sub_lidar_points_.unregister()
        pass

    def Run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.pointcloud_flag == True:
                self.inferencing = True
                self.time_t1 = time.time()
                pointcloud = self.pointcloud_preprocessing(self.recevied_raw_pc)
                
                if len(pointcloud) > 0:
                    scores, lidar_bbox, types = self.inference(pointcloud)
                    rospy.logwarn("Inference time : {:.4f}".format(self.gpu_time_elapsed))
                    jsk_objects, rviz_objects = self.update_objects(scores, lidar_bbox, types, self.header)                  
                    self.Publish(jsk_objects, rviz_objects)
                
                self.pointcloud_flag = False
                self.inferencing = False

                rate.sleep()

   
if __name__ == "__main__":
    online_detection = OnlineDetection()
    online_detection.Run()
