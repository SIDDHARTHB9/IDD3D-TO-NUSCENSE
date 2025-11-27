import os
import json
import logging
import glob
import re
from abc import ABC, abstractmethod
from intermediate_format import *

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

class BaseReader(ABC):    
    @abstractmethod
    def read(self, sequence_path: str) -> IntermediateData:
        pass
    
    @abstractmethod
    def validate(self, sequence_path: str) -> dict:
        pass

class Idd3dReader(BaseReader):    
    FRAME_RATE_HZ = 10
    BASE_TIMESTAMP_US = 1640995200000000  
    def __init__(self):
        self.idd3d_to_standard_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle',
            'Person': 'movable_object.pedestrian',
            'Auto': 'movable_object.van',
            'Rider': 'movable_object.bicyclerider',
            'Animal': 'movable_object.unknown',
            'TrafficLight': 'movable_object.unknown',
            'TrafficSign': 'movable_object.unknown',
            'Pole': 'movable_object.unknown',
            'OtherVehicle': 'movable_object.unknown',
            'Misc': 'movable_object.unknown'
        }
        
        self.idd3d_to_standard_cameras = {
            "cam0": "CAM_FRONT_LEFT",
            "cam1": "CAM_BACK_RIGHT",
            "cam2": "CAM_FRONT_RIGHT",
            "cam3": "CAM_FRONT",
            "cam4": "CAM_BACK_LEFT",
            "cam5": "CAM_BACK"
        }
        
        self.LIDAR_CHANNEL = "LIDAR_TOP"
        
        self.CAMERA_INTRINSIC = [
            [2916.0, 0.0, 720.0],
            [0.0, 2916.0, 540.0],
            [0.0, 0.0, 1.0]
        ]

    def validate(self, sequence_path: str) -> dict:
        sequence_path = os.path.abspath(sequence_path)
        
        if not os.path.isdir(sequence_path):
            return {
                'valid': False,
                'error': f'Not a directory: {sequence_path}'
            }
        
        annot_json_path = os.path.join(sequence_path, 'annot_data.json')
        if not os.path.exists(annot_json_path):
            return {
                'valid': False,
                'error': f'Missing annot_data.json in {sequence_path}'
            }
        
        return {
            'valid': True,
            'info': {
                'sequence_name': os.path.basename(sequence_path)
            }
        }

    def read(self, sequence_path: str) -> IntermediateData:
        log.info(f"Reading IDD3D sequence: {sequence_path}")
        
        validation = self.validate(sequence_path)
        if not validation['valid']:
            raise FileNotFoundError(validation['error'])
        
        sequence_path = os.path.abspath(sequence_path)
        sequence_name = os.path.basename(sequence_path)
        
        annot_json_path = os.path.join(sequence_path, 'annot_data.json')
        label_dir = os.path.join(sequence_path, 'label')
        
        try:
            with open(annot_json_path, 'r') as f:
                annot_data = json.load(f)
            frame_ids = sorted(annot_data.keys())
        except Exception as e:
            raise ValueError(f"Failed to read {annot_json_path}: {e}")
        
        data = IntermediateData(sequence_path=sequence_path)
        
        data.scenes.append(IFScene(
            name=sequence_name,
            description=f"IDD3D sequence {sequence_name}"
        ))
        
        data.calibrations.append(IFCalibration(
            sensor_name=self.LIDAR_CHANNEL,
            translation=[0.0, 0.0, 1.8],  
            rotation=[1.0, 0.0, 0.0, 0.0],  
            camera_intrinsic=[],
            distortion=[],
            resolution=[]
        ))
        
        for standard_cam_name in self.idd3d_to_standard_cameras.values():
            data.calibrations.append(IFCalibration(
                sensor_name=standard_cam_name,
                translation=[0.0, 0.0, 1.6],  
                rotation=[1.0, 0.0, 0.0, 0.0],  
                camera_intrinsic=self.CAMERA_INTRINSIC,
                distortion=[0.0, 0.0, 0.0],
                resolution=[1440, 1080]
            ))
        
        frame_interval_us = int(1_000_000 / self.FRAME_RATE_HZ)  
        instance_tracker = set()  
        
        for i, frame_id in enumerate(frame_ids):
            timestamp = self.BASE_TIMESTAMP_US + (i * frame_interval_us)
            
            data.samples.append(IFSample(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                scene_name=sequence_name
            ))
            
            data.ego_poses.append(IFEgoPose(
                temp_frame_id=frame_id,
                timestamp_us=timestamp,
                translation=[0.0, 0.0, 0.0],  
                rotation=[1.0, 0.0, 0.0, 0.0]  
            ))
            
            data.sensor_data.append(IFSensorData(
                temp_frame_id=frame_id,
                sensor_name=self.LIDAR_CHANNEL,
                original_filename=f"{frame_id}.pcd",
                timestamp_us=timestamp,
                is_keyframe=True
            ))
            
            for idd_cam, standard_cam in self.idd3d_to_standard_cameras.items():
                data.sensor_data.append(IFSensorData(
                    temp_frame_id=frame_id,
                    sensor_name=standard_cam,
                    original_filename=f"{idd_cam}/{frame_id}.png",
                    timestamp_us=timestamp,
                    is_keyframe=True
                ))
            
            label_path = os.path.join(label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = str(obj.get("obj_id"))
                    obj_type = obj.get("obj_type")
                    
                    if not obj_id or not obj_type:
                        continue
                    
                    if obj_id not in instance_tracker:
                        category_name = self.idd3d_to_standard_categories.get(
                            obj_type, 
                            'movable_object.unknown'  
                        )
                        
                        data.instances.append(IFInstance(
                            temp_instance_id=obj_id,
                            category_name=category_name
                        ))
                        instance_tracker.add(obj_id)
                    
                    psr = obj.get("psr", {})
                    pos = psr.get("position", {})
                    scl = psr.get("scale", {})
                    
                    data.annotations.append(IFAnnotation(
                        temp_instance_id=obj_id,
                        temp_frame_id=frame_id,
                        timestamp_us=timestamp,
                        translation=[
                            pos.get("x", 0.0),
                            pos.get("y", 0.0),
                            pos.get("z", 0.0)
                        ],
                        size=[
                            scl.get("x", 1.0),
                            scl.get("y", 1.0),
                            scl.get("z", 1.0)
                        ],
                        rotation=[1.0, 0.0, 0.0, 0.0]
                    ))
            
            except Exception:
                pass
        
        return data

class Argoverse2Reader(BaseReader):
    def __init__(self):
        self.av2_to_standard_categories = {
            "REGULAR_VEHICLE": "vehicle.car",
            "PEDESTRIAN": "movable_object.pedestrian",
            "BICYCLIST": "movable_object.bicyclerider",
            "BICYCLE": "vehicle.bicycle",
            "BUS": "vehicle.bus",
            "TRUCK": "vehicle.truck",
            "TRUCK_CAB": "vehicle.truck",
            "TRAILER": "vehicle.truck",
            "LARGE_VEHICLE": "vehicle.truck",
            "MOTORCYCLIST": "movable_object.scooterrider",
            "WHEELED_RIDER": "movable_object.bicyclerider",
            "BOLLARD": "movable_object.unknown",
            "CONSTRUCTION_CONE": "movable_object.unknown",
            "SIGN": "movable_object.unknown",
            "MPV": "vehicle.car",
            "VEHICLE": "vehicle.car",
            "UNKNOWN": "movable_object.unknown"
        }

    def validate(self, sequence_path: str) -> dict:
        sequence_path = os.path.abspath(sequence_path)
        if not os.path.isdir(sequence_path):
            return {'valid': False, 'error': f'Not a directory: {sequence_path}'}
        
        ego_feather = os.path.join(sequence_path, "city_SE3_egovehicle.feather")
        if not os.path.exists(ego_feather):
            return {'valid': False, 'error': f'Missing city_SE3_egovehicle.feather in {sequence_path}'}
            
        return {'valid': True, 'info': {'sequence_name': os.path.basename(sequence_path)}}

    def read(self, sequence_path: str) -> IntermediateData:
        log.info(f"Reading Argoverse 2 sequence: {sequence_path}")
        sequence_path = os.path.abspath(sequence_path)
        sequence_name = os.path.basename(sequence_path)
        data = IntermediateData(sequence_path=sequence_path)
        
        data.scenes.append(IFScene(name=sequence_name, description=f"Argoverse 2 sequence {sequence_name}"))
        
        ego_path = os.path.join(sequence_path, "city_SE3_egovehicle.feather")
        ego_df = pd.read_feather(ego_path)
        ego_df = ego_df.sort_values('timestamp_ns')
        
        for _, row in ego_df.iterrows():
            ts_us = int(row['timestamp_ns'] / 1000)
            frame_id = str(ts_us) 
            
            data.samples.append(IFSample(
                temp_frame_id=frame_id,
                timestamp_us=ts_us,
                scene_name=sequence_name
            ))
            
            data.ego_poses.append(IFEgoPose(
                temp_frame_id=frame_id,
                timestamp_us=ts_us,
                translation=[row['tx_m'], row['ty_m'], row['tz_m']],
                rotation=[row['qw'], row['qx'], row['qy'], row['qz']]
            ))
            
        calib_path = os.path.join(sequence_path, "calibration", "egovehicle_SE3_sensor.feather")
        int_path = os.path.join(sequence_path, "calibration", "intrinsics.feather")
        
        calib_df = pd.read_feather(calib_path)
        intrinsics_df = pd.read_feather(int_path) if os.path.exists(int_path) else pd.DataFrame()
        
        intrinsics_map = {}
        if not intrinsics_df.empty:
            for _, row in intrinsics_df.iterrows():
                intrinsics_map[row['sensor_name']] = row
        
        for _, row in calib_df.iterrows():
            sensor_name = row['sensor_name']
            
            cam_intrinsic = []
            distortion = []
            resolution = []
            
            if sensor_name in intrinsics_map:
                int_row = intrinsics_map[sensor_name]
                cam_intrinsic = [
                    [float(int_row['fx_px']), 0.0, float(int_row['cx_px'])],
                    [0.0, float(int_row['fy_px']), float(int_row['cy_px'])],
                    [0.0, 0.0, 1.0]
                ]
                distortion = [
                    float(int_row.get('k1', 0.0)),
                    float(int_row.get('k2', 0.0)),
                    float(int_row.get('k3', 0.0))
                ]
                resolution = [
                    int(int_row.get('width_px', 0)),
                    int(int_row.get('height_px', 0))
                ]
            
            data.calibrations.append(IFCalibration(
                sensor_name=sensor_name,
                translation=[float(row['tx_m']), float(row['ty_m']), float(row['tz_m'])],
                rotation=[float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])],
                camera_intrinsic=cam_intrinsic,
                distortion=distortion,
                resolution=resolution
            ))

        ann_path = os.path.join(sequence_path, "annotations.feather")
        if not os.path.exists(ann_path):
            ann_path = os.path.join(sequence_path, "annotation.feather")
            
        if os.path.exists(ann_path):
            ann_df = pd.read_feather(ann_path)
            instance_tracker = set()
            
            for _, row in ann_df.iterrows():
                ts_us = int(row['timestamp_ns'] / 1000)
                track_id = str(row['track_uuid'])
                category = row['category']
                
                if track_id not in instance_tracker:
                    std_cat = self.av2_to_standard_categories.get(category, "movable_object.unknown")
                    data.instances.append(IFInstance(
                        temp_instance_id=track_id,
                        category_name=std_cat
                    ))
                    instance_tracker.add(track_id)
                
                num_pts = int(row.get('num_interior_pts', 0))
                data.annotations.append(IFAnnotation(
                    temp_instance_id=track_id,
                    temp_frame_id=str(ts_us),
                    timestamp_us=ts_us,
                    translation=[float(row['tx_m']), float(row['ty_m']), float(row['tz_m'])],
                    size=[float(row['width_m']), float(row['length_m']), float(row['height_m'])], 
                    rotation=[float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])],
                    num_lidar_pts=num_pts
                ))

        sensors_dir = os.path.join(sequence_path, "sensors")
        
        cameras_dir = os.path.join(sensors_dir, "cameras")
        lidar_dir = os.path.join(sensors_dir, "lidar")
        
        camera_sensors = ["ring_front_left", "ring_front_right", "ring_front_center",
                         "ring_rear_left", "ring_rear_right", "ring_side_left",
                         "ring_side_right", "stereo_front_left", "stereo_front_right"]
        
        if os.path.exists(cameras_dir):
            for sensor_name in camera_sensors:
                sensor_camera_dir = os.path.join(cameras_dir, sensor_name)
                if os.path.exists(sensor_camera_dir):
                    files = glob.glob(os.path.join(sensor_camera_dir, "*.jpg"))
                    for fpath in files:
                        fname = os.path.basename(fpath)
                        match = re.search(r'(\d+)', fname)
                        if match:
                            ts_ns = int(match.group(1))
                            ts_us = int(ts_ns / 1000)
                            
                            data.sensor_data.append(IFSensorData(
                                temp_frame_id=str(ts_us),
                                sensor_name=sensor_name,
                                original_filename=f"sensors/cameras/{sensor_name}/{fname}",
                                timestamp_us=ts_us
                            ))
        
        if os.path.exists(lidar_dir):
            files = glob.glob(os.path.join(lidar_dir, "*.feather"))
            for fpath in files:
                fname = os.path.basename(fpath)
                match = re.search(r'(\d+)', fname)
                if match:
                    ts_ns = int(match.group(1))
                    ts_us = int(ts_ns / 1000)
                    
                    data.sensor_data.append(IFSensorData(
                        temp_frame_id=str(ts_us),
                        sensor_name="lidar",
                        original_filename=f"sensors/lidar/{fname}",
                        timestamp_us=ts_us
                    ))

        return data
