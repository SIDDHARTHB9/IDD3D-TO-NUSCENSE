import os
import json
import logging
from abc import ABC, abstractmethod
from intermediate_format import *

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
        
        required_dirs = ['label', 'lidar', 'camera']
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = os.path.join(sequence_path, dir_name)
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return {
                'valid': False,
                'error': f'Missing directories: {", ".join(missing_dirs)}'
            }
        
        label_dir = os.path.join(sequence_path, 'label')
        lidar_dir = os.path.join(sequence_path, 'lidar')
        
        label_count = len([f for f in os.listdir(label_dir) if f.endswith('.json')])
        lidar_count = len([f for f in os.listdir(lidar_dir) if f.endswith('.pcd')])
        
        return {
            'valid': True,
            'info': {
                'sequence_name': os.path.basename(sequence_path),
                'label_files': label_count,
                'lidar_files': lidar_count
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
            log.info(f"Found {len(frame_ids)} frames in annot_data.json")
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
            camera_intrinsic=[]
        ))
        
        for standard_cam_name in self.idd3d_to_standard_cameras.values():
            data.calibrations.append(IFCalibration(
                sensor_name=standard_cam_name,
                translation=[0.0, 0.0, 1.6],  
                rotation=[1.0, 0.0, 0.0, 0.0],  
                camera_intrinsic=self.CAMERA_INTRINSIC
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
                log.warning(f"Label file not found: {label_path}")
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
                    rot = psr.get("rotation", {})
                    scl = psr.get("scale", {})
                    
                    translation = [
                        pos.get("x", 0.0),
                        pos.get("y", 0.0),
                        pos.get("z", 0.0)
                    ]
                    
                    size = [
                        scl.get("x", 1.0),
                        scl.get("y", 1.0),
                        scl.get("z", 1.0)
                    ]
                    
                    rotation_quat = [1.0, 0.0, 0.0, 0.0]
                    
                    attributes = []
                    
                    data.annotations.append(IFAnnotation(
                        temp_instance_id=obj_id,
                        temp_frame_id=frame_id,
                        timestamp_us=timestamp,
                        translation=translation,
                        size=size,
                        rotation=rotation_quat,
                        attributes=attributes
                    ))
            
            except Exception as e:
                log.error(f"Error processing label file {label_path}: {e}")
        
        log.info("=" * 50)
        log.info("IDD3D Read Complete")
        log.info("=" * 50)
        log.info(f"Sequence:       {sequence_name}")
        log.info(f"Scenes:         {len(data.scenes)}")
        log.info(f"Samples:        {len(data.samples)}")
        log.info(f"Instances:      {len(data.instances)}")
        log.info(f"Annotations:    {len(data.annotations)}")
        log.info(f"SensorData:     {len(data.sensor_data)}")
        log.info(f"EgoPoses:       {len(data.ego_poses)}")
        log.info(f"Calibrations:   {len(data.calibrations)}")
        log.info("=" * 50)
        
        return data
