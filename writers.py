import os
import json
import shutil
import logging
import uuid
import re
import hashlib
from abc import ABC, abstractmethod
from PIL import Image
from datetime import datetime
from intermediate_format import IntermediateData
from utils import append_to_json_list, merge_and_overwrite_json_list, load_json_safely, save_json_safely

try:
    import numpy as np
    import pyarrow.feather as pf
    import pandas as pd
except ImportError:
    pass

try:
    import open3d as o3d
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def convert_lidar_pcd_to_bin(src_path, dst_path):
    try:
        if not os.path.exists(src_path):
            open(dst_path, 'wb').close()
            return
        
        pcd = o3d.io.read_point_cloud(src_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack((xyz, intensity))
        pts.astype(np.float32).tofile(dst_path)
    except Exception:
        open(dst_path, 'wb').close()

def convert_lidar_feather_to_bin(src_path, dst_path):
    try:
        if not os.path.exists(src_path):
            open(dst_path, 'wb').close()
            return
        
        table = pf.read_feather(src_path)
        df = table.to_pandas()
        pts = df[['x', 'y', 'z', 'intensity']].values.astype(np.float32)
        pts.astype(np.float32).tofile(dst_path)
    except Exception:
        open(dst_path, 'wb').close()

def convert_camera_to_jpg(src_path, dst_path, quality=95):
    try:
        if not os.path.exists(src_path):
            return
        
        img = Image.open(src_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_path, 'JPEG', quality=quality)
    except Exception:
        pass

class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        pass

class _NuScenesTokenManager:
    
    def __init__(self, registry_path=None, base_timestamp=None, frame_rate_hz=10):
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / frame_rate_hz)
        
        if base_timestamp is None:
            self.base_timestamp = 1640995200000000  
        else:
            self.base_timestamp = base_timestamp
        
        self.frame_tokens = {}        
        self.instance_tokens = {}     
        self.ego_pose_tokens = {}     
        self.scene_token = None
        self.category_tokens = {}
        self.attribute_tokens = {}
        self.visibility_tokens = {}
        self.map_tokens = {}
        self.log_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        self.registry_path = registry_path
        self._load_registry()
    
    def _load_registry(self):
        if not self.registry_path or not os.path.exists(self.registry_path):
            return
        
        try:
            registry = load_json_safely(self.registry_path, default={})
            self.category_tokens = registry.get('category_tokens', {})
            self.attribute_tokens = registry.get('attribute_tokens', {})
            self.visibility_tokens = registry.get('visibility_tokens', {})
            self.map_tokens = registry.get('map_tokens', {})
            self.log_tokens = registry.get('log_tokens', {})
            self.sensor_tokens = registry.get('sensor_tokens', {})
            self.calibration_tokens = registry.get('calibration_tokens', {})
        except Exception:
            pass
    
    def save_registry(self):
        if not self.registry_path:
            return
        
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'category_tokens': self.category_tokens,
            'attribute_tokens': self.attribute_tokens,
            'visibility_tokens': self.visibility_tokens,
            'map_tokens': self.map_tokens,
            'log_tokens': self.log_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        save_json_safely(self.registry_path, registry)
    
    def get_frame_token(self, frame_id):
        if frame_id not in self.frame_tokens:
            self.frame_tokens[frame_id] = uuid.uuid4().hex
        return self.frame_tokens[frame_id]
    
    def get_ego_pose_token(self, frame_id):
        if frame_id not in self.ego_pose_tokens:
            self.ego_pose_tokens[frame_id] = uuid.uuid4().hex
        return self.ego_pose_tokens[frame_id]
    
    def get_instance_token(self, obj_id):
        if obj_id not in self.instance_tokens:
            self.instance_tokens[obj_id] = uuid.uuid4().hex
        return self.instance_tokens[obj_id]
    
    def get_category_token(self, category_name):
        if category_name not in self.category_tokens:
            self.category_tokens[category_name] = uuid.uuid4().hex
        return self.category_tokens[category_name]
    
    def get_attribute_token(self, attr_name):
        if attr_name not in self.attribute_tokens:
            self.attribute_tokens[attr_name] = uuid.uuid4().hex
        return self.attribute_tokens[attr_name]
    
    def get_map_token(self, map_name):
        if map_name not in self.map_tokens:
            self.map_tokens[map_name] = uuid.uuid4().hex
        return self.map_tokens[map_name]
    
    def get_log_token(self, log_name):
        if log_name not in self.log_tokens:
            self.log_tokens[log_name] = uuid.uuid4().hex
        return self.log_tokens[log_name]
    
    def get_sensor_token(self, sensor_name):
        if sensor_name not in self.sensor_tokens:
            self.sensor_tokens[sensor_name] = uuid.uuid4().hex
        return self.sensor_tokens[sensor_name]
    
    def get_calibration_token(self, sensor_name):
        if sensor_name not in self.calibration_tokens:
            self.calibration_tokens[sensor_name] = uuid.uuid4().hex
        return self.calibration_tokens[sensor_name]
    
    def get_scene_token(self):
        if self.scene_token is None:
            self.scene_token = uuid.uuid4().hex
        return self.scene_token
    
    def generate_annotation_token(self):
        return uuid.uuid4().hex

class NuScenesWriter(BaseWriter):

    OFFICIAL_CATEGORY_TOKENS = {
        "vehicle.motorcycle": "dc39d8b2858e4bc0b7ddf66ede8d734e",
        "movable_object.bicyclerider": "d411b4e8157d445193034d6f408900d3",
        "movable_object.tourcar": "e2325ce5697e45678ee0fe4017918290",
        "movable_object.scooterrider": "9a438c7df65d4ae0b5e87f603a3e91b7",
        "vehicle.bus": "1046b59779f24cf7b55114161208b0f5",
        "movable_object.bicyclegroup": "57c2b779b57b496297048ea55aaed2c7",
        "movable_object.van": "869140488b264d7780ed9cc8233cb5ce",
        "vehicle.truck": "69d88d0df8274f56995aacff1982ec65",
        "movable_object.pedestrian": "9a6c42f9792f40789bc0437eba0aef9b",
        "movable_object.scooter": "f15d03bf64834024a0601aae7a07c156",
        "vehicle.bicycle": "366ad39f728a4ab5ae9a4146f528bd00",
        "movable_object.unknown": "f0add8f1828d4b7ca20d135edd7ecd4e",
        "movable_object.unknown1": "6bc7bdefe76646e193288d5928a2d58a",
        "vehicle.car": "3305eeb43e684538b00bcc41fc38d84e"
    }
    
    def __init__(self):
        self._token_manager = None
        self._output_path = None
        self._annot_dir = None
        self._samples_dir = None
        self._sweeps_dir = None
        self._maps_dir = None
        self._basemap_dir = None       
        self._prediction_dir = None     
        self._expansion_dir = None 
        self._can_bus_dir = None
        self._generated_log_tokens = []
    
    def write(self, data: IntermediateData, output_path: str):
        self._output_path = os.path.abspath(output_path)
        
        self._annot_dir = os.path.join(self._output_path, 'anotations')
        self._samples_dir = os.path.join(self._output_path, 'samples')
        self._sweeps_dir = os.path.join(self._output_path, 'sweeps')
        self._maps_dir = os.path.join(self._output_path, 'maps')
        self._basemap_dir = os.path.join(self._output_path,'basemap')
        self._expansion_dir = os.path.join(self._maps_dir,'expansion')   
        self._prediction_dir =  os.path.join(self._output_path,'prediction')    
        self._can_bus_dir = os.path.join(self._output_path, 'can_bus')
        
        os.makedirs(self._annot_dir, exist_ok=True)
        os.makedirs(self._samples_dir, exist_ok=True)
        os.makedirs(self._maps_dir, exist_ok=True)
        os.makedirs(self._can_bus_dir, exist_ok=True)
        os.makedirs(self._basemap_dir, exist_ok=True)
        os.makedirs(self._expansion_dir, exist_ok=True)
        os.makedirs(self._prediction_dir, exist_ok=True)

        registry_path = os.path.join(self._annot_dir, 'token_registry.json')
        last_timestamp = self._get_last_timestamp()
        new_base_timestamp = (last_timestamp + 1_000_000) if last_timestamp else None
        
        self._token_manager = _NuScenesTokenManager(
            registry_path=registry_path,
            base_timestamp=new_base_timestamp
        )
        
        self._pre_populate_categories()
        
        if not data.scenes:
            return
        
        sequence_name = data.scenes[0].name
        
        self._write_sensor_and_calib(data.calibrations)
        self._write_visibility()
        self._write_attribute()
        self._write_log(data.scenes)
        self._write_map()
        self._write_map_expansion()
        self._write_prediction(data.scenes, data.samples)
        self._write_file_manifest(data, new_base_timestamp)
        self._write_sample_and_ego_pose(data.samples, data.ego_poses, new_base_timestamp)
        self._write_sample_data(data.sensor_data, sequence_name, new_base_timestamp)
        self._write_category(data.instances)
        self._write_instance_and_annotation(data.instances, data.annotations)
        self._write_can_bus(data.scenes, data.samples)
        
        self._process_sensor_files(data.sensor_data, data.sequence_path, sequence_name, new_base_timestamp)
        self._duplicate_sweeps()
        self._token_manager.save_registry()
    
    def _get_last_timestamp(self):
        max_timestamp = None
        
        sample_json_path = os.path.join(self._annot_dir, 'sample.json')
        if os.path.exists(sample_json_path):
            samples = load_json_safely(sample_json_path, default=[])
            if samples and isinstance(samples, list):
                sample_timestamps = [s.get('timestamp') for s in samples if s.get('timestamp')]
                if sample_timestamps:
                    max_timestamp = max(sample_timestamps) if max_timestamp is None else max(max_timestamp, max(sample_timestamps))
        
        ego_pose_json_path = os.path.join(self._annot_dir, 'ego_pose.json')
        if os.path.exists(ego_pose_json_path):
            ego_poses = load_json_safely(ego_pose_json_path, default=[])
            if ego_poses and isinstance(ego_poses, list):
                ego_timestamps = [e.get('timestamp') for e in ego_poses if e.get('timestamp')]
                if ego_timestamps:
                    max_timestamp = max(ego_timestamps) if max_timestamp is None else max(max_timestamp, max(ego_timestamps))
        
        sample_data_json_path = os.path.join(self._annot_dir, 'sample_data.json')
        if os.path.exists(sample_data_json_path):
            sample_data = load_json_safely(sample_data_json_path, default=[])
            if sample_data and isinstance(sample_data, list):
                sd_timestamps = [sd.get('timestamp') for sd in sample_data if sd.get('timestamp')]
                if sd_timestamps:
                    max_timestamp = max(sd_timestamps) if max_timestamp is None else max(max_timestamp, max(sd_timestamps))
        
        return max_timestamp
    
    def _pre_populate_categories(self):
        for cat_name, cat_token in self.OFFICIAL_CATEGORY_TOKENS.items():
            if cat_name not in self._token_manager.category_tokens:
                self._token_manager.category_tokens[cat_name] = cat_token
    
    def _format_scene_name(self, raw_scene_name: str) -> str:
        match = re.search(r'\d+$', raw_scene_name)
        if match:
            num_str = match.group(0)
            return f"scene-{num_str.zfill(3)}"
        else:
            fallback_hash = hashlib.md5(raw_scene_name.encode()).hexdigest()[:3]
            return f"scene-{fallback_hash}"
    
    def _write_can_bus(self, scenes, samples):
        if not scenes or not samples:
            return
        
        for scene in scenes:
            raw_scene_name = scene.name
            formatted_scene_name = self._format_scene_name(raw_scene_name)
            
            scene_samples = [s for s in samples if s.scene_name == raw_scene_name]
            if not scene_samples:
                continue
            
            scene_samples.sort(key=lambda x: x.timestamp_us)
            
            ms_imu_data = []
            pose_data = []
            route_data = []
            steer_data = []
            vehicle_monitor_data = []
            zoesensors_data = []
            zoe_veh_data = []
            
            base_x = 983.0155677603801
            base_y = 569.4627572428807
            dx = 0.12616908872
            dy = 0.15518170334
            
            for i, sample in enumerate(scene_samples):
                ms_imu_data.append({
                    "utime": sample.timestamp_us,
                    "linear_accel": [3.379, 3.379, 3.379],
                    "q": [0.5, 0.5, 0.5, 0.5],
                    "rotation_rate": [0.044, 0.001, 0.282]
                })
                
                pose_data.append({
                    "accel": [3.379, 3.379, 3.379],
                    "orientation": [0.7479305678167669, 0.0, 0.0, 0.663776],
                    "pos": [1010.1436201720262, 610.8882352282457, 0.0],
                    "rotation_rate": [0.040320225059986115, -0.002563952235504985, 0.28492140769958496],
                    "utime": sample.timestamp_us,
                    "vel": [4.1688763951334185, 0.0, 0.0]
                })
                
                route_data.append([base_x + i * dx, base_y + i * dy])
                
                steer_data.append({
                    "utime": sample.timestamp_us,
                    "value": 3.379
                })
                
                vehicle_monitor_data.append({
                    "available_distance": 100.0,
                    "battery_level": 100.0,
                    "brake": 0.0,
                    "brake_switch": 0,
                    "gear_position": 4,
                    "left_signal": 0,
                    "rear_left_rpm": 167.58,
                    "rear_right_rpm": 169.88,
                    "right_signal": 0,
                    "steering": 23.95,
                    "steering_speed": -10.98,
                    "throttle": 95.5,
                    "utime": sample.timestamp_us,
                    "vehicle_speed": 19.32,
                    "yaw_rate": 0.105
                })
                
                zoesensors_data.append({
                    "brake_sensor": 0.172,
                    "steering_sensor": 0.188,
                    "throttle_sensor": 0.192,
                    "utime": sample.timestamp_us
                })
                
                zoe_veh_data.append({
                    "FL_wheel_speed": 166.90,
                    "FR_wheel_speed": 166.90,
                    "RL_wheel_speed": 166.90,
                    "RR_wheel_speed": 166.90,
                    "left_solar": -16.12,
                    "longitudinal_accel": 0.59,
                    "meanEffTorque": 91.67,
                    "odom": 60.67,
                    "odom_speed": 60.67,
                    "pedal_cc": 136.46,
                    "regen": 136.46,
                    "requestedTorqueAfterProc": -338.93,
                    "right_solar": -16.12,
                    "steer_corrected": 27.39,
                    "steer_offset_can": 27.39,
                    "steer_raw": 27.39,
                    "transversal_accel": -0.49,
                    "utime": sample.timestamp_us
                })
            
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_ms_imu.json"), ms_imu_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_pose.json"), pose_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_route.json"), route_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_steeranglefeedback.json"), steer_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_vehicle_monitor.json"), vehicle_monitor_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_zoesensors.json"), zoesensors_data)
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_zoe_veh.json"), zoe_veh_data)
            
            first_ts = scene_samples[0].timestamp_us
            last_ts = scene_samples[-1].timestamp_us
            count = len(scene_samples)
            
            meta_data = {
                "MS_IMU": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "linear_accel": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "q": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "rotation_rate": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "utime": {"max": last_ts, "max_diff": 40353.0, "mean": (first_ts+last_ts)/2, "mean_diff": 10298.1, "min": first_ts, "min_diff": 2620.0, "std": 5648567.4, "std_diff": 2102.8}
                    }
                },
                "POSE": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "accel": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "orientation": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "pos": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "rotation_rate": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87},
                        "utime": {"max": last_ts, "max_diff": 40353.0, "mean": (first_ts+last_ts)/2, "mean_diff": 10298.1, "min": first_ts, "min_diff": 2620.0, "std": 5648567.4, "std_diff": 2102.8},
                        "vel": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87}
                    }
                },
                "SteerAngleFeedback": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "utime": {"max": last_ts, "max_diff": 40353.0, "mean": (first_ts+last_ts)/2, "mean_diff": 10298.1, "min": first_ts, "min_diff": 2620.0, "std": 5648567.4, "std_diff": 2102.8},
                        "value": {"max": 10.96, "max_diff": 11.10, "mean": 3.38, "mean_diff": 4.78, "min": -0.89, "min_diff": -1.35, "std": 4.54, "std_diff": 4.87}
                    }
                },
                "VehicleMonitor": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "available_distance": {"max": 100.0, "max_diff": 0.0, "mean": 100.0, "mean_diff": 0.0, "min": 100.0, "min_diff": 0.0, "std": 0.0, "std_diff": 0.0},
                        "battery_level": {"max": 100.0, "max_diff": 0.0, "mean": 100.0, "mean_diff": 0.0, "min": 100.0, "min_diff": 0.0, "std": 0.0, "std_diff": 0.0},
                        "brake": {"max": 0.0, "max_diff": 0.0, "mean": 0.0, "mean_diff": 0.0, "min": 0.0, "min_diff": 0.0, "std": 0.0, "std_diff": 0.0},
                        "brake_switch": {"max": 0, "max_diff": 0, "mean": 0.0, "mean_diff": 0.0, "min": 0, "min_diff": 0, "std": 0.0, "std_diff": 0.0},
                        "gear_position": {"max": 5, "max_diff": 5, "mean": 3.5, "mean_diff": 2.0, "min": 2, "min_diff": 2, "std": 1.5, "std_diff": 1.58},
                        "left_signal": {"max": 0, "max_diff": 0, "mean": 0.0, "mean_diff": 0.0, "min": 0, "min_diff": 0, "std": 0.46, "std_diff": 0.50},
                        "rear_left_rpm": {"max": 205.37, "max_diff": 17.76, "mean": 167.58, "mean_diff": 2.25, "min": 115.68, "min_diff": -8.88, "std": 28.04, "std_diff": 6.73},
                        "rear_right_rpm": {"max": 210.49, "max_diff": 18.35, "mean": 169.88, "mean_diff": 2.54, "min": 120.68, "min_diff": -7.88, "std": 29.04, "std_diff": 7.73},
                        "right_signal": {"max": 1, "max_diff": 1, "mean": 0.1, "mean_diff": 0.0, "min": 0, "min_diff": 0, "std": 0.32, "std_diff": 0.39},
                        "steering": {"max": 206.4, "max_diff": 20.0, "mean": 23.95, "mean_diff": -6.06, "min": -44.4, "min_diff": -78.2, "std": 63.95, "std_diff": 19.52},
                        "steering_speed": {"max": 65.7, "max_diff": 67.7, "mean": -10.98, "mean_diff": -2.61, "min": -183.3, "min_diff": -123.5, "std": 40.82, "std_diff": 34.91},
                        "throttle": {"max": 202, "max_diff": 72.0, "mean": 95.5, "mean_diff": 2.41, "min": 0, "min_diff": -98.0, "std": 65.78, "std_diff": 33.01},
                        "utime": {"max": last_ts, "max_diff": 40353.0, "mean": (first_ts+last_ts)/2, "mean_diff": 10298.1, "min": first_ts, "min_diff": 2620.0, "std": 5648567.4, "std_diff": 2102.8},
                        "vehicle_speed": {"max": 23.12, "max_diff": 1.19, "mean": 19.32, "mean_diff": 0.21, "min": 14.53, "min_diff": -0.96, "std": 2.96, "std_diff": 0.69},
                        "yaw_rate": {"max": 0.175, "max_diff": 0.035, "mean": 0.105, "mean_diff": 0.005, "min": 0.035, "min_diff": -0.017, "std": 0.039, "std_diff": 0.012}
                    }
                },
                "ZOE_VEH_INFO": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "FL_wheel_speed": {"max": 202.08, "max_diff": 2.84, "mean": 166.90, "mean_diff": 0.04, "min": 118.43, "min_diff": -3.00, "std": 27.19, "std_diff": 0.48},
                        "FR_wheel_speed": {"max": 202.08, "max_diff": 2.84, "mean": 166.90, "mean_diff": 0.04, "min": 118.43, "min_diff": -3.00, "std": 27.19, "std_diff": 0.48},
                        "RL_wheel_speed": {"max": 202.08, "max_diff": 2.84, "mean": 166.90, "mean_diff": 0.04, "min": 118.43, "min_diff": -3.00, "std": 27.19, "std_diff": 0.48},
                        "RR_wheel_speed": {"max": 202.08, "max_diff": 2.84, "mean": 166.90, "mean_diff": 0.04, "min": 118.43, "min_diff": -3.00, "std": 27.19, "std_diff": 0.48},
                        "left_solar": {"max": 0, "max_diff": 112.0, "mean": -16.12, "mean_diff": 0.0, "min": -112, "min_diff": -112.0, "std": 39.32, "std_diff": 3.60},
                        "longitudinal_accel": {"max": 3.5, "max_diff": 4.0, "mean": 0.59, "mean_diff": 0.0, "min": -4.0, "min_diff": -4.0, "std": 1.85, "std_diff": 1.53},
                        "meanEffTorque": {"max": 73.0, "max_diff": 20.0, "mean": 91.67, "mean_diff": 2.0, "min": 50.0, "min_diff": -20.0, "std": 14.49, "std_diff": 7.62},
                        "odom": {"max": 120, "max_diff": 64.0, "mean": 60.67, "mean_diff": 0.004, "min": 0, "min_diff": -84.0, "std": 34.79, "std_diff": 18.79},
                        "odom_speed": {"max": 120, "max_diff": 64.0, "mean": 60.67, "mean_diff": 0.004, "min": 0, "min_diff": -84.0, "std": 34.79, "std_diff": 18.79},
                        "pedal_cc": {"max": 273.0, "max_diff": 15.0, "mean": 136.46, "mean_diff": 0.06, "min": 0.0, "min_diff": -22.0, "std": 89.45, "std_diff": 1.98},
                        "regen": {"max": 273.0, "max_diff": 15.0, "mean": 136.46, "mean_diff": 0.06, "min": 0.0, "min_diff": -22.0, "std": 89.45, "std_diff": 1.98},
                        "requestedTorqueAfterProc": {"max": -272.5, "max_diff": 123.5, "mean": -338.93, "mean_diff": 0.01, "min": -398.5, "min_diff": -125.5, "std": 35.12, "std_diff": 4.06},
                        "right_solar": {"max": 0, "max_diff": 112.0, "mean": -16.12, "mean_diff": 0.0, "min": -112, "min_diff": -112.0, "std": 39.32, "std_diff": 3.60},
                        "steer_corrected": {"max": 206.4, "max_diff": 1.4, "mean": 27.39, "mean_diff": -0.09, "min": -28.4, "min_diff": -2.0, "std": 65.01, "std_diff": 0.46},
                        "steer_offset_can": {"max": 206.4, "max_diff": 1.4, "mean": 27.39, "mean_diff": -0.09, "min": -28.4, "min_diff": -2.0, "std": 65.01, "std_diff": 0.46},
                        "steer_raw": {"max": 206.4, "max_diff": 1.4, "mean": 27.39, "mean_diff": -0.09, "min": -28.4, "min_diff": -2.0, "std": 65.01, "std_diff": 0.46},
                        "transversal_accel": {"max": -0.312, "max_diff": 0.032, "mean": -0.49, "mean_diff": 0.0, "min": -0.6, "min_diff": -0.028, "std": 0.063, "std_diff": 0.006},
                        "utime": {"max": last_ts, "max_diff": 40353.0, "mean": (first_ts+last_ts)/2, "mean_diff": 10298.1, "min": first_ts, "min_diff": 2620.0, "std": 5648567.4, "std_diff": 2102.8}
                    }
                },
                "ZoeSensors": {
                    "message_count": count, "message_frequency": count/10.0, "timespan": 10.0,
                    "var_stats": {
                        "brake_sensor": {"max": 0.201, "max_diff": 0.0003, "mean": 0.172, "mean_diff": 0.0, "min": 0.171, "min_diff": -0.0004, "std": 0.005, "std_diff": 0.00002},
                        "steering_sensor": {"max": 0.190, "max_diff": 0.0006, "mean": 0.188, "mean_diff": 0.0, "min": 0.187, "min_diff": -0.0005, "std": 0.0003, "std_diff": 0.00006},
                        "throttle_sensor": {"max": 0.253, "max_diff": 0.0017, "mean": 0.192, "mean_diff": 0.0, "min": 0.120, "min_diff": -0.0016, "std": 0.041, "std_diff": 0.0002},
                        "utime": {"max": last_ts, "max_diff": 24269.0, "mean": (first_ts+last_ts)/2, "mean_diff": 1050.9, "min": first_ts, "min_diff": 1.0, "std": 5626101.9, "std_diff": 837.4}
                    }
                }
            }
            save_json_safely(os.path.join(self._can_bus_dir, f"{formatted_scene_name}_meta.json"), meta_data)
            
            log.info(f"Generated CAN bus files for {formatted_scene_name}")
        
    def _write_sensor_and_calib(self, calibrations):
        new_sensors = []
        new_calib_sensors = []
        
        for if_calib in calibrations:
            sensor_token = self._token_manager.get_sensor_token(if_calib.sensor_name)
            is_camera = len(if_calib.camera_intrinsic) > 0
            
            new_sensors.append({
                "token": sensor_token,
                "modality": "camera" if is_camera else "lidar",
                "channel": if_calib.sensor_name,
            })
            
            new_calib_sensors.append({
                "token": self._token_manager.get_calibration_token(if_calib.sensor_name),
                "sensor_token": sensor_token,
                "translation": if_calib.translation,
                "rotation": if_calib.rotation,
                "camera_intrinsic": if_calib.camera_intrinsic
            })
        
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'sensor.json'), new_sensors, key_field='channel')
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'calibrated_sensor.json'), new_calib_sensors, key_field='sensor_token')
    
    def _write_visibility(self):
        vis_levels = [
            {"token": "1", "level": "v1-0", "description": "visibility 0-40%"},
            {"token": "2", "level": "v2-0", "description": "visibility 40-60%"},
            {"token": "3", "level": "v3-0", "description": "visibility 60-80%"},
            {"token": "4", "level": "v4-0", "description": "visibility 80-100%"}
        ]
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'visibility.json'), vis_levels, key_field='level')
    
    def _write_attribute(self):
        attributes = [
            {"name": "vehicle.moving", "description": "Vehicle is moving"},
            {"name": "pedestrian.moving", "description": "Pedestrian is moving"},
        ]
        new_entries = []
        for attr in attributes:
            new_entries.append({
                "token": self._token_manager.get_attribute_token(attr["name"]),
                "name": attr["name"],
                "description": attr["description"]
            })
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'attribute.json'), new_entries, key_field='name')
    
    def _write_log(self, scenes):
        new_entries = []
        for if_scene in scenes:
            logfile = f"{if_scene.name}-{datetime.now().strftime('%Y-%m-%d')}"
            log_token = self._token_manager.get_log_token(f"log_{logfile}")
            self._generated_log_tokens.append(log_token)
            new_entries.append({
                "token": log_token,
                "logfile": logfile,
                "vehicle": "stub_vehicle",
                "date_captured": datetime.now().strftime('%Y-%m-%d'),
                "location": "hyderabad"
            })
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'log.json'), new_entries, key_field='token')
    
    def _write_map(self):
        location = "Hyderabad"
        map_token = self._token_manager.get_map_token(f"map_{location}")
        new_map_entry = {
            "token": map_token,
            "log_tokens": self._generated_log_tokens,
            "category": "semantic_prior",
            "filename": f"maps/{location.lower()}.png",
        }
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'map.json'), [new_map_entry], key_field='token')
        
    def _write_map_expansion(self):
        expansion_path = os.path.join(self._expansion_dir, 'singapore-queenstown.json')
        
        node_tokens = [uuid.uuid4().hex for _ in range(100)]
        nodes = []
        for i, token in enumerate(node_tokens):
            nodes.append({"token": token, "x": 1000.0 + i * 50.0, "y": 1500.0 + i * 30.0})
        
        poly_tokens = [uuid.uuid4().hex for _ in range(6)]
        polygons = []
        for i, token in enumerate(poly_tokens):
            start_idx = i * 15
            end_idx = start_idx + 15
            polygons.append({
                "token": token,
                "exterior_node_tokens": node_tokens[start_idx:end_idx] if end_idx <= len(node_tokens) else node_tokens[start_idx:],
                "holes": []
            })
        
        line_tokens = [uuid.uuid4().hex for _ in range(10)]
        lines = []
        for i, token in enumerate(line_tokens):
            start_idx = i * 8
            end_idx = start_idx + 8
            lines.append({
                "token": token,
                "node_tokens": node_tokens[start_idx:end_idx] if end_idx <= len(node_tokens) else node_tokens[start_idx:]
            })
        
        lane_divider_tokens = [uuid.uuid4().hex for _ in range(8)]
        lane_dividers = []
        segment_types = ["DOUBLE_DASHED_WHITE", "SINGLE_SOLID_WHITE", "NIL"]
        for i, token in enumerate(lane_divider_tokens):
            segments = []
            segment_type = segment_types[i % len(segment_types)]
            for j in range(6):
                node_idx = i * 6 + j
                segments.append({
                    "node_token": node_tokens[node_idx] if node_idx < len(node_tokens) else node_tokens[0],
                    "segment_type": segment_type
                })
            lane_dividers.append({
                "token": token,
                "line_token": line_tokens[i % len(line_tokens)],
                "lane_divider_segments": segments
            })
        
        lane_tokens = [uuid.uuid4().hex for _ in range(8)]
        lanes = []
        for i, token in enumerate(lane_tokens):
            left_segments = []
            right_segments = []
            left_divider_idx = i % len(lane_divider_tokens)
            right_divider_idx = (i + 1) % len(lane_divider_tokens)
            
            for seg in lane_dividers[left_divider_idx]["lane_divider_segments"]:
                left_segments.append({"node_token": seg["node_token"], "segment_type": seg["segment_type"]})
            
            for seg in lane_dividers[right_divider_idx]["lane_divider_segments"]:
                right_segments.append({"node_token": seg["node_token"], "segment_type": seg["segment_type"]})
            
            lanes.append({
                "token": token,
                "polygon_token": poly_tokens[i % len(poly_tokens)],
                "lane_type": "CAR",
                "from_edge_line_token": line_tokens[i % len(line_tokens)],
                "to_edge_line_token": line_tokens[(i + 1) % len(line_tokens)],
                "left_lane_divider_segments": left_segments,
                "right_lane_divider_segments": right_segments
            })
        
        road_segment_tokens = [uuid.uuid4().hex for _ in range(5)]
        road_segments = []
        for i, token in enumerate(road_segment_tokens):
            road_segments.append({
                "token": token,
                "polygon_token": poly_tokens[i % len(poly_tokens)],
                "is_intersection": i == 0,
                "lane_tokens": [lane_tokens[i % len(lane_tokens)], lane_tokens[(i+1) % len(lane_tokens)]]
            })
        
        road_block_tokens = [uuid.uuid4().hex for _ in range(5)]
        road_blocks = []
        for i, token in enumerate(road_block_tokens):
            road_blocks.append({
                "token": token,
                "polygon_token": poly_tokens[i % len(poly_tokens)],
                "from_edge_line_token": line_tokens[i % len(line_tokens)],
                "to_edge_line_token": line_tokens[(i + 1) % len(line_tokens)],
                "road_segment_token": road_segment_tokens[i % len(road_segment_tokens)]
            })
        
        drivable_area_tokens = [uuid.uuid4().hex for _ in range(4)]
        drivable_areas = []
        for i, token in enumerate(drivable_area_tokens):
            drivable_areas.append({
                "token": token,
                "polygon_token": poly_tokens[i % len(poly_tokens)]
            })
        
        connectivity = {}
        for i in range(min(5, len(lane_tokens))):
            lane_token = lane_tokens[i]
            incoming = [lane_tokens[(i-1) % len(lane_tokens)]] if i > 0 else []
            outgoing = [lane_tokens[(i+1) % len(lane_tokens)], lane_tokens[(i+2) % len(lane_tokens)]] if i < len(lane_tokens) - 1 else []
            connectivity[lane_token] = {"incoming": incoming, "outgoing": outgoing}
        
        ped_crossings = []
        for i in range(3):
            ped_crossings.append({
                "token": uuid.uuid4().hex,
                "polygon_token": poly_tokens[i % len(poly_tokens)],
                "road_segment_token": None
            })
        
        walkways = []
        for i in range(2):
            walkways.append({
                "token": uuid.uuid4().hex,
                "polygon_token": poly_tokens[(i+3) % len(poly_tokens)]
            })
        
        traffic_light_tokens = [uuid.uuid4().hex for _ in range(3)]
        traffic_lights = []
        for i, token in enumerate(traffic_light_tokens):
            traffic_lights.append({
                "token": token,
                "line_token": line_tokens[(i+8) % len(line_tokens)],
                "traffic_light_type": "VERTICAL"
            })
        
        stop_lines = []
        for i in range(3):
            stop_lines.append({
                "token": uuid.uuid4().hex,
                "polygon_token": poly_tokens[i % len(poly_tokens)],
                "stop_line_type": "TURN_STOP",
                "ped_crossing_tokens": [],
                "traffic_light_token": traffic_light_tokens[i] if i < len(traffic_light_tokens) else None,
                "road_block_token": None
            })
        
        carpark_areas = []
        for i in range(2):
            carpark_areas.append({
                "token": uuid.uuid4().hex,
                "polygon_token": poly_tokens[(i+4) % len(poly_tokens)],
                "orientation": 2.46383638 + i * 0.5,
                "road_block_token": None
            })
        
        road_dividers = []
        for i in range(2):
            road_dividers.append({
                "token": uuid.uuid4().hex,
                "line_token": line_tokens[(i+5) % len(line_tokens)],
                "road_segment_token": None
            })
        
        lane_connectors = []
        for i in range(3):
            lane_connectors.append({
                "token": uuid.uuid4().hex,
                "polygon_token": poly_tokens[i % len(poly_tokens)]
            })
        
        arcline_path_3 = {}
        for lane_token in lane_tokens[:5]:
            arcline_path_3[lane_token] = [
                {
                    "start_pose": [1000.0 + i * 10, 1500.0 + i * 5, 0.0],
                    "end_pose": [1010.0 + i * 10, 1505.0 + i * 5, 0.1],
                    "shape": "LSL",
                    "radius": 5.0,
                    "segment_length": [2.5, 10.0, 2.5]
                }
                for i in range(3)
            ]
        
        stub_data = {
            "version": "1.3",
            "canvas_edge": [563,400],
            "polygon": polygons,
            "node": nodes,
            "line": lines,
            "lane": lanes,
            "lane_divider": lane_dividers,
            "road_segment": road_segments,
            "road_block": road_blocks,
            "drivable_area": drivable_areas,
            "ped_crossing": ped_crossings,
            "walkway": walkways,
            "stop_line": stop_lines,
            "carpark_area": carpark_areas,
            "road_divider": road_dividers,
            "traffic_light": traffic_lights,
            "lane_connector": lane_connectors,
            "connectivity": connectivity,
            "arcline_path_3": arcline_path_3
        }
        
        save_json_safely(expansion_path, stub_data)
    
    def _write_prediction(self, scenes, samples):
        if not scenes or not samples:
            return

        prediction_path = os.path.join(self._prediction_dir, 'prediction.json')
        prediction_data = load_json_safely(prediction_path, default={})
        
        raw_scene_name = scenes[0].name
        formatted_scene_name = self._format_scene_name(raw_scene_name)
        
        sorted_samples = sorted(samples, key=lambda x: x.timestamp_us)
        
        prediction_id = uuid.uuid4().hex
        predictions = []
        
        for sample in sorted_samples[:20]:
            sample_token = self._token_manager.get_frame_token(sample.temp_frame_id)
            predictions.append(f"{prediction_id}_{sample_token}")
        
        prediction_data[formatted_scene_name] = predictions
        save_json_safely(prediction_path, prediction_data)
    
    def _write_file_manifest(self, data: IntermediateData, base_timestamp):
        new_entries = []
        frame_to_sensor_data = {}
        
        for sd in data.sensor_data:
            if sd.temp_frame_id not in frame_to_sensor_data:
                frame_to_sensor_data[sd.temp_frame_id] = []
            frame_to_sensor_data[sd.temp_frame_id].append(sd)
        
        if base_timestamp is None:
            base_timestamp = 1640995200000000
        
        sorted_frames = sorted(frame_to_sensor_data.items(), key=lambda x: x[1][0].timestamp_us)
        
        current_timestamp = base_timestamp
        
        for frame_id, sensors_in_frame in sorted_frames:
            sequence_name = sensors_in_frame[0].sensor_name.split('_')[0] if sensors_in_frame else data.scenes[0].name if data.scenes else "unknown"
            if data.samples:
                for sample in data.samples:
                    if sample.temp_frame_id == frame_id:
                        sequence_name = sample.scene_name
                        break
            
            manifest_entry = {
                "frame_id": frame_id,
                "sequence": sequence_name,
                "sample_token": self._token_manager.get_frame_token(frame_id),
                "sensors": []
            }
            
            # Sort sensors for consistent ordering
            sensors_in_frame.sort(key=lambda x: x.sensor_name)
            
            for sd in sensors_in_frame:
                if sd.sensor_name.startswith("CAM_"):
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{sd.sensor_name}.jpg"
                    source_file = f"{sequence_name}/camera/{sd.original_filename}"
                else:
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{sd.sensor_name}.pcd.bin"
                    source_file = f"{sequence_name}/lidar/{sd.original_filename}"
                
                manifest_entry["sensors"].append({
                    "channel": sd.sensor_name,
                    "source_file": source_file,
                    "output_file": f"samples/{sd.sensor_name}/{output_filename}"
                })
                
                current_timestamp += 1
            
            new_entries.append(manifest_entry)
        append_to_json_list(os.path.join(self._annot_dir, 'file_manifest.json'), new_entries)
    
    def _write_sample_and_ego_pose(self, samples, ego_poses, base_timestamp):
        sample_path = os.path.join(self._annot_dir, 'sample.json')
        ego_pose_path = os.path.join(self._annot_dir, 'ego_pose.json')
        
        all_samples = load_json_safely(sample_path, default=[])
        all_ego_poses = load_json_safely(ego_pose_path, default=[])
        
        # Use token manager's base timestamp or calculate from existing data
        if base_timestamp is None:
            base_timestamp = 1640995200000000
        
        # Sort input samples and ego poses by their original timestamps to maintain order
        sorted_samples = sorted(samples, key=lambda x: x.timestamp_us)
        sorted_ego_poses = sorted(ego_poses, key=lambda x: x.timestamp_us)
        
        # Assign globally unique timestamps to samples
        current_timestamp = base_timestamp
        for if_sample in sorted_samples:
            all_samples.append({
                "token": self._token_manager.get_frame_token(if_sample.temp_frame_id),
                "timestamp": current_timestamp,
                "scene_token": self._token_manager.get_scene_token()
            })
            current_timestamp += 1
        
        # Assign globally unique timestamps to ego poses
        current_timestamp = base_timestamp
        for if_pose in sorted_ego_poses:
            all_ego_poses.append({
                "token": self._token_manager.get_ego_pose_token(if_pose.temp_frame_id),
                "timestamp": current_timestamp,
                "translation": if_pose.translation,
                "rotation": if_pose.rotation
            })
            current_timestamp += 1
        
        all_samples.sort(key=lambda x: x['timestamp'])
        all_ego_poses.sort(key=lambda x: x['timestamp'])
        
        scene_tokens = {s['scene_token'] for s in all_samples}
        final_samples = []
        
        for scene_token in scene_tokens:
            scene_samples = [s for s in all_samples if s['scene_token'] == scene_token]
            for i, sample in enumerate(scene_samples):
                sample['prev'] = scene_samples[i-1]['token'] if i > 0 else ""
                sample['next'] = scene_samples[i+1]['token'] if i < len(scene_samples) - 1 else ""
            final_samples.extend(scene_samples)
        
        save_json_safely(sample_path, final_samples)
        save_json_safely(ego_pose_path, all_ego_poses)
        
        if samples:
            raw_scene_name = samples[0].scene_name
            formatted_scene_name = self._format_scene_name(raw_scene_name)
            
            new_scene = {
                "token": self._token_manager.get_scene_token(),
                "log_token": self._generated_log_tokens[-1] if self._generated_log_tokens else "",
                "nbr_samples": len(samples),
                "first_sample_token": self._token_manager.get_frame_token(sorted_samples[0].temp_frame_id),
                "last_sample_token": self._token_manager.get_frame_token(sorted_samples[-1].temp_frame_id),
                "name": formatted_scene_name,
                "description": f"Scene {raw_scene_name}"
            }
            append_to_json_list(os.path.join(self._annot_dir, 'scene.json'), [new_scene])
    
    def _write_sample_data(self, sensor_data, sequence_name, base_timestamp):
        sample_data_path = os.path.join(self._annot_dir, 'sample_data.json')
        all_sample_data = load_json_safely(sample_data_path, default=[])
        
        # Use token manager's base timestamp or calculate from existing data
        if base_timestamp is None:
            base_timestamp = 1640995200000000
        
        # Group sensor data by frame
        frame_groups = {}
        for if_data in sensor_data:
            frame_id = if_data.temp_frame_id
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(if_data)
        
        # Sort frames by timestamp to maintain order
        sorted_frames = sorted(frame_groups.items(), key=lambda x: x[1][0].timestamp_us)
        
        # Assign globally unique timestamps
        current_timestamp = base_timestamp
        
        # Process each frame group
        for frame_id, sensors_in_frame in sorted_frames:
            # Sort sensors by name for consistent ordering
            sensors_in_frame.sort(key=lambda x: x.sensor_name)
            
            for if_data in sensors_in_frame:
                is_camera = if_data.sensor_name.startswith("CAM_")
                
                # Filename uses current_timestamp + sensor name for uniqueness
                if is_camera:
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{if_data.sensor_name}.jpg"
                    fileformat = "jpg"
                else:
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{if_data.sensor_name}.pcd.bin"
                    fileformat = "pcd.bin"
                
                all_sample_data.append({
                    "token": uuid.uuid4().hex,
                    "sample_token": self._token_manager.get_frame_token(if_data.temp_frame_id),
                    "ego_pose_token": self._token_manager.get_ego_pose_token(if_data.temp_frame_id),
                    "calibrated_sensor_token": self._token_manager.get_calibration_token(if_data.sensor_name),
                    "filename": f"samples/{if_data.sensor_name}/{output_filename}",
                    "fileformat": fileformat,
                    "width": 1440 if is_camera else 0,
                    "height": 1080 if is_camera else 0,
                    "timestamp": current_timestamp,
                    "is_key_frame": if_data.is_keyframe,
                })
                
                current_timestamp += 1
        
        sensor_groups = {}
        for sd in all_sample_data:
            token = sd['calibrated_sensor_token']
            if token not in sensor_groups:
                sensor_groups[token] = []
            sensor_groups[token].append(sd)
        
        final_sample_data = []
        for sensor_token, sd_list in sensor_groups.items():
            sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
            for i, sd in enumerate(sorted_list):
                sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
            final_sample_data.extend(sorted_list)
        
        save_json_safely(sample_data_path, final_sample_data)
    
    def _write_category(self, instances):
        new_categories = []
        all_category_names = {inst.category_name for inst in instances}
        
        for name, token in self._token_manager.category_tokens.items():
            new_categories.append({
                "token": token,
                "name": name,
                "description": f"{name} category"
            })
        
        for name in all_category_names:
            if name not in self._token_manager.category_tokens:
                token = self._token_manager.get_category_token(name)
                new_categories.append({
                    "token": token,
                    "name": name,
                    "description": f"{name} category"
                })
        
        merge_and_overwrite_json_list(os.path.join(self._annot_dir, 'category.json'), new_categories, key_field='name')
    
    def _write_instance_and_annotation(self, instances, annotations):
        instance_path = os.path.join(self._annot_dir, 'instance.json')
        ann_path = os.path.join(self._annot_dir, 'sample_annotation.json')
        
        all_anns = load_json_safely(ann_path, default=[])
        inst_list = load_json_safely(instance_path, default=[])
        inst_db = {i['token']: i for i in inst_list}
        
        new_anns_by_inst = {}
        for ann in annotations:
            if ann.temp_instance_id not in new_anns_by_inst:
                new_anns_by_inst[ann.temp_instance_id] = []
            new_anns_by_inst[ann.temp_instance_id].append(ann)
        
        inst_name_map = {inst.temp_instance_id: inst.category_name for inst in instances}
        used_category_tokens = {inst['category_token'] for inst in inst_db.values()}
        
        for temp_inst_id, new_anns_list in new_anns_by_inst.items():
            inst_token = self._token_manager.get_instance_token(temp_inst_id)
            new_anns_list.sort(key=lambda x: x.timestamp_us)
            
            last_ann_token = ""
            if inst_token in inst_db:
                last_ann_token = inst_db[inst_token]['last_annotation_token']
            
            generated_tokens = [self._token_manager.generate_annotation_token() for _ in new_anns_list]
            
            for i, if_ann in enumerate(new_anns_list):
                category_name = inst_name_map.get(temp_inst_id, "")
                
                attribute_tokens = []
                if category_name.startswith('vehicle.'):
                    attribute_tokens = [self._token_manager.get_attribute_token("vehicle.moving")]
                elif category_name.startswith('human.') or 'pedestrian' in category_name.lower():
                    attribute_tokens = [self._token_manager.get_attribute_token("pedestrian.moving")]
                
                ann_token = generated_tokens[i]
                prev_token = generated_tokens[i-1] if i > 0 else last_ann_token
                next_token = generated_tokens[i+1] if i < len(generated_tokens) - 1 else ""
                
                all_anns.append({
                    "token": ann_token,
                    "sample_token": self._token_manager.get_frame_token(if_ann.temp_frame_id),
                    "instance_token": inst_token,
                    "attribute_tokens": attribute_tokens,
                    "visibility_token": "4",
                    "translation": if_ann.translation,
                    "size": if_ann.size,
                    "rotation": if_ann.rotation,
                    "prev": prev_token,
                    "next": next_token,
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0
                })
            
            category_token = self._token_manager.get_category_token(inst_name_map.get(temp_inst_id, ""))
            used_category_tokens.add(category_token)
            
            if inst_token not in inst_db:
                inst_db[inst_token] = {
                    "token": inst_token,
                    "category_token": category_token,
                    "nbr_annotations": len(generated_tokens),
                    "first_annotation_token": generated_tokens[0],
                    "last_annotation_token": generated_tokens[-1]
                }
            else:
                inst_db[inst_token]["nbr_annotations"] += len(generated_tokens)
                inst_db[inst_token]["last_annotation_token"] = generated_tokens[-1]
        
        for cat_name, cat_token in self._token_manager.category_tokens.items():
            if cat_token not in used_category_tokens:
                dummy_instance_token = self._token_manager.get_instance_token(f"dummy_{cat_name}")
                if dummy_instance_token not in inst_db:
                    inst_db[dummy_instance_token] = {
                        "token": dummy_instance_token,
                        "category_token": cat_token,
                        "nbr_annotations": 0,
                        "first_annotation_token": dummy_instance_token,
                        "last_annotation_token": dummy_instance_token
                    }
        
        save_json_safely(instance_path, list(inst_db.values()))
        save_json_safely(ann_path, all_anns)
    
    def _process_sensor_files(self, sensor_data, sequence_path, sequence_name, base_timestamp):
        if base_timestamp is None:
            base_timestamp = 1640995200000000
        
        # Group by frame and sort
        frame_groups = {}
        for sd in sensor_data:
            frame_id = sd.temp_frame_id
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(sd)
        
        sorted_frames = sorted(frame_groups.items(), key=lambda x: x[1][0].timestamp_us)
        
        current_timestamp = base_timestamp
        
        for frame_id, sensors_in_frame in sorted_frames:
            sensors_in_frame.sort(key=lambda x: x.sensor_name)
            
            for sd in sensors_in_frame:
                if sd.original_filename.endswith('.pcd'):
                    src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{sd.sensor_name}.pcd.bin"
                    dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_file = os.path.join(dst_folder, output_filename)
                    
                    if not os.path.exists(dst_file):
                        convert_lidar_pcd_to_bin(src_file, dst_file)
                
                elif sd.original_filename.endswith('.feather'):
                    src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{sd.sensor_name}.pcd.bin"
                    dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_file = os.path.join(dst_folder, output_filename)
                    
                    if not os.path.exists(dst_file):
                        convert_lidar_feather_to_bin(src_file, dst_file)
                
                else:  
                    src_file = os.path.join(sequence_path, 'camera', sd.original_filename)
                    output_filename = f"{sequence_name}_frame_{current_timestamp}_{sd.sensor_name}.jpg"
                    dst_folder = os.path.join(self._samples_dir, sd.sensor_name)
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_file = os.path.join(dst_folder, output_filename)
                    
                    if not os.path.exists(dst_file):
                        convert_camera_to_jpg(src_file, dst_file)
                
                current_timestamp += 1
    
    def _duplicate_sweeps(self):
        if os.path.exists(self._sweeps_dir):
            try:
                shutil.rmtree(self._sweeps_dir)
            except Exception:
                return
        try:
            shutil.copytree(self._samples_dir, self._sweeps_dir)
        except Exception:
            pass
