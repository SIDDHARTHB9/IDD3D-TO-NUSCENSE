from abc import ABC, abstractmethod
import os
import json
import threading
import shutil
from queue import Queue
from datetime import datetime
import logging
import uuid
import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversion_state = {
    'active': False,
    'logs': Queue(),
    'progress': 0,
    'total_steps': 0,
    'current_step': 0
}

conversion_lock = threading.Lock()
json_file_lock = threading.Lock()


class LogHandler:
    
    def __init__(self, log_queue):
        self.queue = log_queue
    
    def log(self, message, log_type='info'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")

def append_to_json_list(file_path, new_data_list, log_handler):
 
    if not new_data_list:
        log_handler.log(f"No new data to append to {os.path.basename(file_path)}", 'info')
        return

    with json_file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        log_handler.log(f"Warning: {file_path} is not a list. Overwriting.", 'warning')
                        existing_data = []
            except json.JSONDecodeError:
                log_handler.log(f"Warning: {file_path} is corrupted. Overwriting.", 'warning')
                existing_data = []
        
        final_data = existing_data + new_data_list
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            log_handler.log(f"Appended {len(new_data_list)} items to {os.path.basename(file_path)}. Total items: {len(final_data)}", 'success')
        except Exception as e:
            log_handler.log(f"FATAL: Could not write to {file_path}: {e}", 'error')
            raise

class TokenTimestampManager:
    """
    Manages consistent token generation and timestamp synchronization
    across all converted files for the intermediate format.
    Loads from a registry path to be persistent across runs.
    """
    
    def __init__(self, registry_path=None, base_timestamp=None, frame_rate_hz=10):
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / self.frame_rate_hz)
        
        if base_timestamp is None:
            self.base_timestamp = 1640995200000000
        else:
            self.base_timestamp = base_timestamp
        
        # Local tokens (frame, instance) are ALWAYS new for each run.
        self.frame_tokens = {}
        self.instance_tokens = {}
        self.scene_token = None
        self.ego_pose_tokens = {} # <-- NEW: For linking ego_pose to sample_data
        
        # Global tokens (category, sensor) are loaded from the registry.
        self.category_tokens = {}
        self.sensor_tokens = {}
        self.calibration_tokens = {}
        
        self.registry_path = registry_path
        self.load_registry()
        
    def load_registry(self):
        """
        Loads ONLY global tokens (category, sensor, calibration)
        from the registry path if it exists.
        """
        if self.registry_path and os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                
                self.category_tokens = registry.get('category_tokens', {})
                self.sensor_tokens = registry.get('sensor_tokens', {})
                self.calibration_tokens = registry.get('calibration_tokens', {})
                
                logger.info(f"Loaded {len(self.category_tokens)} global category tokens from registry.")
                logger.info(f"Loaded {len(self.sensor_tokens)} global sensor tokens from registry.")
 
            except Exception as e:
                logger.warning(f"Could not load token registry: {e}")
        else:
            logger.info("No existing token registry found. Starting fresh.")
 
    
    def get_timestamp(self, frame_index):
        return self.base_timestamp + (frame_index * self.frame_interval_us)
    
    def get_frame_token(self, frame_id):
        if frame_id not in self.frame_tokens:
            self.frame_tokens[frame_id] = uuid.uuid4().hex
        return self.frame_tokens[frame_id]
        
    # --- NEW METHOD ---
    def get_ego_pose_token(self, frame_id):
        """Generates a deterministic ego_pose token for this frame_id."""
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
    
    def save_registry(self, output_path):
        """
        Save ONLY global tokens (category, sensor, calibration) to the registry.
        """
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'category_tokens': self.category_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Global token registry saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save token registry: {e}")
 

class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, root: str, sequence: str = None):
        self.root = os.path.abspath(root)
        self.sequence = sequence
    
    @abstractmethod
    def ensure_output_dirs(self):
        pass
    
    @abstractmethod
    def validate(self) -> dict:
        pass


class BaseConverter(ABC):    
    def __init__(self, name: str):
        self.name = name
        self.dry_run = False
    
    @abstractmethod
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        pass


class DatasetConversionPipeline:
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
        self.converters = []
    
    def add_converter(self, converter: BaseConverter):
        self.converters.append(converter)
        return self
    
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        if not self.converters:
            log_handler.log("No converters in pipeline", "warning")
            return
        
        for idx, converter in enumerate(self.converters):
            log_handler.log(
                f"\n[{idx+1}/{len(self.converters)}] Running {converter.name} converter...",
                'info'
            )
            try:
                converter.run(data_loader, log_handler)
                conversion_state['progress'] = ((idx + 1) / len(self.converters)) * 100
            except Exception as e:
                log_handler.log(f"{converter.name} failed: {str(e)}", 'error')
                raise


IDD3D_TO_NUSCENES_CAM_MAP = {
    "cam0": "CAM_FRONT_LEFT",
    "cam1": "CAM_BACK_RIGHT",
    "cam2": "CAM_FRONT_RIGHT",
    "cam3": "CAM_FRONT",
    "cam4": "CAM_BACK_LEFT",
    "cam5": "CAM_BACK"
}
LIDAR_CHANNEL = "LIDAR_TOP"

class IDD3DDataLoader(BaseDataLoader):
    
    def __init__(self, sequence_path: str):
        
        self.seq_base = os.path.abspath(sequence_path)
        root_path = os.path.dirname(self.seq_base)
        sequence_name = os.path.basename(self.seq_base)
        
        super().__init__(root_path, sequence_name)
        
        #  INPUT PATHS 
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'label') 
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json') 
        
        #  UNIFIED OUTPUT PATHS 
        self.output_base = os.path.join(self.root, 'nuScenesFormat')
        self.annot_out = os.path.join(self.output_base, 'anotations')
        self.samples_dir = os.path.join(self.output_base, 'samples')
        self.maps_dir = os.path.join(self.output_base, 'maps') # <-- NEW: For dummy map images
        
        # 'samples' subfolders
        self.converted_lidar = os.path.join(self.samples_dir, LIDAR_CHANNEL)
        
        # Path for the persistent token registry
        self.token_registry_path = os.path.join(self.annot_out, 'token_registry.json')
 
    
    def ensure_output_dirs(self):        
        os.makedirs(self.annot_out, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True) 
        os.makedirs(self.converted_lidar, exist_ok=True)
        
        for cam_channel in IDD3D_TO_NUSCENES_CAM_MAP.values():
            os.makedirs(os.path.join(self.samples_dir, cam_channel), exist_ok=True)
 
    
    def validate(self) -> dict:
        if not os.path.exists(self.seq_base):
            return {'valid': False, 'error': f'Sequence path not found: {self.seq_base}'}
        
        required_dirs = ['lidar', 'label', 'calib']
        missing = []
        for dir_name in required_dirs:
            dir_path = os.path.join(self.seq_base, dir_name)
            if not os.path.exists(dir_path):
                missing.append(dir_name)
        
        if missing:
            return {'valid': False, 'error': f'Missing directories: {", ".join(missing)} in {self.seq_base}'}
        
        if not os.path.exists(self.annot_json):
             return {'valid': False, 'error': f'Missing file: {self.annot_json}'}
 
        lidar_count = len([f for f in os.listdir(self.lidar_dir) 
                          if f.lower().endswith('.pcd')])
        label_count = len([f for f in os.listdir(self.label_dir) 
                          if f.lower().endswith('.json')])
        
        return {
            'valid': True,
            'path': self.seq_base,
            'lidar_files': lidar_count,
            'label_files': label_count
        }
    
    def list_lidar_files(self):
        if not os.path.exists(self.lidar_dir):
            return []
        # Return frame_ids (e.g., '00000', '00200') instead of full paths
        return sorted([os.path.splitext(f)[0] for f in os.listdir(self.lidar_dir) if f.lower().endswith('.pcd')])
    
    def read_annotations(self):
        if not os.path.exists(self.annot_json):
            return {}
        try:
            with open(self.annot_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {self.annot_json}: {e}")
            return {}
 
class IDD3DLidarConverter(BaseConverter):
    
    def __init__(self):
        super().__init__('lidar')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            import numpy as np
            import open3d as o3d
            use_o3d = True
        except ImportError:
            use_o3d = False
            log_handler.log("Warning: open3d not available, creating placeholder files", 'warning')
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping LiDAR conversion.", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        dst_dir = data_loader.converted_lidar 
        src_dir = data_loader.lidar_dir       
        sequence_name = data_loader.sequence  
        
        converted = 0
        placeholders = 0
        overwritten = 0
        
        for frame_id in frame_ids:
            src_file = os.path.join(src_dir, f"{frame_id}.pcd")
            if not os.path.exists(src_file):
                log_handler.log(f"Source file not found: {src_file}", 'warning')
                continue
 
            if frame_id not in annot_data:
                log_handler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            
            dst_filename_base_pcd = os.path.basename(frame_data.get('lidar', f"{frame_id}.pcd"))
            dst_filename_base = os.path.splitext(dst_filename_base_pcd)[0]
            
            # Prepend sequence name to make unique 
            dst_filename = f"{sequence_name}_{dst_filename_base}.pcd.bin"
            
            dst_path = os.path.join(dst_dir, dst_filename)
            
            if os.path.exists(dst_path):
                overwritten += 1
            
            try:
                if use_o3d:
                    pcd = o3d.io.read_point_cloud(src_file)
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                    pts = np.hstack((xyz, intensity))
                    pts.astype(np.float32).tofile(dst_path)
                    converted += 1
                else:
                    open(dst_path, 'wb').close()
                    placeholders += 1
            except Exception as e:
                log_handler.log(f"Error converting {src_file}: {e}", 'error')
                open(dst_path, 'wb').close()
                placeholders += 1
        
        log_handler.log(f"LiDAR conversion complete: {converted} converted, {placeholders} placeholders.", 'success')
        if overwritten > 0:
            log_handler.log(f"Warning: {overwritten} existing LiDAR files were overwritten.", 'warning')
        log_handler.log(f"  Output: {dst_dir}", 'info')
 
class IDD3DCalibConverter(BaseConverter):
    """
    Generate calibration stubs for IDD3D.
    --- UPDATED ---
    Saves to 'anotations' folder.
    Uses new channel names (e.g., 'CAM_FRONT').
    Adds a REAL 3x3 intrinsic matrix for cameras.
    Adds an EMPTY intrinsic list [] for LIDAR_TOP.
    """
    def __init__(self, token_manager):
        super().__init__('calib')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        
        sensors = [LIDAR_CHANNEL] + list(IDD3D_TO_NUSCENES_CAM_MAP.values())
        
        # --- NEW: Define the camera intrinsic matrix (fx, fy, cx, cy) ---
        # Based on 1440x1080 resolution
        # fx=2916, fy=2916, cx=720, cy=540
        intrinsic_matrix = [
            [2916.0, 0.0, 720.0],
            [0.0, 2916.0, 540.0],
            [0.0, 0.0, 1.0]
        ]
        
        calibrated_list = []
        sensors_j = []
        
        for s in sensors:
            sensor_token = self.token_manager.get_sensor_token(s)
            calib_token = self.token_manager.get_calibration_token(s)
            
            # --- NEW: Logic to add correct intrinsics ---
            is_camera = (s != LIDAR_CHANNEL)
            intrinsics = intrinsic_matrix if is_camera else []
            
            entry = {
                "token": calib_token,
                "sensor_token": sensor_token,
                "translation": [0.0, 0.0, 1.8] if not is_camera else [0.0, 0.0, 1.6],
                "rotation": [1.0, 0.0, 0.0, 0.0], # Identity quaternion
                "camera_intrinsic": intrinsics # <-- UPDATED
            }
            calibrated_list.append(entry)
            
            sensors_j.append({
                "token": sensor_token,
                "modality": "lidar" if not is_camera else "camera",
                "channel": s,
            })
        
        calib_path = os.path.join(data_loader.annot_out, 'calibrated_sensor.json')
        sensor_path = os.path.join(data_loader.annot_out, 'sensor.json')
 
        with open(calib_path, 'w') as f:
            json.dump(calibrated_list, f, indent=2)
        with open(sensor_path, 'w') as f:
            json.dump(sensors_j, f, indent=2)
        
        log_handler.log("Calibration stubs created/overwritten with correct intrinsics", 'success')
        log_handler.log(f"  Output: {data_loader.annot_out}", 'info')
 

class IDD3DLogConverter(BaseConverter):
 
    def __init__(self, token_manager, sequence_name):
        super().__init__('log')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'log.json')
        
        logfile = f"{data_loader.sequence}-{datetime.now().strftime('%Y-%m-%d')}"
        log_token = self.token_manager.get_category_token(f"log_{logfile}")
        
        new_log_entry = {
            "token": log_token,
            "logfile": logfile,
            "vehicle": "idd3d_stub_vehicle",
            "date_captured": datetime.now().strftime('%Y-%m-%d'),
            "location": "Hyderabad"
        }
        
        with json_file_lock:
            logs = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        logs = json.load(f)
                        if not isinstance(logs, list): logs = []
                except Exception as e:
                    log_handler.log(f"Could not read existing log.json: {e}", 'warning')
                    logs = []
            
            found = False
            for i, log in enumerate(logs):
                if log.get('logfile') == logfile:
                    logs[i] = new_log_entry
                    found = True
                    log_handler.log(f"Updating existing log entry for {logfile}", 'info')
                    break
            
            if not found:
                logs.append(new_log_entry)
                log_handler.log(f"Adding new log entry for {logfile}", 'info')

            try:
                with open(out_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                log_handler.log(f"Unified log.json updated. Total logs: {len(logs)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to log.json: {e}", 'error')
                raise

class IDD3DCameraConverter(BaseConverter):

    
    def __init__(self):
        super().__init__("camera")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        try:
            from PIL import Image
            usepil = True
        except ImportError:
            usepil = False
            loghandler.log("PIL/Pillow not available, skipping camera conversion", "warning")
            return
        
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found, skipping Camera conversion.", 'warning')
            return
            
        frame_ids = sorted(annot_data.keys())
        
        src_camera_dir = os.path.join(dataloader.seq_base, "camera")
        dst_samples_dir = dataloader.samples_dir
        sequence_name = dataloader.sequence # e.g., "idd3d_seq10"
        
        if not os.path.exists(src_camera_dir):
            loghandler.log("No camera directory found", "warning")
            return
        
        converted = 0
        errors = 0
        overwritten = 0
 
        for frame_id in frame_ids:
            if frame_id not in annot_data:
                loghandler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items(): # e.g., "cam0", "CAM_FRONT_LEFT"
                
                src_path = os.path.join(src_camera_dir, idd_cam, f"{frame_id}.png")
                if not os.path.exists(src_path):
                    loghandler.log(f"Source file not found: {src_path}", 'warning')
                    continue
                
                dst_filename_png = frame_data.get(idd_cam)
                if not dst_filename_png:
                    loghandler.log(f"No filename for {idd_cam} in frame {frame_id}", 'warning')
                    continue
                
                dst_filename_base = os.path.splitext(os.path.basename(dst_filename_png))[0]
                
                # --- NEW: Prepend sequence name to make unique ---
                dst_filename = f"{sequence_name}_{dst_filename_base}.jpg"
 
                dst_path = os.path.join(dst_samples_dir, nu_cam, dst_filename)
                
                if os.path.exists(dst_path):
                    overwritten += 1
                
                try:
                    if usepil:
                        img = Image.open(src_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dst_path, 'JPEG', quality=95)
                        converted += 1
                except Exception as e:
                    errors += 1
                    loghandler.log(f"Error converting {src_path} to {dst_path}: {str(e)}", "error")
        
        loghandler.log(f"Camera conversion complete: {converted} images converted, {errors} errors", "success")
        if overwritten > 0:
            loghandler.log(f"Warning: {overwritten} existing image files were overwritten.", 'warning')
        loghandler.log(f"  Output: {dst_samples_dir}", 'info')
 
class IDD3DEgoPoseConverter(BaseConverter):
    """
    Generates stubbed ego_pose.json entries for this sequence.
    --- UPDATED ---
    Uses a deterministic token from TokenTimestampManager to link to sample_data.
    """
    def __init__(self, token_manager):
        super().__init__('ego_pose')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping ego_pose", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        new_poses = []
        
        for i, frame_id in enumerate(frame_ids):
            timestamp = self.token_manager.get_timestamp(i)
            # --- UPDATED: Get deterministic token ---
            ego_pose_token = self.token_manager.get_ego_pose_token(frame_id)
            
            new_poses.append({
                "token": ego_pose_token, # <-- USE THE DETERMINISTIC TOKEN
                "timestamp": timestamp,
                "translation": [0.0, 0.0, 0.0], # Stubbed
                "rotation": [1.0, 0.0, 0.0, 0.0]  # Stubbed (identity quaternion)
            })
            
        out_path = os.path.join(data_loader.annot_out, 'ego_pose.json')
        append_to_json_list(out_path, new_poses, log_handler)
 


class IDD3DMapConverter(BaseConverter):
 
    def __init__(self, token_manager):
        super().__init__('map')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'map.json')
        log_path = os.path.join(data_loader.annot_out, 'log.json')
        
        
        maps_dir = data_loader.maps_dir
        os.makedirs(maps_dir, exist_ok=True)
 
        location = "Hyderabad"
        map_token = self.token_manager.get_category_token(f"map_{location}")
        
        with json_file_lock:
            all_log_tokens_for_location = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        all_logs = json.load(f)
                        if isinstance(all_logs, list):
                            for log in all_logs:
                                if log.get('location') == location and log.get('token'):
                                    all_log_tokens_for_location.append(log.get('token'))
                except Exception as e:
                    log_handler.log(f"Could not read log.json to link maps: {e}", 'warning')
            
            new_map_entry = {
                "token": map_token,
                "log_tokens": all_log_tokens_for_location,
                "category": "semantic_prior",
                "filename": f"maps/{location.lower()}.png",
            }
            
            maps = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        maps = json.load(f)
                        if not isinstance(maps, list): maps = []
                except Exception as e:
                    log_handler.log(f"Could not read existing map.json: {e}", 'warning')
                    maps = []
            
            found = False
            for i, map_entry in enumerate(maps):
                if map_entry.get('token') == map_token:
                    maps[i] = new_map_entry
                    found = True
                    log_handler.log(f"Updating existing map entry for {location}", 'info')
                    break
            
            if not found:
                maps.append(new_map_entry)
                log_handler.log(f"Adding new map entry for {location}", 'info')
 
            try:
                with open(out_path, 'w') as f:
                    json.dump(maps, f, indent=2)
                log_handler.log(f"Unified map.json updated. Total maps: {len(maps)}", 'success')
                log_handler.log(f"  Map '{location}' now linked to {len(all_log_tokens_for_location)} logs.", 'info')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to map.json: {e}", 'error')
                raise
 

class IDD3DSceneConverter(BaseConverter):

    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('scene')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        scene_token = self.token_manager.get_scene_token()
        logfile = f"{data_loader.sequence}-{datetime.now().strftime('%Y-%m-%d')}"
        log_token = self.token_manager.get_category_token(f"log_{logfile}")
        
        first_sample_token = self.token_manager.get_frame_token(frame_ids[0])
        last_sample_token = self.token_manager.get_frame_token(frame_ids[-1])
        
        seq_num_str = self.sequence_name.split('_')[-1].replace('seq', '')
        formatted_num = seq_num_str.zfill(3)
        new_scene_name = f"scene-{formatted_num}"

        current_scene = {
            "token": scene_token,
            "log_token": log_token, 
            "nbr_samples": len(frame_ids),
            "first_sample_token": first_sample_token,
            "last_sample_token": last_sample_token,
            "name": new_scene_name, 
            "description": f"IDD3D sequence {self.sequence_name}"
        }
        
        out_path = os.path.join(data_loader.annot_out, 'scene.json')
        
        with json_file_lock:
            scenes = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        scenes = json.load(f)
                        if not isinstance(scenes, list): scenes = []
                except Exception as e:
                    log_handler.log(f"Could not read shared scene.json: {e}", 'warning')
                    scenes = []
            
            found = False
            for i, scene in enumerate(scenes):
                if scene.get('name') == new_scene_name:
                    log_handler.log(f"Updating existing scene: {new_scene_name}", 'info')
                    scenes[i] = current_scene
                    found = True
                    break
            
            if not found:
                log_handler.log(f"Adding new scene: {new_scene_name}", 'info')
                scenes.append(current_scene)
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(scenes, f, indent=2)
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to shared scene.json: {e}", 'error')
                raise

        log_handler.log(f"Unified scene.json updated ({len(scenes)} total scenes)", 'success')
        log_handler.log(f"  Output: {out_path}", 'info')

class IDD3DSampleDataConverter(BaseConverter):
    """
    Generate sample_data.json. Appends to existing file.
    --- UPDATED ---
    Adds "ego_pose_token" to link to the ego_pose.json entry.
    """
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample_data')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        new_sample_data_list = []
        
        for i, frame_id in enumerate(frame_ids):
            if frame_id not in annot_data:
                log_handler.log(f"Skipping frame {frame_id}: not found in annot_data.json", 'warning')
                continue
            frame_data = annot_data[frame_id]
            sample_token = self.token_manager.get_frame_token(frame_id)
            timestamp = self.token_manager.get_timestamp(i)
            # --- NEW: Get the deterministic ego_pose token ---
            ego_pose_token = self.token_manager.get_ego_pose_token(frame_id)
            
            # --- LiDAR Data ---
            lidar_filename_raw = frame_data.get('lidar', f'{frame_id}.pcd.bin')
            lidar_filename_base = os.path.splitext(os.path.basename(lidar_filename_raw))[0]
            lidar_filename = f"{data_loader.sequence}_{lidar_filename_base}.pcd.bin"
            
            new_sample_data_list.append({
                "token": uuid.uuid4().hex, 
                "sample_token": sample_token,
                "ego_pose_token": ego_pose_token, # <-- ADDED TOKEN
                "calibrated_sensor_token": self.token_manager.get_calibration_token(LIDAR_CHANNEL),
                "filename": f"samples/{LIDAR_CHANNEL}/{lidar_filename}", 
                "fileformat": "pcd.bin", "width": 0, "height": 0, "timestamp": timestamp,
                "is_key_frame": True, "next": "", "prev": ""
            })
            
            # --- Camera Data ---
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items(): 
                
                cam_filename_raw = frame_data.get(idd_cam, f'{frame_id}.jpg')
                cam_filename_base = os.path.splitext(os.path.basename(cam_filename_raw))[0]
                cam_filename = f"{data_loader.sequence}_{cam_filename_base}.jpg"
                
                new_sample_data_list.append({
                    "token": uuid.uuid4().hex,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token, # <-- ADDED TOKEN
                    "calibrated_sensor_token": self.token_manager.get_calibration_token(nu_cam),
                    "filename": f"samples/{nu_cam}/{cam_filename}", 
                    "fileformat": "jpg", 
                    "width": 1440,
                    "height": 1080, 
                    "timestamp": timestamp,
                    "is_key_frame": True, "next": "", "prev": ""
                })
        
        # --- PREV/NEXT LINKING LOGIC ---
        out_path = os.path.join(data_loader.annot_out, 'sample_data.json')
        
        with json_file_lock:
            existing_data = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list): existing_data = []
                except Exception as e:
                    log_handler.log(f"Warning: could not read existing sample_data.json: {e}", "warning")
                    existing_data = []
 
            all_data = existing_data + new_sample_data_list
            
            sensor_groups = {}
            for sd in all_data:
                token = sd['calibrated_sensor_token']
                if token not in sensor_groups: sensor_groups[token] = []
                sensor_groups[token].append(sd)
 
            log_handler.log(f"Linking prev/next tokens for {len(all_data)} total sample_data entries...", 'info')
            
            final_linked_list = []
            
            for sensor_token, sd_list in sensor_groups.items():
                sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
                for i, sd in enumerate(sorted_list):
                    sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                    sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
                
                final_linked_list.extend(sorted_list)
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(final_linked_list, f, indent=2)
                log_handler.log(f"Appended {len(new_sample_data_list)} items to sample_data.json. Total items: {len(final_linked_list)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to {out_path}: {e}", 'error')
                raise
 

class IDD3DCategoryConverter(BaseConverter):
    """
    Generate category.json.
    --- UPDATED ---
    Reads the FINAL instance.json and token_registry.json to build a
    perfectly synchronized category.json file. This OVERWRITES.
    """
    
    def __init__(self, token_manager):
        super().__init__('category')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        instance_path = os.path.join(data_loader.annot_out, 'instance.json')
        registry_path = data_loader.token_registry_path
        out_path = os.path.join(data_loader.annot_out, 'category.json')
 
        if not os.path.exists(instance_path):
            log_handler.log("instance.json not found, skipping category creation.", 'warning')
            return
        if not os.path.exists(registry_path):
            log_handler.log("token_registry.json not found, skipping category creation.", 'warning')
            return
 
        with json_file_lock:
            try:
                # 1. Read the token registry to get token->name mappings
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                
                # Invert the registry to be {token: name}
                # We only care about category tokens
                token_to_name = {
                    tok: name for name, tok in registry.get('category_tokens', {}).items()
                }
                
                # 2. Read the entire instance.json to find all *used* tokens
                with open(instance_path, 'r') as f:
                    all_instances = json.load(f)
                
                used_category_tokens = set()
                if isinstance(all_instances, list):
                    for inst in all_instances:
                        if 'category_token' in inst:
                            used_category_tokens.add(inst['category_token'])
                
                log_handler.log(f"Found {len(used_category_tokens)} unique category tokens in use in instance.json", 'info')
 
                # 3. Build the new category.json
                final_categories = []
                for token in used_category_tokens:
                    if token in token_to_name:
                        name = token_to_name[token]
                        final_categories.append({
                            "token": token,
                            "name": name,
                            "description": f"{name} category" # Simple description
                        })
                    else:
                        log_handler.log(f"CRITICAL ERROR: Token {token} from instance.json not found in registry!", 'error')
                
                # 4. Overwrite category.json
                with open(out_path, 'w') as f:
                    json.dump(final_categories, f, indent=2)
                
                log_handler.log(f"Successfully overwrote category.json with {len(final_categories)} in-use categories.", 'success')
                
            except Exception as e:
                log_handler.log(f"FATAL: Could not rebuild category.json: {e}", 'error')
                import traceback
                log_handler.log(traceback.format_exc(), 'error')
                raise
 

class IDD3DSampleConverter(BaseConverter):

    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader, log_handler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        scene_token = self.token_manager.get_scene_token()
        samples = []
        
        for i, frame_id in enumerate(frame_ids):
            token = self.token_manager.get_frame_token(frame_id)
            timestamp = self.token_manager.get_timestamp(i)
            prev = self.token_manager.get_frame_token(frame_ids[i-1]) if i > 0 else ""
            next_token = self.token_manager.get_frame_token(frame_ids[i+1]) if i < len(frame_ids)-1 else ""
            
            sample = {
                "token": token,
                "timestamp": timestamp,
                "prev": prev,
                "next": next_token,
                "scene_token": scene_token
            }
            samples.append(sample)
        
        out_path = os.path.join(data_loader.annot_out, 'sample.json')
        append_to_json_list(out_path, samples, log_handler)

class IDD3DAttributeConverter(BaseConverter):

    def __init__(self, token_manager):
        super().__init__('attribute')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'attribute.json')
        
        # --- NEW: Define the specific attributes we will link ---
        attributes = [
            {"name": "vehicle.moving", "description": "Vehicle is moving (default stub)"},
            {"name": "pedestrian.moving", "description": "Pedestrian is moving (default stub)"},
            {"name": "static.on_tarmac", "description": "Static object is on tarmac (default stub)"}
        ]
        
        populated_attributes = []
        for attr in attributes:
            populated_attributes.append({
                "token": self.token_manager.get_category_token(attr["name"]), # Re-use category token logic
                "name": attr["name"],
                "description": attr["description"]
            })
 
        with json_file_lock:
            try:
                with open(out_path, 'w') as f:
                    json.dump(populated_attributes, f, indent=2)
                log_handler.log(f"Created/Overwritten stub attribute.json with {len(populated_attributes)} entries", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to attribute.json: {e}", 'error')
                raise
 
 
class IDD3DSampleAnnotationConverter(BaseConverter):
    """
    Generate sample_annotation.json. Appends to existing file.
    --- UPDATED ---
    Adds a linked attribute_token and a stubbed visibility_token.
    """
    
    def __init__(self, token_manager, sequence_name: str = 'seq'):
        super().__init__('sample_annotation')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
        
        # Get the tokens for our default attributes
        self.attr_vehicle_moving = self.token_manager.get_category_token("vehicle.moving")
        self.attr_pedestrian_moving = self.token_manager.get_category_token("pedestrian.moving")
        self.attr_static = self.token_manager.get_category_token("static.on_tarmac")
        
        # --- NEW: Get the token for 80-100% visibility ---
        self.vis_full = self.token_manager.get_category_token("v4-0")
        
        self.idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car', 'Truck': 'vehicle.truck', 'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle', 'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle', 'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult', 'Rider': 'human.pedestrian.rider',
            'Animal': 'animal', 
            'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign', 
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other', 'Misc': 'movable_object.debris'
        }
 
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        sample_annotations = []
        object_instances_in_this_run = {}
        
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get("obj_id")
                    obj_type = obj.get("obj_type") # e.g., "Car"
                    if not obj_id or not obj_type:
                        continue
                    
                    instance_token = self.token_manager.get_instance_token(obj_id)
                    
                    if obj_id not in object_instances_in_this_run:
                        object_instances_in_this_run[obj_id] = {'annotations': []}
                    
                    ann_token = self.token_manager.generate_annotation_token()
                    frame_token = self.token_manager.get_frame_token(frame_id)
                    
                    # Smart Attribute Linking
                    category_name = self.idd3d_to_nuscenes_categories.get(obj_type, 'Misc')
                    attribute_tokens = [] # Default to null vector
                    
                    if category_name.startswith('vehicle.'):
                        attribute_tokens = [self.attr_vehicle_moving]
                    elif category_name.startswith('human.'):
                        attribute_tokens = [self.attr_pedestrian_moving]
                    elif category_name.startswith('static_object.'):
                        attribute_tokens = [self.attr_static]
 
                    psr = obj.get("psr", {})
                    pos = psr.get("position", {})
                    rot = psr.get("rotation", {})
                    scl = psr.get("scale", {})
                    
                    translation = [pos.get("x",0), pos.get("y",0), pos.get("z",0)]
                    size = [scl.get("x",1), scl.get("y",1), scl.get("z",1)]
                    rotation_quat = [rot.get("x",0), rot.get("y",0), rot.get("z",0), 1.0]
                    
                    annotation = {
                        "token": ann_token,
                        "sample_token": frame_token,
                        "instance_token": instance_token,
                        "attribute_tokens": attribute_tokens,
                        "visibility_token": self.vis_full, # <-- ADDED THIS LINE
                        "translation": translation, "size": size, "rotation": rotation_quat,
                        "prev": "", "next": "",
                        "num_lidar_pts": 0, "num_radar_pts": 0
                    }
                    object_instances_in_this_run[obj_id]['annotations'].append(annotation)
                    
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        for obj_id, instance_data in object_instances_in_this_run.items():
            annotations = instance_data['annotations']
            for i, ann in enumerate(annotations):
                if i > 0:
                    ann['prev'] = annotations[i-1]['token']
                if i < len(annotations) - 1:
                    ann['next'] = annotations[i+1]['token']
                sample_annotations.append(ann)
        
        out_path = os.path.join(data_loader.annot_out, 'sample_annotation.json')
        append_to_json_list(out_path, sample_annotations, log_handler)
 
 

class IDD3DInstanceConverter(BaseConverter):
    
    def __init__(self, token_manager):
        super().__init__('instance')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car', 'Truck': 'vehicle.truck', 'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle', 'MotorcyleRider': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle', 'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult', 'Rider': 'human.pedestrian.rider',
            'Animal': 'animal', 'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign', 'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other', 'Misc': 'movable_object.debris'
        }
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        instance_tracker = {} 
        
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get("obj_id")
                    obj_type = obj.get("obj_type")
                    if not obj_id or not obj_type:
                        continue
                    
                    if obj_id not in instance_tracker:
                        instance_token = self.token_manager.get_instance_token(obj_id)
                        category_name = idd3d_to_nuscenes_categories.get(obj_type, f'movable_object.{obj_type.lower()}')
                        category_token = self.token_manager.get_category_token(category_name)
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'obj_type': obj_type,
                            'first_ann_token': self.token_manager.generate_annotation_token(),
                            'last_ann_token': self.token_manager.generate_annotation_token()
                        }
                    else:
                        instance_tracker[obj_id]['last_ann_token'] = self.token_manager.generate_annotation_token()
                        
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        new_instances = []
        for obj_id, data in instance_tracker.items():
            new_instances.append({
                "token": data['instance_token'],
                "category_token": data['category_token'],
                "nbr_annotations": None, 
                "first_annotation_token": data['first_ann_token'],
                "last_annotation_token": data['last_ann_token']
            })
        
        out_path = os.path.join(data_loader.annot_out, 'instance.json')
        
        with json_file_lock:
            existing_instances = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        insts = json.load(f)
                        existing_instances = {inst['token']: inst for inst in insts}
                except Exception as e:
                    log_handler.log(f"Could not read existing instance.json: {e}", 'warning')

            updated_count = 0
            new_count = 0
            for inst in new_instances:
                if inst['token'] in existing_instances:
                    existing_instances[inst['token']]['last_annotation_token'] = inst['last_annotation_token']
                    updated_count += 1
                else:
                    existing_instances[inst['token']] = inst
                    new_count += 1
            
            final_instances = list(existing_instances.values())
            try:
                with open(out_path, 'w') as f:
                    json.dump(final_instances, f, indent=2)
                log_handler.log(f"Instance file merged. Added {new_count}, updated {updated_count}. Total: {len(final_instances)}", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to instance.json: {e}", 'error')
                raise

# --- NEW STUB CONVERTER ---
class IDD3DVisibilityConverter(BaseConverter):
    """
    Generates a stubbed, populated visibility.json file.
    This file is static and will be overwritten on each run.
    """
    def __init__(self, token_manager):
        super().__init__('visibility')
        self.token_manager = token_manager
        
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        out_path = os.path.join(data_loader.annot_out, 'visibility.json')
        
        # nuScenes standard visibility levels
        vis_levels = [
            {"level": "v1-0", "description": "visibility 0-40%"},
            {"level": "v2-0", "description": "visibility 40-60%"},
            {"level": "v3-0", "description": "visibility 60-80%"},
            {"level": "v4-0", "description": "visibility 80-100%"}
        ]
        
        populated_visibility = []
        for vis in vis_levels:
            populated_visibility.append({
                "token": self.token_manager.get_category_token(vis["level"]), # Re-use category token logic
                "level": vis["level"],
                "description": vis["description"]
            })
 
        with json_file_lock:
            try:
                # Note: nuScenes visibility.json is a LIST, not a dictionary
                with open(out_path, 'w') as f:
                    json.dump(populated_visibility, f, indent=2)
                log_handler.log(f"Created/Overwritten stub visibility.json with {len(populated_visibility)} entries", 'success')
            except Exception as e:
                log_handler.log(f"FATAL: Could not write to visibility.json: {e}", 'error')
                raise
 


class IDD3DFileManifestConverter(BaseConverter):

    def __init__(self, token_manager, sequence_name):
        super().__init__('manifest')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found, skipping manifest", 'warning')
            return
            
        frame_ids = sorted(annot_data.keys())
        new_manifest_entries = []
        sequence_name = data_loader.sequence # e.g., "idd3d_seq10"
 
        for i, frame_id in enumerate(frame_ids):
            if frame_id not in annot_data:
                continue
            frame_data = annot_data[frame_id]
            
            bag_id = frame_data.get("bag_id", "unknown_bag_id")
            
            manifest_entry = {
                "frame_id": frame_id,
                "sequence": self.sequence_name,
                "bag_id": bag_id,
                "sample_token": self.token_manager.get_frame_token(frame_id),
                "sensors": []
            }
 
            # --- LiDAR Entry ---
            src_lidar_name = f"{frame_id}.pcd"
            dst_lidar_raw = frame_data.get('lidar', f"{frame_id}.pcd.bin")
            dst_lidar_base = os.path.splitext(os.path.basename(dst_lidar_raw))[0]
            dst_lidar_name = f"{sequence_name}_{dst_lidar_base}.pcd.bin"
            
            manifest_entry["sensors"].append({
                "channel": LIDAR_CHANNEL,
                "source_file": f"{self.sequence_name}/lidar/{src_lidar_name}",
                "output_file": f"samples/{LIDAR_CHANNEL}/{dst_lidar_name}"
            })
            
            # --- Camera Entries ---
            for idd_cam, nu_cam in IDD3D_TO_NUSCENES_CAM_MAP.items():
                src_cam_name = f"{frame_id}.png"
                dst_cam_raw = frame_data.get(idd_cam, f"{frame_id}.jpg")
                dst_cam_base = os.path.splitext(os.path.basename(dst_cam_raw))[0]
                dst_cam_name_jpg = f"{sequence_name}_{dst_cam_base}.jpg"
                
                manifest_entry["sensors"].append({
                    "channel": nu_cam,
                    "source_file": f"{self.sequence_name}/camera/{idd_cam}/{src_cam_name}",
                    "output_file": f"samples/{nu_cam}/{dst_cam_name_jpg}"
                })
            
            new_manifest_entries.append(manifest_entry)
        
        out_path = os.path.join(data_loader.annot_out, 'file_manifest.json')
        append_to_json_list(out_path, new_manifest_entries, log_handler)
 
# --- NEW CONVERTER CLASS ---
class IDD3DDuplicateSweepsConverter(BaseConverter):
    """
    Duplicates the 'samples' directory to 'sweeps' for nuScenes compliance.
    This is a full copy and will overwrite the old sweeps directory.
    """
    def __init__(self):
        super().__init__('duplicate_sweeps')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        source_dir = data_loader.samples_dir
        dest_dir = os.path.join(data_loader.output_base, 'sweeps')
 
        if not os.path.exists(source_dir):
            log_handler.log(f"Source 'samples' directory not found. Skipping sweeps copy.", 'warning')
            return
 
        # Clean destination directory for a fresh copy
        if os.path.exists(dest_dir):
            try:
                shutil.rmtree(dest_dir)
                log_handler.log(f"Removed old 'sweeps' directory.", 'info')
            except Exception as e:
                log_handler.log(f"Warning: Could not remove old 'sweeps' directory: {e}", 'warning')
 
        # Copy the entire 'samples' directory to 'sweeps'
        try:
            shutil.copytree(source_dir, dest_dir)
            log_handler.log(f"Successfully duplicated 'samples' to 'sweeps' directory.", 'success')
        except Exception as e:
            log_handler.log(f"FATAL: Could not copy 'samples' to 'sweeps': {e}", 'error')
            import traceback
            log_handler.log(traceback.format_exc(), 'error')
            raise
 
 
class IDD3DTokenRegistrySaver(BaseConverter):
    
    def __init__(self, token_manager, registry_path):
        super().__init__('save_registry')
        self.token_manager = token_manager
        self.registry_path = registry_path
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            self.token_manager.save_registry(self.registry_path)
            log_handler.log(f"Token registry saved to: {self.registry_path}", 'success')
        except Exception as e:
            log_handler.log(f"Warning: Could not save token registry: {str(e)}", 'warning')
 
 

class IDD3DTokenRegistrySaver(BaseConverter):

    
    def __init__(self, token_manager, registry_path):
        super().__init__('save_registry')
        self.token_manager = token_manager
        self.registry_path = registry_path
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            self.token_manager.save_registry(self.registry_path)
            log_handler.log(f"Token registry saved to: {self.registry_path}", 'success')
        except Exception as e:
            log_handler.log(f"Warning: Could not save token registry: {str(e)}", 'warning')



class ConverterRegistry:
    
    _conversions = {}
    
    @classmethod
    def register(cls, source: str, target: str, pipeline_builder):
        key = (source, target)
        cls._conversions[key] = pipeline_builder
    
    @classmethod
    def get_pipeline(cls, source: str, target: str, config: dict):
        key = (source, target)
        if key not in cls._conversions:
            raise ValueError(f"No conversion registered for {source} -> {target}")
        pipeline_builder = cls._conversions[key]
        return pipeline_builder(config)
    
    @classmethod
    def get_available_conversions(cls):
        return [{'source': s, 'target': t} for s, t in cls._conversions.keys()]

def build_idd3d_to_nuscenes_pipeline(config: dict) -> DatasetConversionPipeline:
    """Build conversion pipeline for IDD3D -> nuScenes with TokenTimestampManager"""
    pipeline = DatasetConversionPipeline('idd3d', 'nuscenes')
    
    conversions = config.get('conversions', {})
    sequence_name = config.get('sequence_id', 'idd3d_seq10')
    root_path = config.get('root_path')
    
    if not root_path:
        raise ValueError("root_path is required in config to build pipeline")
        
    registry_path = os.path.join(root_path, 'nuScenesFormat', 'anotations', 'token_registry.json')
    
    # --- Calculate new base_timestamp ---
    last_timestamp = 1640995200000000 # Default start
    sample_json_path = os.path.join(root_path, 'nuScenesFormat', 'anotations', 'sample.json')
    
    with json_file_lock:
        if os.path.exists(sample_json_path):
            try:
                with open(sample_json_path, 'r') as f:
                    samples = json.load(f)
                    if samples and isinstance(samples, list):
                        last_timestamp = samples[-1].get('timestamp', last_timestamp)
            except Exception:
                pass 
                
    new_base_timestamp = last_timestamp + 20_000_000 # 20-second gap
    
    token_manager = TokenTimestampManager(
        frame_rate_hz=10, 
        registry_path=registry_path,
        base_timestamp=new_base_timestamp
    )
    
    # ---
    
    # PHASE 1: Data Conversion
    if conversions.get('lidar', False):
        pipeline.add_converter(IDD3DLidarConverter())
    if conversions.get('camera', False):
        pipeline.add_converter(IDD3DCameraConverter())
    if conversions.get('calib', False):
        pipeline.add_converter(IDD3DCalibConverter(token_manager))
    
    # PHASE 2: Taxonomy & Stubs (Merges)
    if conversions.get('log', False):
        pipeline.add_converter(IDD3DLogConverter(token_manager, sequence_name))
    if conversions.get('map', False):
        pipeline.add_converter(IDD3DMapConverter(token_manager))
    
    # --- NEW: Add attribute.json and visibility.json stubs ---
    # These must run before SampleAnnotationConverter to populate tokens
    pipeline.add_converter(IDD3DAttributeConverter(token_manager))
    pipeline.add_converter(IDD3DVisibilityConverter(token_manager)) # <-- ADDED
    
    if conversions.get('category', False):
        pipeline.add_converter(IDD3DCategoryConverter(token_manager))
    if conversions.get('scene', False):
        pipeline.add_converter(IDD3DSceneConverter(token_manager, sequence_name))
 
    # PHASE 3: Core Data (Appends)
    if conversions.get('sample', False):
        pipeline.add_converter(IDD3DSampleConverter(token_manager, sequence_name))
    if conversions.get('sample_data', False):
        pipeline.add_converter(IDD3DSampleDataConverter(token_manager, sequence_name))
    if conversions.get('ego_pose', False):
        pipeline.add_converter(IDD3DEgoPoseConverter(token_manager))
 
    # PHASE 4: Annotations (Appends)
    if conversions.get('instance', False):
        pipeline.add_converter(IDD3DInstanceConverter(token_manager))
    if conversions.get('sample_annotation', False):
        pipeline.add_converter(IDD3DSampleAnnotationConverter(token_manager, sequence_name))
    
    # PHASE 5: Post-processing
    pipeline.add_converter(IDD3DTokenRegistrySaver(token_manager, registry_path))
    pipeline.add_converter(IDD3DFileManifestConverter(token_manager, sequence_name))
 
    # CategoryConverter must run *after* InstanceConverter and *after* RegistrySaver
    # This logic is no longer correct. Category must run *before* instance.
    # Let's re-order the pipeline logic.
    
    # --- RE-ORDERING ---
    # The converters must be added in a specific logical order for the token
    # dependencies to work.
    
    # Clear pipeline to rebuild with correct order
    pipeline.converters = []
 
    # PHASE 1: Data Conversion (Physical files)
    if conversions.get('lidar', False):
        pipeline.add_converter(IDD3DLidarConverter())
    if conversions.get('camera', False):
        pipeline.add_converter(IDD3DCameraConverter())
    if conversions.get('calib', False):
        pipeline.add_converter(IDD3DCalibConverter(token_manager))
 
    # PHASE 2: Create "Dictionaries" (Global Tokens)
    # These must run *before* sample_annotation
    pipeline.add_converter(IDD3DAttributeConverter(token_manager))
    pipeline.add_converter(IDD3DVisibilityConverter(token_manager))
    # Category *must* run before Instance/SampleAnnotation
    if conversions.get('category', False):
        pipeline.add_converter(IDD3DCategoryConverter(token_manager))
        
    # PHASE 3: Create "Logs"
    if conversions.get('log', False):
        pipeline.add_converter(IDD3DLogConverter(token_manager, sequence_name))
    if conversions.get('map', False):
        pipeline.add_converter(IDD3DMapConverter(token_manager))
        
    # PHASE 4: Create "Scenes" and "Timestamps"
    if conversions.get('scene', False):
        pipeline.add_converter(IDD3DSceneConverter(token_manager, sequence_name))
    if conversions.get('sample', False):
        pipeline.add_converter(IDD3DSampleConverter(token_manager, sequence_name))
    if conversions.get('ego_pose', False):
        pipeline.add_converter(IDD3DEgoPoseConverter(token_manager))
        
    # PHASE 5: Link everything
    # SampleData must run *after* Sample and EgoPose
    if conversions.get('sample_data', False):
        pipeline.add_converter(IDD3DSampleDataConverter(token_manager, sequence_name))
    # Instance/SampleAnnotation must run *after* Categories and Samples
    if conversions.get('instance', False):
        pipeline.add_converter(IDD3DInstanceConverter(token_manager))
    if conversions.get('sample_annotation', False):
        pipeline.add_converter(IDD3DSampleAnnotationConverter(token_manager, sequence_name))
    pipeline.add_converter(IDD3DDuplicateSweepsConverter()) 
    # PHASE 6: Final bookkeeping
    pipeline.add_converter(IDD3DFileManifestConverter(token_manager, sequence_name))
    pipeline.add_converter(IDD3DTokenRegistrySaver(token_manager, registry_path))
    
    return pipeline
  
ConverterRegistry.register('idd3d', 'nuscenes', build_idd3d_to_nuscenes_pipeline)
