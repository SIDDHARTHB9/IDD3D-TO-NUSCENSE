from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from abc import ABC, abstractmethod
import os
import json
import threading
from queue import Queue
from datetime import datetime
import logging
import uuid
import math

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for active conversions
conversion_state = {
    'active': False,
    'logs': Queue(),
    'progress': 0,
    'total_steps': 0,
    'current_step': 0
}

conversion_lock = threading.Lock()


class LogHandler:
    """Handler to capture conversion logs and emit them"""
    
    def __init__(self, log_queue):
        self.queue = log_queue
    
    def log(self, message, log_type='info'):
        """Add a log entry to the queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'type': log_type
        }
        self.queue.put(log_entry)
        logger.info(f"[{log_type.upper()}] {message}")


# TOKEN AND TIMESTAMP MANAGER

class TokenTimestampManager:
    """
    Manages consistent token generation and timestamp synchronization
    across all converted files for the intermediate format.
    """
    
    def __init__(self, base_timestamp=None, frame_rate_hz=10):
        """
        Initialize the manager.
        
        Args:
            base_timestamp: Starting timestamp in microseconds (default: current time)
            frame_rate_hz: Frame rate in Hz (default: 10 for IDD3D)
        """
        self.frame_rate_hz = frame_rate_hz
        self.frame_interval_us = int(1_000_000 / frame_rate_hz)  # microseconds between frames
        
        # Use provided base timestamp or generate one
        if base_timestamp is None:
            # Use a reasonable base timestamp (January 1, 2022, 00:00:00 UTC)
            self.base_timestamp = 1640995200000000  # microseconds
        else:
            self.base_timestamp = base_timestamp
        
        # Token registries - store tokens by ID for consistency
        self.frame_tokens = {}           # frame_id -> token
        self.instance_tokens = {}        # obj_id -> instance_token
        self.category_tokens = {}        # category_name -> category_token
        self.sensor_tokens = {}          # sensor_name -> sensor_token
        self.calibration_tokens = {}    # sensor_name -> calibration_token
        self.scene_token = None
        
    def get_timestamp(self, frame_index):
        """
        Generate timestamp for a frame based on its index.
        
        Args:
            frame_index: 0-based frame index
            
        Returns:
            timestamp in microseconds
        """
        return self.base_timestamp + (frame_index * self.frame_interval_us)
    
    def get_frame_token(self, frame_id):
        """Get or create a consistent token for a frame."""
        if frame_id not in self.frame_tokens:
            self.frame_tokens[frame_id] = uuid.uuid4().hex
        return self.frame_tokens[frame_id]
    
    def get_instance_token(self, obj_id):
        """Get or create a consistent token for an object instance."""
        if obj_id not in self.instance_tokens:
            self.instance_tokens[obj_id] = uuid.uuid4().hex
        return self.instance_tokens[obj_id]
    
    def get_category_token(self, category_name):
        """Get or create a consistent token for a category."""
        if category_name not in self.category_tokens:
            self.category_tokens[category_name] = uuid.uuid4().hex
        return self.category_tokens[category_name]
    
    def get_sensor_token(self, sensor_name):
        """Get or create a consistent token for a sensor."""
        if sensor_name not in self.sensor_tokens:
            self.sensor_tokens[sensor_name] = uuid.uuid4().hex
        return self.sensor_tokens[sensor_name]
    
    def get_calibration_token(self, sensor_name):
        """Get or create a consistent token for sensor calibration."""
        if sensor_name not in self.calibration_tokens:
            self.calibration_tokens[sensor_name] = uuid.uuid4().hex
        return self.calibration_tokens[sensor_name]
    
    def get_scene_token(self):
        """Get or create the scene token."""
        if self.scene_token is None:
            self.scene_token = uuid.uuid4().hex
        return self.scene_token
    
    def generate_annotation_token(self):
        """Generate a unique token for an annotation (not tracked)."""
        return uuid.uuid4().hex
    
    def save_registry(self, output_path):
        """Save the token registry to a JSON file for debugging."""
        registry = {
            'base_timestamp': self.base_timestamp,
            'frame_rate_hz': self.frame_rate_hz,
            'frame_interval_us': self.frame_interval_us,
            'scene_token': self.scene_token,
            'frame_tokens': self.frame_tokens,
            'instance_tokens': self.instance_tokens,
            'category_tokens': self.category_tokens,
            'sensor_tokens': self.sensor_tokens,
            'calibration_tokens': self.calibration_tokens
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2)


# CONVERTER FRAMEWORK - Base Classes

class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, root: str, sequence: str = None):
        self.root = os.path.abspath(root)
        self.sequence = sequence
    
    @abstractmethod
    def ensure_output_dirs(self):
        """Create necessary output directories"""
        pass
    
    @abstractmethod
    def validate(self) -> dict:
        """Validate dataset structure. Return {'valid': bool, 'error': str, ...}"""
        pass


class BaseConverter(ABC):
    """Abstract base converter class"""
    
    def __init__(self, name: str):
        self.name = name
        self.dry_run = False
    
    @abstractmethod
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        """Execute conversion. Must be implemented by subclasses."""
        pass


class DatasetConversionPipeline:
    """Orchestrates multi-step dataset conversions"""
    
    def __init__(self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
        self.converters = []
    
    def add_converter(self, converter: BaseConverter):
        """Add a converter to the pipeline"""
        self.converters.append(converter)
        return self
    
    def run(self, data_loader: BaseDataLoader, log_handler: LogHandler):
        """Execute all converters in sequence"""
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


# IDD3D IMPLEMENTATION

class IDD3DDataLoader(BaseDataLoader):
    """Loader for IDD3D dataset"""
    
    def __init__(self, root: str, sequence: str = '20220118103308_seq_10'):
        super().__init__(root, sequence)
        self.seq_base = os.path.join(
            self.root,
            'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val',
            sequence
        )
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'label')
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json')
        
        self.out_data = os.path.join(self.root, 'Intermediate_format/data')
        self.annot_out = os.path.join(self.root, 'Intermediate_format/anotations')
        self.converted_lidar = os.path.join(self.out_data, 'lidar')
        self.cam_dir = os.path.join(self.out_data, 'cam')
    
    def ensure_output_dirs(self):
        import shutil
        
        # Clean up old output directories if they exist
        if os.path.exists(self.annot_out):
            try:
                shutil.rmtree(self.annot_out)
            except Exception:
                pass
        
        if os.path.exists(self.out_data):
            # Don't delete out_data entirely, just clean subdirectories
            # This will be handled by individual converters
            pass
        
        # Create fresh directories
        os.makedirs(self.out_data, exist_ok=True)
        os.makedirs(self.annot_out, exist_ok=True)
        os.makedirs(self.converted_lidar, exist_ok=True)
    
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
            return {'valid': False, 'error': f'Missing directories: {", ".join(missing)}'}
        
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
        return [os.path.join(self.lidar_dir, f) for f in sorted(os.listdir(self.lidar_dir)) 
                if f.lower().endswith('.pcd')]
    
    def read_annotations(self):
        if not os.path.exists(self.annot_json):
            return {}
        with open(self.annot_json, 'r') as f:
            return json.load(f)


class IDD3DLidarConverter(BaseConverter):
    """Convert IDD3D PCD files to nuScenes .pcd.bin files"""
    
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
        
        files = [os.path.basename(p) for p in data_loader.list_lidar_files()]
        dst_dir = data_loader.converted_lidar
        src_dir = data_loader.lidar_dir
        
        if not files:
            log_handler.log("No LiDAR files found", 'warning')
            return
        
        # CLEAN UP EXISTING CONVERTED_LIDAR DIRECTORY
        if os.path.exists(dst_dir):
            import shutil
            log_handler.log(f"Cleaning up existing converted_lidar directory: {dst_dir}", "info")
            try:
                shutil.rmtree(dst_dir)
                log_handler.log("Old converted_lidar directory removed", "success")
            except Exception as e:
                log_handler.log(f"Warning: Could not remove old converted_lidar directory: {str(e)}", "warning")
        
        # Create fresh directory
        os.makedirs(dst_dir, exist_ok=True)
        converted = 0
        placeholders = 0
        
        for i, fname in enumerate(files):
            src = os.path.join(src_dir, fname)
            base = os.path.splitext(fname)[0]
            dst = os.path.join(dst_dir, base + '.pcd.bin')
            
            try:
                if use_o3d:
                    pcd = o3d.io.read_point_cloud(src)
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                    pts = np.hstack((xyz, intensity))
                    pts.astype(np.float32).tofile(dst)
                    converted += 1
                else:
                    open(dst, 'wb').close()
                    placeholders += 1
            except Exception:
                open(dst, 'wb').close()
                placeholders += 1
        
        log_handler.log(f"LiDAR conversion complete: {converted} converted, {placeholders} placeholders", 'success')


class IDD3DCameraConverter(BaseConverter):
    """Convert IDD3D camera images from PNG to JPEG - keeps cam0-cam5 naming"""
    
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
        
        cameradir = os.path.join(dataloader.seq_base, "camera")
        if not os.path.exists(cameradir):
            loghandler.log("No camera directory found", "warning")
            return
        
        # Keep original camera naming - output folders: cam0, cam1, cam2, cam3, cam4, cam5
        camerachannels = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
        
        camdir = os.path.join(dataloader.out_data, "cam")
        
        # CLEAN UP EXISTING CAM DIRECTORY
        if os.path.exists(camdir):
            import shutil
            loghandler.log(f"Cleaning up existing cam directory: {camdir}", "info")
            try:
                shutil.rmtree(camdir)
                loghandler.log("Old cam directory removed", "success")
            except Exception as e:
                loghandler.log(f"Warning: Could not remove old cam directory: {str(e)}", "warning")
        
        # Create fresh cam directory
        os.makedirs(camdir, exist_ok=True)
        
        converted = 0
        errors = 0
        
        for camid in camerachannels:
            camfolder = os.path.join(cameradir, camid)
            if not os.path.exists(camfolder):
                loghandler.log(f"Camera folder not found: {camfolder}", "warning")
                continue
            
            # Output folder uses the same naming: cam0, cam1, etc. (with space)
            output_camid = f"cam {camid[-1]}"  # "cam0" -> "cam 0", "cam1" -> "cam 1"
            camsubdir = os.path.join(camdir, output_camid)
            os.makedirs(camsubdir, exist_ok=True)
            
            pngfiles = sorted([f for f in os.listdir(camfolder) if f.lower().endswith('.png')])
            
            loghandler.log(f"Processing {camid}: {len(pngfiles)} images -> {output_camid}", "info")
            
            for fname in pngfiles:
                srcpath = os.path.join(camfolder, fname)
                basename = os.path.splitext(fname)[0]
                dstpath = os.path.join(camsubdir, basename + '.jpg')
                
                try:
                    if usepil:
                        img = Image.open(srcpath)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dstpath, 'JPEG', quality=95)
                        converted += 1
                except Exception as e:
                    errors += 1
                    loghandler.log(f"Error converting {fname}: {str(e)}", "error")
        
        loghandler.log(f"Camera conversion complete: {converted} images converted to cam, {errors} errors", "success")


class IDD3DCalibConverter(BaseConverter):
    """Generate calibration stubs for IDD3D"""
    
    def __init__(self):
        super().__init__('calib')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        sensors = ['Lidar', 'cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5']
        calibrated_list = []
        sensors_j = []
        
        for s in sensors:
            token = uuid.uuid4().hex
            sensor_token = uuid.uuid4().hex
            entry = {
                "token": token,
                "sensor_token": sensor_token,
                "translation": [0.0, 0.0, 1.8] if s.upper().startswith('LIDAR') else [0.0, 0.0, 1.6],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "camera_intrinsic": []
            }
            calibrated_list.append(entry)
            sensors_j.append({
                "token": sensor_token,
                "modality": "lidar" if s.upper().startswith('LIDAR') else "camera",
                "channel": s,
                "description": f"Stub for {s}",
                "firmware_rev": "",
                "data": {}
            })
        
        out_calib_dir = os.path.join(data_loader.out_data, 'calibration')
        os.makedirs(out_calib_dir, exist_ok=True)
        
        with open(os.path.join(out_calib_dir, 'calibrated_sensor.json'), 'w') as f:
            json.dump(calibrated_list, f, indent=2)
        with open(os.path.join(out_calib_dir, 'sensors.json'), 'w') as f:
            json.dump(sensors_j, f, indent=2)
        
        log_handler.log("Calibration stubs created", 'success')


class IDD3DFrameConverter(BaseConverter):
    """Convert IDD3D frame annotations to frames.json"""
    
    def __init__(self, token_manager, sequence_name: str = 'seq'):
        super().__init__('frame') # Renamed from 'annot'
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        frames = []
        
        # Get scene token from token manager
        scene_token = self.token_manager.get_scene_token()
        
        for i, frame_id in enumerate(frame_ids):
            data = annot_data[frame_id]
            
            # Generate frame token
            frame_token = self.token_manager.get_frame_token(frame_id)
            
            # Generate proper timestamp
            timestamp = self.token_manager.get_timestamp(i)
            
            # Build sensor_data structure (as dict of strings, per user template)
            sensor_data = {}
            
            # Add camera data (cam0 to cam5)
            for cam_idx in range(6):
                cam_key = f"cam{cam_idx}"
                cam_filename = data.get(cam_key, f"{frame_id}.jpg")
                # Path format matches IDD3DCameraConverter output
                sensor_data[cam_key] = f"data/cam/cam {cam_idx}/{cam_filename}"
            
            # Add lidar data
            lidar_filename = data.get("lidar", f"{frame_id}.pcd.bin")
            # Path format matches IDD3DLidarConverter output
            sensor_data["lidar"] = f"data/lidar/{lidar_filename}"
            
            frame = {
                "frame_token": frame_token,
                "timestamp": timestamp,
                "scene_token": scene_token,
                "sensor_data": sensor_data,
                "prev_frame_token": self.token_manager.get_frame_token(frame_ids[i-1]) if i > 0 else None,
                "next_frame_token": self.token_manager.get_frame_token(frame_ids[i+1]) if i < len(frame_ids)-1 else None,
                "calibration_token": "calib_default" # Added as per user template
            }
            
            frames.append(frame)
        
        out_path = os.path.join(data_loader.annot_out, 'frames.json')
        with open(out_path, 'w') as f:
            json.dump(frames, f, indent=2)
        
        log_handler.log(f"frames.json converted ({len(frames)} frames)", 'success')


class IDD3DSceneConverter(BaseConverter):
    """Generate scene.json with scene metadata"""
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('scene')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader, log_handler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for scene conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        # Get scene token from token manager
        scene_token = self.token_manager.get_scene_token()
        
        # Get log token (can be a placeholder since we're skipping log.json)
        log_token = uuid.uuid4().hex
        
        # Get first and last sample tokens
        first_sample_token = self.token_manager.get_frame_token(frame_ids[0])
        last_sample_token = self.token_manager.get_frame_token(frame_ids[-1])
        
        # Try to get metadata from annot_data.json
        first_frame_data = annot_data[frame_ids[0]]
        
        # Create scene entry following nuScenes schema
        scene = {
            "token": scene_token,
            "log_token": log_token,
            "nbr_samples": None,
            "first_sample_token": first_sample_token,
            "last_sample_token": last_sample_token,
            "name": self.sequence_name,
            "description": f"IDD3D sequence {self.sequence_name}"
        }
        
        # Save scene.json (as a list with single scene)
        out_path = os.path.join(data_loader.annot_out, 'scene.json')
        with open(out_path, 'w') as f:
            json.dump([scene], f, indent=2)
        
        log_handler.log(f"Scene file created", 'success')
        log_handler.log(f"  Scene token: {scene_token}", 'info')
        log_handler.log(f"  Number of samples: null (to be calculated)", 'info')
        log_handler.log(f"  First sample token: {first_sample_token}", 'info')
        log_handler.log(f"  Last sample token: {last_sample_token}", 'info')


class IDD3DSampleDataConverter(BaseConverter):
    """Generate sample_data.json linking samples to sensor data files"""
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample_data')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader, log_handler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for sample_data conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        sample_data_list = []
        
        # Camera channels
        camera_channels = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
        
        for i, frame_id in enumerate(frame_ids):
            frame_data = annot_data[frame_id]
            sample_token = self.token_manager.get_frame_token(frame_id)
            timestamp = self.token_manager.get_timestamp(i)
            
            # Get calibrated_sensor_token (from calibration)
            # We'll use placeholder tokens that match calibration file
            
            # Add LiDAR sample_data
            lidar_filename = frame_data.get('lidar', f'{frame_id}.pcd.bin')
            lidar_sample_data_token = uuid.uuid4().hex
            
            sample_data_list.append({
                "token": lidar_sample_data_token,
                "sample_token": sample_token,
                "calibrated_sensor_token": self.token_manager.get_calibration_token("Lidar"),
                "filename": f"data/lidar/{lidar_filename}",
                "fileformat": "pcd.bin",
                "width": 0,
                "height": 0,
                "timestamp": timestamp,
                "is_key_frame": True,
                "next": "",
                "prev": ""
            })
            
            # Add Camera sample_data for each channel
            for cam_idx, cam_channel in enumerate(camera_channels):
                cam_filename = frame_data.get(f'cam{cam_idx}', f'{frame_id}.jpg')
                cam_sample_data_token = uuid.uuid4().hex
                
                sample_data_list.append({
                    "token": cam_sample_data_token,
                    "sample_token": sample_token,
                    "calibrated_sensor_token": self.token_manager.get_calibration_token(cam_channel),
                    "filename": f"data/cam/cam {cam_idx}/{cam_filename}",
                    "fileformat": "jpg",
                    "width": 1920,  # IDD3D typical resolution
                    "height": 1080,
                    "timestamp": timestamp,
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
        
        # Link prev/next for each sensor modality
        # Group by calibrated_sensor_token
        sensor_groups = {}
        for sd in sample_data_list:
            sensor_token = sd['calibrated_sensor_token']
            if sensor_token not in sensor_groups:
                sensor_groups[sensor_token] = []
            sensor_groups[sensor_token].append(sd)
        
        # Link prev/next within each sensor group
        for sensor_token, sd_list in sensor_groups.items():
            for i, sd in enumerate(sd_list):
                if i > 0:
                    sd['prev'] = sd_list[i-1]['token']
                if i < len(sd_list) - 1:
                    sd['next'] = sd_list[i+1]['token']
        
        # Save sample_data.json
        out_path = os.path.join(data_loader.annot_out, 'sample_data.json')
        with open(out_path, 'w') as f:
            json.dump(sample_data_list, f, indent=2)
        
        log_handler.log(f"Sample data file created with {len(sample_data_list)} entries", 'success')
        log_handler.log(f"  LiDAR entries: {len(frame_ids)}", 'info')
        log_handler.log(f"  Camera entries: {len(frame_ids) * 6}", 'info')


class IDD3DCategoryConverter(BaseConverter):
    """Generate category.json with synced tokens"""
    
    def __init__(self, token_manager):
        super().__init__('category')
        self.token_manager = token_manager
    
    def run(self, data_loader, log_handler):
        # IDD3D to nuScenes category mapping (Standardized to 15 keys)
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle', # Standardized
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }

        # Descriptions for the target nuScenes categories
        nuscenes_descriptions = {
            'vehicle.car': 'A car.',
            'vehicle.truck': 'A truck.',
            'vehicle.bus': 'A bus.',
            'vehicle.motorcycle': 'A motorcycle or motorcyclist.',
            'vehicle.bicycle': 'A bicycle.',
            'vehicle.auto': 'An auto-rickshaw.',
            'human.pedestrian.adult': 'An adult pedestrian.',
            'human.pedestrian.rider': 'A person riding a vehicle (e.g., bicycle).',
            'animal': 'An animal.',
            'static_object.traffic_light': 'A traffic light.',
            'static_object.traffic_sign': 'A traffic sign.',
            'static_object.pole': 'A pole.',
            'vehicle.other': 'Other vehicle types.',
            'movable_object.debris': 'Miscellaneous debris or movable objects.'
        }
        
        categories = []
        processed_nuscenes_names = set()
        
        # Statically create all 15 categories
        for idd_type, nuscenes_name in idd3d_to_nuscenes_categories.items():
            if nuscenes_name in processed_nuscenes_names:
                continue
            processed_nuscenes_names.add(nuscenes_name)
            
            category = {
                "token": self.token_manager.get_category_token(nuscenes_name),
                "name": nuscenes_name,
                "description": nuscenes_descriptions.get(nuscenes_name, f"Category for {nuscenes_name}")
            }
            categories.append(category)
        
        # Save category.json
        out_path = os.path.join(data_loader.annot_out, 'category.json')
        with open(out_path, 'w') as f:
            json.dump(categories, f, indent=2)
        
        log_handler.log(f"Category file created with {len(categories)} static categories", 'success')
        for cat in categories:
            log_handler.log(f"  - Generated: {cat['name']} (token: ...{cat['token'][-6:]})", 'info')


class IDD3DSampleConverter(BaseConverter):
    """Generate sample.json with proper timestamps and synced tokens"""
    
    def __init__(self, token_manager, sequence_name='seq'):
        super().__init__('sample')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader, log_handler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for sample conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        scene_token = self.token_manager.get_scene_token()
        samples = []
        
        for i, frame_id in enumerate(frame_ids):
            # Use token manager for consistent frame tokens
            token = self.token_manager.get_frame_token(frame_id)
            
            # Generate proper timestamp based on frame rate (10Hz = 100ms intervals)
            timestamp = self.token_manager.get_timestamp(i)
            
            # Link to previous and next samples
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
        
        # Save sample.json
        out_path = os.path.join(data_loader.annot_out, 'sample.json')
        with open(out_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        log_handler.log(f"Sample file created with {len(samples)} samples", 'success')
        log_handler.log(f"  Frame rate: {self.token_manager.frame_rate_hz} Hz", 'info')
        log_handler.log(f"  Timestamp range: {samples[0]['timestamp']} - {samples[-1]['timestamp']}", 'info')


class IDD3DSampleAnnotationConverter(BaseConverter):
    """Convert IDD3D object annotations to nuScenes sample_annotation.json format"""
    
    def __init__(self, token_manager, sequence_name: str = 'seq'):
        super().__init__('sample_annotation')
        self.token_manager = token_manager
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for sample_annotation conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        # IDD3D to category mapping (Standardized to 15 keys)
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle', # Standardized
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        sample_annotations = []
        object_instances = {}
        
        # First pass: collect all annotations
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get("obj_id")
                    if not obj_id:
                        continue
                    
                    # Use token manager for instance token
                    instance_token = self.token_manager.get_instance_token(obj_id)
                    
                    if obj_id not in object_instances:
                        object_instances[obj_id] = {
                            'instance_token': instance_token,
                            'annotations': []
                        }
                    
                    # Generate annotation token
                    ann_token = self.token_manager.generate_annotation_token()
                    
                    # Get frame token
                    frame_token = self.token_manager.get_frame_token(frame_id)
                    
                    # Extract PSR
                    psr = obj.get("psr", {})
                    position = psr.get("position", {})
                    rotation = psr.get("rotation", {})
                    scale = psr.get("scale", {})
                    
                    translation = [
                        position.get("x", 0.0),
                        position.get("y", 0.0),
                        position.get("z", 0.0)
                    ]
                    
                    size = [
                        scale.get("x", 1.0),
                        scale.get("y", 1.0),
                        scale.get("z", 1.0)
                    ]
                    
                    rotation_quat = [
                        rotation.get("x", 0.0),
                        rotation.get("y", 0.0),
                        rotation.get("z", 0.0),
                        1.0
                    ]
                    
                    annotation = {
                        "token": ann_token,
                        "sample_token": frame_token,
                        "instance_token": instance_token,
                        "translation": translation,
                        "size": size,
                        "rotation": rotation_quat,
                        "prev": "",
                        "next": "",
                        "num_lidar_pts": 0,
                        "num_radar_pts": 0
                    }
                    
                    object_instances[obj_id]['annotations'].append(annotation)
                    
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        # Second pass: link prev/next
        for obj_id, instance_data in object_instances.items():
            annotations = instance_data['annotations']
            for i, ann in enumerate(annotations):
                if i > 0:
                    ann['prev'] = annotations[i-1]['token']
                if i < len(annotations) - 1:
                    ann['next'] = annotations[i+1]['token']
                sample_annotations.append(ann)
        
        # Save sample_annotation.json
        out_path = os.path.join(data_loader.annot_out, 'sample_annotation.json')
        with open(out_path, 'w') as f:
            json.dump(sample_annotations, f, indent=2)
        
        log_handler.log(f"Sample annotation file created with {len(sample_annotations)} annotations", 'success')


class IDD3DInstanceConverter(BaseConverter):
    """Generate instance.json with synced tokens"""
    
    def __init__(self, token_manager):
        super().__init__('instance')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        # IDD3D to nuScenes category mapping (Standardized to 15 keys)
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle', # Standardized
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        # Track unique instances across all frames
        instance_tracker = {}
        
        # Collect all objects and their appearances
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
                    
                    # Create new instance entry if this obj_id hasn't been seen
                    if obj_id not in instance_tracker:
                        # Use token manager for instance token
                        instance_token = self.token_manager.get_instance_token(obj_id)
                        
                        # Map to category
                        category_name = idd3d_to_nuscenes_categories.get(
                            obj_type, 
                            f'movable_object.{obj_type.lower()}'
                        )
                        
                        # Use token manager for category token
                        # This token will match the one in category.json
                        category_token = self.token_manager.get_category_token(category_name)
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'obj_type': obj_type,
                            'first_frame': frame_id,
                            'last_frame': frame_id,
                            'first_annotation_token': self.token_manager.generate_annotation_token(),
                            'last_annotation_token': self.token_manager.generate_annotation_token()
                        }
                    else:
                        # Update last frame for this instance
                        instance_tracker[obj_id]['last_frame'] = frame_id
                        instance_tracker[obj_id]['last_annotation_token'] = self.token_manager.generate_annotation_token()
                        
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        # Create instance entries
        instances = []
        obj_type_counts = {}
        
        for obj_id, data in instance_tracker.items():
            instance = {
                "token": data['instance_token'],
                "category_token": data['category_token'],
                "nbr_annotations": None,
                "first_annotation_token": data['first_annotation_token'],
                "last_annotation_token": data['last_annotation_token']
            }
            instances.append(instance)
            
            # Count instances by type
            obj_type = data['obj_type']
            obj_type_counts[obj_type] = obj_type_counts.get(obj_type, 0) + 1
        
        # Log statistics
        log_handler.log(f"Instance breakdown by object type:", 'info')
        for obj_type, count in sorted(obj_type_counts.items()):
            log_handler.log(f"  - {obj_type}: {count} instances", 'info')
        
        # Save instance.json
        out_path = os.path.join(data_loader.annot_out, 'instance.json')
        with open(out_path, 'w') as f:
            json.dump(instances, f, indent=2)
        
        log_handler.log(f"Instance file created with {len(instances)} instances", 'success')


class IDD3DObjectsJsonConverter(BaseConverter):
    """Generate objects.json with bbox_3d format and synced tokens"""
    
    def __init__(self, token_manager):
        super().__init__("objects_json")
        self.token_manager = token_manager
    
    def run(self, dataloader, loghandler):
        annot_data = dataloader.read_annotations()
        if not annot_data:
            loghandler.log("No annotations found for objects.json conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        # IDD3D to nuScenes category mapping (Standardized to 15 keys)
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'MotorcyleRider': 'vehicle.motorcycle', # Standardized
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.traffic_light',
            'TrafficSign': 'static_object.traffic_sign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        objects_list = []
        
        loghandler.log(f"Processing {len(frame_ids)} frames for objects.json", 'info')
        
        # Process each frame
        for frame_id in frame_ids:
            label_path = os.path.join(dataloader.label_dir, f"{frame_id}.json")
            
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                
                for obj in label_objects:
                    obj_id = obj.get('obj_id')
                    obj_type = obj.get('obj_type')
                    psr = obj.get('psr', {})
                    position = psr.get('position', {})
                    rotation = psr.get('rotation', {})
                    scale = psr.get('scale', {})
                    
                    if not obj_id or not obj_type:
                        continue
                    
                    # Use token manager for consistent tokens
                    instance_token = self.token_manager.get_instance_token(obj_id)
                    frame_token = self.token_manager.get_frame_token(frame_id)
                    
                    # Map obj_type to category
                    category_name = idd3d_to_nuscenes_categories.get(
                        obj_type, 
                        f'movable_object.{obj_type.lower()}'
                    )
                    # This token will match the one in category.json
                    category_token = self.token_manager.get_category_token(category_name)
                    
                    # Generate unique annotation token (not tracked)
                    object_token = self.token_manager.generate_annotation_token()
                    
                    # Extract bbox data
                    center = [
                        position.get('x', 0.0),
                        position.get('y', 0.0),
                        position.get('z', 0.0)
                    ]
                    
                    size = [
                        scale.get('x', 1.0),
                        scale.get('y', 1.0),
                        scale.get('z', 1.0)
                    ]
                    
                    # Convert rotation from Euler to quaternion
                    rx = rotation.get('x', 0.0)
                    ry = rotation.get('y', 0.0)
                    rz = rotation.get('z', 0.0)
                    
                    cy = math.cos(rz * 0.5)
                    sy = math.sin(rz * 0.5)
                    cp = math.cos(ry * 0.5)
                    sp = math.sin(ry * 0.5)
                    cr = math.cos(rx * 0.5)
                    sr = math.sin(rx * 0.5)
                    
                    qw = cr * cp * cy + sr * sp * sy
                    qx = sr * cp * cy - cr * sp * sy
                    qy = cr * sp * cy + sr * cp * sy
                    qz = cr * cp * sy - sr * sp * cy
                    
                    rotation_quat = [qw, qx, qy, qz]
                    
                    # Create object entry
                    obj_entry = {
                        "object_token": object_token,
                        "frame_token": frame_token,
                        "instance_token": instance_token,
                        "category_token": category_token,
                        "attribute_tokens": [],
                        "bbox_3d": {
                            "center": center,
                            "size": size,
                            "rotation": rotation_quat
                        },
                        "num_lidar_pts": 0
                    }
                    
                    objects_list.append(obj_entry)
                    
            except Exception as e:
                loghandler.log(f"Error processing frame {frame_id}: {str(e)}", 'warning')
                continue
        
        # Ensure output directory exists
        os.makedirs(dataloader.annot_out, exist_ok=True)
        
        # Save objects.json
        out_path = os.path.join(dataloader.annot_out, 'objects.json')
        
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(objects_list, f, indent=2)
            
            loghandler.log(f"objects.json created with {len(objects_list)} 3D bounding boxes", "success")
            loghandler.log(f"Tracked {len(self.token_manager.instance_tokens)} unique instances", "info")
            loghandler.log(f"Output saved to: {out_path}", "success")
                
        except Exception as e:
            loghandler.log(f"Error writing objects.json: {str(e)}", 'error')
            import traceback
            loghandler.log(traceback.format_exc(), 'error')
            raise


class IDD3DTimestampSyncConverter(BaseConverter):
    """Synchronize timestamps across all JSON files using TokenTimestampManager"""
    
    def __init__(self, token_manager):
        super().__init__('timestamp_sync')
        self.token_manager = token_manager
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        """
        Ensure consistent timestamps across all files using the token manager
        """
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for timestamp sync", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        log_handler.log(f"Timestamps already synced via TokenTimestampManager", 'info')
        log_handler.log(f"  Base timestamp: {self.token_manager.base_timestamp} μs", 'info')
        log_handler.log(f"  Frame rate: {self.token_manager.frame_rate_hz} Hz", 'info')
        log_handler.log(f"  Frame interval: {self.token_manager.frame_interval_us} μs", 'info')
        
        # Update frames.json with synced timestamps (if it exists)
        frames_path = os.path.join(data_loader.annot_out, 'frames.json')
        if os.path.exists(frames_path):
            try:
                with open(frames_path, 'r') as f:
                    frames = json.load(f)
                
                # Re-check frame tokens and timestamps
                updated_frames = 0
                for i, frame in enumerate(frames):
                    if i >= len(frame_ids):
                        log_handler.log(f"Warning: frames.json has more entries ({len(frames)}) than frame_ids ({len(frame_ids)})", 'warning')
                        break
                    
                    frame_id = frame_ids[i] # Assume frames are in order
                    
                    # Update with proper timestamp from token manager
                    new_timestamp = self.token_manager.get_timestamp(i)
                    if frame.get('timestamp') != new_timestamp:
                        frame['timestamp'] = new_timestamp
                        updated_frames += 1
                        
                    # Update tokens to match
                    new_token = self.token_manager.get_frame_token(frame_id)
                    if frame.get('frame_token') != new_token:
                        frame['frame_token'] = new_token
                        if updated_frames == 0: updated_frames = 1 # Mark as updated
                
                if updated_frames > 0:
                    with open(frames_path, 'w') as f:
                        json.dump(frames, f, indent=2)
                    log_handler.log(f"Verified/Updated timestamps in frames.json ({updated_frames} frames)", 'success')
                else:
                    log_handler.log("Timestamps in frames.json are already correct", 'info')
                    
            except Exception as e:
                log_handler.log(f"Error updating frames.json timestamps: {str(e)}", 'warning')
        
        # Save token registry for debugging
        registry_path = os.path.join(data_loader.annot_out, 'token_registry.json')
        try:
            self.token_manager.save_registry(registry_path)
            log_handler.log(f"Token registry saved to: {registry_path}", 'info')
        except Exception as e:
            log_handler.log(f"Warning: Could not save token registry: {str(e)}", 'warning')
        
        log_handler.log("Timestamp synchronization complete", 'success')


# CONVERTER REGISTRY

class ConverterRegistry:
    """Registry for dataset conversions"""
    
    _conversions = {}
    
    @classmethod
    def register(cls, source: str, target: str, pipeline_builder):
        """Register a conversion pipeline"""
        key = (source, target)
        cls._conversions[key] = pipeline_builder
    
    @classmethod
    def get_pipeline(cls, source: str, target: str, config: dict):
        """Get a pipeline for source->target conversion"""
        key = (source, target)
        if key not in cls._conversions:
            raise ValueError(f"No conversion registered for {source} -> {target}")
        pipeline_builder = cls._conversions[key]
        return pipeline_builder(config)
    
    @classmethod
    def get_available_conversions(cls):
        """Get all available conversions"""
        return [{'source': s, 'target': t} for s, t in cls._conversions.keys()]


# REGISTER CONVERSIONS

def build_idd3d_to_nuscenes_pipeline(config: dict) -> DatasetConversionPipeline:
    """Build conversion pipeline for IDD3D -> nuScenes with TokenTimestampManager"""
    pipeline = DatasetConversionPipeline('idd3d', 'nuscenes')
    
    conversions = config.get('conversions', {})
    sequence_name = config.get('sequence_id', 'seq_10')
    
    # Create token manager with 10Hz frame rate for IDD3D
    token_manager = TokenTimestampManager(frame_rate_hz=10)
    
    # PHASE 1: Data Conversion (no dependencies)
    if conversions.get('lidar', False):
        pipeline.add_converter(IDD3DLidarConverter())
    if conversions.get('camera', False):
        pipeline.add_converter(IDD3DCameraConverter())
    if conversions.get('calib', False):
        pipeline.add_converter(IDD3DCalibConverter())
    
    # PHASE 2: Taxonomy (no dependencies)
    if conversions.get('category', False):
        pipeline.add_converter(IDD3DCategoryConverter(token_manager))
    
    # PHASE 3: Scene (creates scene_token for samples)
    if conversions.get('scene', False):
        pipeline.add_converter(IDD3DSceneConverter(token_manager, sequence_name))
    
    # PHASE 4: Frames (generates frames.json, depends on scene token)
    if conversions.get('frame', False): # Renamed from 'annot'
        pipeline.add_converter(IDD3DFrameConverter(token_manager, sequence_name)) # Renamed class
    
    # PHASE 4: Samples (depends on scene)
    if conversions.get('sample', False):
        pipeline.add_converter(IDD3DSampleConverter(token_manager, sequence_name))
    
    # PHASE 5: Sample Data (depends on sample and calibration)
    if conversions.get('sample_data', False):
        pipeline.add_converter(IDD3DSampleDataConverter(token_manager, sequence_name))
    
    # PHASE 6: Annotations (depends on sample, instance, category)
    if conversions.get('instance', False):
        pipeline.add_converter(IDD3DInstanceConverter(token_manager))
    if conversions.get('sample_annotation', False):
        pipeline.add_converter(IDD3DSampleAnnotationConverter(token_manager, sequence_name))
    if conversions.get('objects', False):
        pipeline.add_converter(IDD3DObjectsJsonConverter(token_manager))
    
    # PHASE 7: Timestamp sync (always runs last)
    pipeline.add_converter(IDD3DTimestampSyncConverter(token_manager))
    
    return pipeline


ConverterRegistry.register('idd3d', 'nuscenes', build_idd3d_to_nuscenes_pipeline)


# FLASK API ENDPOINTS

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Dataset Converter API'})


@app.route('/api/conversions', methods=['GET'])
def get_conversions():
    """Get all available conversions"""
    conversions = ConverterRegistry.get_available_conversions()
    return jsonify({'conversions': conversions})


@app.route('/api/validate-paths', methods=['POST'])
def validate_paths():
    """Validate dataset paths"""
    data = request.json
    source = data.get('source', 'idd3d')
    root_path = data.get('root_path')
    sequence_id = data.get('sequence_id', '20220118103308_seq_10')
    
    if not root_path or not os.path.exists(root_path):
        return jsonify({'valid': False, 'error': f'Root path does not exist: {root_path}'}), 400
    
    if source == 'idd3d':
        loader = IDD3DDataLoader(root_path, sequence_id)
        validation = loader.validate()
        return jsonify(validation)
    
    return jsonify({'valid': False, 'error': f'Unknown source dataset: {source}'}), 400


@app.route('/api/convert/stream', methods=['POST'])
def convert_stream():
    """Start conversion and stream logs via SSE"""
    with conversion_lock:
        if conversion_state['active']:
            return jsonify({'error': 'Conversion already in progress'}), 409
        conversion_state['active'] = True
    
    data = request.json
    source = data.get('source', 'idd3d')
    target = data.get('target', 'nuscenes')
    root_path = data.get('root_path')
    sequence_id = data.get('sequence_id', '20220118103308_seq_10')
    conversions = data.get('conversions', {})
    
    def generate():
        try:
            while not conversion_state['logs'].empty():
                conversion_state['logs'].get()
            
            log_handler = LogHandler(conversion_state['logs'])
            
            log_handler.log(f"Starting conversion: {source} → {target}", 'info')
            log_handler.log(f"Root path: {root_path}", 'info')
            log_handler.log(f"Sequence ID: {sequence_id}", 'info')
            
            # Create data loader
            if source == 'idd3d':
                loader = IDD3DDataLoader(root_path, sequence_id)
            else:
                raise ValueError(f"Unknown source dataset: {source}")
            
            loader.ensure_output_dirs()
            
            # Build and run pipeline
            pipeline = ConverterRegistry.get_pipeline(
                source, target,
                {'conversions': conversions, 'sequence_id': sequence_id}
            )
            conversion_state['total_steps'] = len(pipeline.converters)
            
            if conversion_state['total_steps'] == 0:
                log_handler.log("No conversion modules selected", 'warning')
            else:
                pipeline.run(loader, log_handler)
                log_handler.log("Conversion pipeline completed successfully!", 'success')
                log_handler.log(f"Output directory: {root_path}/Intermediate_format/", 'info')
        
        except Exception as e:
            log_handler.log(f"Conversion failed: {str(e)}", 'error')
            import traceback
            log_handler.log(traceback.format_exc(), 'error')
        
        finally:
            conversion_state['active'] = False
            while not conversion_state['logs'].empty():
                log_entry = conversion_state['logs'].get()
                yield f"data: {json.dumps(log_entry)}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


if __name__ == '__main__':
    print("Starting Extensible Dataset Converter API...")
    print("Registered conversions:")
    for conv in ConverterRegistry.get_available_conversions():
        print(f"  {conv['source']} → {conv['target']}")
    print("\nServer running on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
