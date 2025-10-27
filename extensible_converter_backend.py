from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from abc import ABC, abstractmethod
import os
import json
import threading
from queue import Queue
from datetime import datetime
import logging

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
                log_handler.log(f"✗ {converter.name} failed: {str(e)}", 'error')
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
        self.converted_lidar = os.path.join(self.out_data, 'converted_lidar')
    
    def ensure_output_dirs(self):
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
            log_handler.log("⚠ open3d not available, creating placeholder files", 'warning')
        
        files = [os.path.basename(p) for p in data_loader.list_lidar_files()]
        dst_dir = data_loader.converted_lidar
        src_dir = data_loader.lidar_dir
        
        if not files:
            log_handler.log("No LiDAR files found", 'warning')
            return
        
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
        
        log_handler.log(f"✓ LiDAR conversion complete: {converted} converted, {placeholders} placeholders", 'success')


class IDD3DCameraConverter(BaseConverter):
    """Convert IDD3D camera images from PNG to JPEG (nuScenes format)"""
    
    def __init__(self):
        super().__init__('camera')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        try:
            from PIL import Image
            use_pil = True
        except ImportError:
            use_pil = False
            log_handler.log("PIL/Pillow not available, skipping camera conversion", 'warning')
            return
        
        camera_dir = os.path.join(data_loader.seq_base, 'camera')
        if not os.path.exists(camera_dir):
            log_handler.log("No camera directory found", 'warning')
            return
        
        # Camera mapping for nuScenes
        camera_channels = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5']
        nuscenes_cameras = {
            'cam0': 'CAM_FRONT',
            'cam1': 'CAM_FRONT_LEFT',
            'cam2': 'CAM_FRONT_RIGHT',
            'cam3': 'CAM_BACK_LEFT',
            'cam4': 'CAM_BACK_RIGHT',
            'cam5': 'CAM_BACK'
        }
        
        # Create output directory for sweeps only
        sweeps_dir = os.path.join(data_loader.out_data, 'sweeps')
        
        converted = 0
        errors = 0
        
        for cam_id in camera_channels:
            cam_folder = os.path.join(camera_dir, cam_id)
            if not os.path.exists(cam_folder):
                continue
            
            nuscenes_cam_name = nuscenes_cameras[cam_id]
            
            # Create output directory for this camera in sweeps
            sweep_cam_dir = os.path.join(sweeps_dir, nuscenes_cam_name)
            os.makedirs(sweep_cam_dir, exist_ok=True)
            
            # Get all PNG files
            png_files = sorted([f for f in os.listdir(cam_folder) if f.lower().endswith('.png')])
            
            for fname in png_files:
                src_path = os.path.join(cam_folder, fname)
                base_name = os.path.splitext(fname)[0]
                dst_path = os.path.join(sweep_cam_dir, base_name + '.jpg')
                
                try:
                    if use_pil:
                        img = Image.open(src_path)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Save as JPEG with quality 95 (nuScenes standard)
                        img.save(dst_path, 'JPEG', quality=95)
                        converted += 1
                except Exception as e:
                    errors += 1
                    log_handler.log(f"Error converting {fname}: {str(e)}", 'error')
        
        log_handler.log(f"✓ Camera conversion complete: {converted} images converted to sweeps, {errors} errors", 'success')


class IDD3DCalibConverter(BaseConverter):
    """Generate calibration stubs for IDD3D"""
    
    def __init__(self):
        super().__init__('calib')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        import uuid
        
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


class IDD3DAnnotationConverter(BaseConverter):
    """Convert IDD3D frame annotations to intermediate format"""
    
    def __init__(self, sequence_name: str = 'seq'):
        super().__init__('annot')
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        frames = []
        
        for i, frame_id in enumerate(frame_ids):
            data = annot_data[frame_id]
            frame = {
                "frame_id": frame_id,
                "sequence": self.sequence_name,
                "lidar": data.get("lidar", ""),
                "timestamp": int(frame_id) * 100_000,
                "cameras": {
                    "CAM_FRONT": data.get("cam0", ""),
                    "CAM_FRONT_LEFT": data.get("cam1", ""),
                    "CAM_FRONT_RIGHT": data.get("cam2", ""),
                    "CAM_BACK_LEFT": data.get("cam3", ""),
                    "CAM_BACK_RIGHT": data.get("cam4", ""),
                    "CAM_BACK": data.get("cam5", "")
                },
                "session_id": data.get("session_id", ""),
                "prev_frame_token": frame_ids[i-1] if i > 0 else None,
                "next_frame_token": frame_ids[i+1] if i < len(frame_ids)-1 else None,
                "objects": []
            }
            
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        label_objects = json.load(f)
                    frame["objects"] = [
                        {
                            "obj_id": obj.get("obj_id"),
                            "obj_type": obj.get("obj_type"),
                            "position": obj.get("psr", {}).get("position"),
                            "rotation": obj.get("psr", {}).get("rotation"),
                            "scale": obj.get("psr", {}).get("scale")
                        }
                        for obj in label_objects
                    ]
                except Exception:
                    pass
            
            frames.append(frame)
        
        out_path = os.path.join(data_loader.annot_out, 'frames.json')
        with open(out_path, 'w') as f:
            json.dump(frames, f, indent=2)
        
        log_handler.log(f"✓ Annotations converted ({len(frames)} frames)", 'success')


class IDD3DSampleConverter(BaseConverter):
    """Convert IDD3D frames to nuScenes sample.json format"""
    
    def __init__(self, sequence_name: str = 'seq'):
        super().__init__('sample')
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        import uuid
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for sample conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        # Read scene.json to get scene_token
        scene_path = os.path.join(data_loader.annot_out, 'scene.json')
        scene_token = uuid.uuid4().hex
        if os.path.exists(scene_path):
            try:
                with open(scene_path, 'r') as f:
                    scenes = json.load(f)
                    if scenes and len(scenes) > 0:
                        scene_token = scenes[0]['token']
            except Exception:
                pass
        
        samples = []
        
        for i, frame_id in enumerate(frame_ids):
            # Use frame_id as token
            token = frame_id
            
            # Generate timestamp from frame_id (multiplied to match nuScenes format)
            timestamp = int(frame_id) * 100000
            
            # prev and next tokens
            prev = frame_ids[i-1] if i > 0 else ""
            next_token = frame_ids[i+1] if i < len(frame_ids)-1 else ""
            
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
        
        log_handler.log(f"✓ Sample file created with {len(samples)} samples", 'success')


class IDD3DSampleAnnotationConverter(BaseConverter):
    """Convert IDD3D object annotations to nuScenes sample_annotation.json format"""
    
    def __init__(self, sequence_name: str = 'seq'):
        super().__init__('sample_annotation')
        self.sequence_name = sequence_name
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        import uuid
        
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for sample_annotation conversion", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("No frames found", 'warning')
            return
        
        sample_annotations = []
        # Track objects across frames for instance tokens and prev/next linking
        object_instances = {}  # {obj_id: {'instance_token': ..., 'annotations': [...]}}
        
        # First pass: collect all annotations and assign instance tokens
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
                    
                    # Create or get instance token for this object
                    if obj_id not in object_instances:
                        object_instances[obj_id] = {
                            'instance_token': uuid.uuid4().hex,
                            'annotations': []
                        }
                    
                    # Generate annotation token
                    ann_token = uuid.uuid4().hex
                    
                    # Extract position, rotation, scale from PSR
                    psr = obj.get("psr", {})
                    position = psr.get("position", {})
                    rotation = psr.get("rotation", {})
                    scale = psr.get("scale", {})
                    
                    # Convert to nuScenes format
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
                    
                    # Convert rotation (Euler to quaternion - simplified)
                    # For now, using rotation values as-is, may need proper conversion
                    rotation_quat = [
                        rotation.get("x", 0.0),
                        rotation.get("y", 0.0),
                        rotation.get("z", 0.0),
                        1.0  # w component
                    ]
                    
                    annotation = {
                        "token": ann_token,
                        "sample_token": frame_id,
                        "instance_token": object_instances[obj_id]['instance_token'],
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
        
        # Second pass: link prev/next for each instance
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


class IDD3DCategoryConverter(BaseConverter):
    """Generate nuScenes category.json from IDD3D object types"""
    
    def __init__(self):
        super().__init__('category')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        import uuid
        
        # IDD3D object types mapped to nuScenes-like categories
        # Based on IDD3D's 17 classes
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
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
        
        # Collect unique object types from all label files
        unique_obj_types = set()
        frame_ids = sorted(data_loader.read_annotations().keys())
        
        for frame_id in frame_ids:
            label_path = os.path.join(data_loader.label_dir, f"{frame_id}.json")
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    label_objects = json.load(f)
                for obj in label_objects:
                    obj_type = obj.get("obj_type")
                    if obj_type:
                        unique_obj_types.add(obj_type)
            except Exception:
                pass
        
        # Create category entries
        categories = []
        for obj_type in sorted(unique_obj_types):
            # Map IDD3D type to nuScenes category name
            category_name = idd3d_to_nuscenes_categories.get(obj_type, f'movable_object.{obj_type.lower()}')
            
            category = {
                "token": uuid.uuid4().hex,
                "name": category_name,
                "description": f"{obj_type} category from IDD3D"
            }
            categories.append(category)
        
        # Save category.json
        out_path = os.path.join(data_loader.annot_out, 'category.json')
        with open(out_path, 'w') as f:
            json.dump(categories, f, indent=2)
        
        log_handler.log(f"✓ Category file created with {len(categories)} categories", 'success')


class IDD3DInstanceConverter(BaseConverter):
    """Generate nuScenes instance.json from label files"""
    
    def __init__(self):
        super().__init__('instance')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        import uuid
        
        # Read category.json to get category tokens
        category_path = os.path.join(data_loader.annot_out, 'category.json')
        category_map = {}  # {category_name: token}
        if os.path.exists(category_path):
            try:
                with open(category_path, 'r') as f:
                    categories = json.load(f)
                for cat in categories:
                    category_map[cat['name']] = cat['token']
            except Exception:
                pass
        
        # IDD3D to nuScenes category mapping
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
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
        
        # Read annot_data.json to get frame information
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        # Track unique instances across all frames
        instance_tracker = {}  # {obj_id: {'instance_token': ..., 'category_token': ..., 'frames': []}}
        
        # First pass: collect all objects and their appearances
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
                        # Generate instance token
                        instance_token = f"instance_{obj_id}"
                        
                        # Get category token
                        category_name = idd3d_to_nuscenes_categories.get(obj_type, f'movable_object.{obj_type.lower()}')
                        category_token = category_map.get(category_name, uuid.uuid4().hex)
                        
                        instance_tracker[obj_id] = {
                            'instance_token': instance_token,
                            'category_token': category_token,
                            'first_frame': frame_id,
                            'last_frame': frame_id,
                            'first_annotation_token': f"obj_{frame_id}_{obj_id}",
                            'last_annotation_token': f"obj_{frame_id}_{obj_id}"
                        }
                    else:
                        # Update last frame for this instance
                        instance_tracker[obj_id]['last_frame'] = frame_id
                        instance_tracker[obj_id]['last_annotation_token'] = f"obj_{frame_id}_{obj_id}"
                        
            except Exception as e:
                log_handler.log(f"Error processing label {frame_id}: {str(e)}", 'warning')
        
        # Create instance entries
        instances = []
        for obj_id, data in instance_tracker.items():
            instance = {
                "instance_token": data['instance_token'],
                "category_token": data['category_token'],
                "first_annotation_token": data['first_annotation_token'],
                "last_annotation_token": data['last_annotation_token'],
                "first_frame_token": f"frame_{data['first_frame']}",
                "last_frame_token": f"frame_{data['last_frame']}"
            }
            instances.append(instance)
        
        # Save instance.json
        out_path = os.path.join(data_loader.annot_out, 'instance.json')
        with open(out_path, 'w') as f:
            json.dump(instances, f, indent=2)
        
        log_handler.log(f"Instance file created with {len(instances)} instances", 'success')


class IDD3DTimestampSyncConverter(BaseConverter):
    """Synchronize timestamps across all JSON files"""
    
    def __init__(self):
        super().__init__('timestamp_sync')
    
    def run(self, data_loader: IDD3DDataLoader, log_handler: LogHandler):
        """
        Ensure consistent timestamps across:
        - sample.json
        - frames.json (if exists)
        - Any other files that need timestamps
        """
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("No annotations found for timestamp sync", 'warning')
            return
        
        frame_ids = sorted(annot_data.keys())
        
        # Create a timestamp mapping: frame_id -> timestamp
        timestamp_map = {}
        base_timestamp = 1532402927647951  # Example base timestamp (can be adjusted)
        
        for i, frame_id in enumerate(frame_ids):
            # Generate consistent timestamp (100ms intervals = 10Hz)
            timestamp = base_timestamp + (i * 100000)
            timestamp_map[frame_id] = timestamp
        
        log_handler.log(f"Generated timestamps for {len(timestamp_map)} frames", 'info')
        
        # Update sample.json with synced timestamps
        sample_path = os.path.join(data_loader.annot_out, 'sample.json')
        if os.path.exists(sample_path):
            try:
                with open(sample_path, 'r') as f:
                    samples = json.load(f)
                
                for sample in samples:
                    frame_id = sample['token']
                    if frame_id in timestamp_map:
                        sample['timestamp'] = timestamp_map[frame_id]
                
                with open(sample_path, 'w') as f:
                    json.dump(samples, f, indent=2)
                
                log_handler.log("Updated timestamps in sample.json", 'success')
            except Exception as e:
                log_handler.log(f"Error updating sample.json timestamps: {str(e)}", 'warning')
        
        # Update frames.json with synced timestamps (if it exists)
        frames_path = os.path.join(data_loader.annot_out, 'frames.json')
        if os.path.exists(frames_path):
            try:
                with open(frames_path, 'r') as f:
                    frames = json.load(f)
                
                for frame in frames:
                    frame_id = frame['frame_id']
                    if frame_id in timestamp_map:
                        frame['timestamp'] = timestamp_map[frame_id]
                
                with open(frames_path, 'w') as f:
                    json.dump(frames, f, indent=2)
                
                log_handler.log("✓ Updated timestamps in frames.json", 'success')
            except Exception as e:
                log_handler.log(f"Error updating frames.json timestamps: {str(e)}", 'warning')
        
        log_handler.log("Timestamp synchronization complete", 'success')



# CONVERTER REGISTRY - Easy to add new datasets
 
class ConverterRegistry:
    """Registry for dataset conversions"""
    
    _conversions = {}
    
    @classmethod
    def register(cls, source: str, target: str, pipeline_builder):
        """Register a conversion pipeline. pipeline_builder is a callable that returns a DatasetConversionPipeline"""
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
    """Build conversion pipeline for IDD3D -> nuScenes"""
    pipeline = DatasetConversionPipeline('idd3d', 'nuscenes')
    
    conversions = config.get('conversions', {})
    sequence_name = config.get('sequence_id', 'seq_10')
    
    if conversions.get('lidar', False):
        pipeline.add_converter(IDD3DLidarConverter())
    if conversions.get('camera', False):
        pipeline.add_converter(IDD3DCameraConverter())
    if conversions.get('calib', False):
        pipeline.add_converter(IDD3DCalibConverter())
    if conversions.get('annot', False):
        pipeline.add_converter(IDD3DAnnotationConverter(sequence_name))
    if conversions.get('sample', False):
        pipeline.add_converter(IDD3DSampleConverter(sequence_name))
    if conversions.get('sample_annotation', False):
        pipeline.add_converter(IDD3DSampleAnnotationConverter(sequence_name))
    if conversions.get('category', False):
        pipeline.add_converter(IDD3DCategoryConverter())
    if conversions.get('instance', False):
        pipeline.add_converter(IDD3DInstanceConverter())
    
    # Always run timestamp sync at the end to ensure consistency
    pipeline.add_converter(IDD3DTimestampSyncConverter())
    
    return pipeline


# Register the conversion AFTER the function is defined
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
                log_handler.log("✓ Conversion pipeline completed successfully!", 'success')
                log_handler.log(f"Output directory: {root_path}/Intermediate_format/", 'info')
        
        except Exception as e:
            log_handler.log(f"✗ Conversion failed: {str(e)}", 'error')
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