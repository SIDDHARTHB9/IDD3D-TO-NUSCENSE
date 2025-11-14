# writers.py
# ----------------------
# This file contains "Writer" classes.
# Each Writer is responsible for consuming the 'IntermediateData' object
# and writing it to a specific dataset format (like nuScenes).
# ----------------------

import os
import json
import shutil
import logging
import uuid  # <-- Used for map expansion and prediction
import re    # <-- ADDED for scene name formatting
import hashlib # <-- ADDED for scene name formatting
from abc import ABC, abstractmethod
from PIL import Image
from datetime import datetime
from intermediate_format import IntermediateData
from utils import TokenTimestampManager, append_to_json_list, json_file_lock, merge_and_overwrite_json_list

# --- MODIFIED: Added pyarrow and pandas ---
try:
    import numpy as np
    import pyarrow.feather as pf
    import pandas as pd
except ImportError:
    print("WARNING: numpy, pyarrow or pandas not found. Argoverse LiDAR conversion will fail.")
    print("Please run: pip install numpy pyarrow pandas")

try:
    import open3d as o3d
except ImportError:
    print("WARNING: open3d not found. IDD3D .pcd conversion will fail.")
    print("Please run: pip install open3d")
# --- END MODIFIED ---


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- File Conversion Helpers (from old IDD3DLidar/Camera Converters) ---
def convert_lidar_file(src_path, dst_path):
    """
    Converts a .pcd file to a .pcd.bin file.
    If open3d is not available, creates an empty placeholder file.
    """
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close(); return
        pcd = o3d.io.read_point_cloud(src_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack((xyz, intensity))
        pts.astype(np.float32).tofile(dst_path)
    except Exception as e:
        log.error(f"Error converting {src_path} (is open3d installed?): {e}. Creating empty file.")
        open(dst_path, 'wb').close()

def convert_camera_file(src_path, dst_path):
    """
    Converts a .png or .jpg file to a .jpg file.
    If PIL is not available, does nothing.
    """
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source camera file not found: {src_path}")
            return
        img = Image.open(src_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_path, 'JPEG', quality=95)
    except ImportError:
        log.warning(f"PIL/Pillow not available. Skipping camera conversion for {src_path}")
    except Exception as e:
        log.error(f"Error converting {src_path} to {dst_path}: {e}")

# --- NEW HELPER FUNCTION FOR ARGOVERSE ---
def convert_feather_to_pcd_bin(src_path, dst_path):
    """
    Converts Argoverse .feather LiDAR file to .pcd.bin format.
    """
    try:
        if not os.path.exists(src_path):
            log.warning(f"Source LiDAR file not found: {src_path}")
            open(dst_path, 'wb').close(); return
        
        table = pf.read_feather(src_path)
        df = table.to_pandas() # Columns: 'x', 'y', 'z', 'intensity', ...
        
        # We only need x, y, z, and intensity for nuScenes format
        # Note: nuScenes is [x, y, z, intensity]
        pts = df[['x', 'y', 'z', 'intensity']].values.astype(np.float32)
        
        pts.astype(np.float32).tofile(dst_path)
        
    except Exception as e:
        log.error(f"Error converting {src_path} (is pyarrow/pandas installed?): {e}. Creating empty file.")
        open(dst_path, 'wb').close()
# --- END NEW ---

# -----------------------------------------------------------------------------
#  BASE WRITER
# -----------------------------------------------------------------------------
class BaseWriter(ABC):
    @abstractmethod
    def write(self, data: IntermediateData, output_path: str):
        pass

# -----------------------------------------------------------------------------
#  NUSCENES WRITER
# -----------------------------------------------------------------------------
class NuScenesWriter(BaseWriter):
    """
    Writes data to the nuScenes dataset format.
    Handles merging and linking data across multiple runs.
    """
    
    def __init__(self):
        self.token_manager = None
        self.output_path = None
        self.annot_out_dir = None
        self.samples_out_dir = None
        self.sweeps_out_dir = None
        self.maps_out_dir = None
        self.map_expansion_dir = None 
        self.map_expansion_basemap_dir = None 
        self.map_expansion_expansion_dir = None 
        self.map_expansion_prediction_dir = None 
        
        # --- Holders for cross-run data ---
        self.generated_log_tokens = []
        self.all_sample_annotations = []
        self.instance_db = {} # Holds all instance data

    def write(self, data: IntermediateData, output_path: str):
        log.info(f"Initializing NuScenesWriter for output to: {output_path}")
        self.output_path = os.path.abspath(output_path)
        
        # --- 1. Setup Output Directories ---
        self.annot_out_dir = os.path.join(self.output_path, 'anotations')
        self.samples_out_dir = os.path.join(self.output_path, 'samples')
        self.sweeps_out_dir = os.path.join(self.output_path, 'sweeps')
        self.maps_out_dir = os.path.join(self.output_path, 'maps')

        os.makedirs(self.annot_out_dir, exist_ok=True)
        os.makedirs(self.samples_out_dir, exist_ok=True)
        os.makedirs(self.maps_out_dir, exist_ok=True)

        # --- NEW: Create Map Expansion Dirs ---
        self.map_expansion_dir = os.path.join(self.output_path, 'idd3d_map_expansion')
        self.map_expansion_basemap_dir = os.path.join(self.map_expansion_dir, 'basemap')
        self.map_expansion_expansion_dir = os.path.join(self.map_expansion_dir, 'expansion')
        self.map_expansion_prediction_dir = os.path.join(self.map_expansion_dir, 'prediction')

        os.makedirs(self.map_expansion_basemap_dir, exist_ok=True)
        os.makedirs(self.map_expansion_expansion_dir, exist_ok=True)
        os.makedirs(self.map_expansion_prediction_dir, exist_ok=True)
        # --- END NEW ---

        # --- 2. Initialize TokenManager ---
        registry_path = os.path.join(self.annot_out_dir, 'token_registry.json')
        last_timestamp = self._get_last_timestamp()
        new_base_timestamp = (last_timestamp + 20_000_000) if last_timestamp else None # 20-sec gap
        
        self.token_manager = TokenTimestampManager(
            registry_path=registry_path,
            base_timestamp=new_base_timestamp
        )

        self._pre_populate_categories()

        if not data.scenes:
            log.error("No scenes found in intermediate data. Cannot proceed."); return
        sequence_name = data.scenes[0].name
        log.info(f"Processing sequence: {sequence_name}")

        # --- 3. Run Writing Tasks in Order ---
        log.info("Writing JSON metadata files...")
        
        self._write_sensor_and_calib(data.calibrations)
        self._write_visibility()
        self._write_attribute()
        
        self._write_log(data.scenes)
        self._write_map() # Creates hyderabad.png
        self._write_map_expansion() 
        self._write_prediction(data.scenes, data.samples)
        self._write_file_manifest(data) 

        self._write_sample_and_ego_pose(data.samples, data.ego_poses)
        self._write_sample_data(data.sensor_data, sequence_name)
        
        self._write_category(data.instances)
        self._write_instance_and_annotation(data.instances, data.annotations)
        
        # --- 4. Process Physical Files ---
        log.info("Converting and copying physical sensor files...")
        self._process_sensor_files(data.sensor_data, data.sequence_path, sequence_name)
        
        # --- 5. Duplicate Sweeps ---
        log.info("Duplicating 'samples' directory to 'sweeps'...")
        self._duplicate_sweeps()
        
        # --- 6. Save Token Registry ---
        log.info("Saving global token registry...")
        self.token_manager.save_registry(registry_path)
        
        log.info(f"--- NuScenes Write Complete ---")
        log.info(f"Output successfully written to: {self.output_path}")

    def _get_last_timestamp(self):
        sample_json_path = os.path.join(self.annot_out_dir, 'sample.json')
        last_timestamp = None
        if os.path.exists(sample_json_path):
            with json_file_lock:
                try:
                    with open(sample_json_path, 'r') as f:
                        samples = json.load(f)
                        if samples and isinstance(samples, list):
                            last_timestamp = samples[-1].get('timestamp')
                except Exception as e:
                    log.warning(f"Could not read last timestamp: {e}")
        if last_timestamp: log.info(f"Found existing data. Last timestamp: {last_timestamp}")
        return last_timestamp

    # --- NEW HELPER for scene name formatting ---
    def _format_scene_name(self, raw_scene_name: str) -> str:
        """
        Converts a sequence name like 'idd3d_seq10'
        into a nuScenes-style 'scene-NNNN' format.
        """
        # Try to extract digits from the end of the name
        match = re.search(r'\d+$', raw_scene_name)
        if match:
            num_str = match.group(0)
            # Format to 4 digits, e.g., "10" -> "0010"
            return f"scene-{num_str.zfill(4)}"
        else:
            # Fallback for names without numbers
            fallback_hash = hashlib.md5(raw_scene_name.encode()).hexdigest()[:4]
            log.warning(f"Could not parse number from scene '{raw_scene_name}'. Using fallback name 'scene-{fallback_hash}'")
            return f"scene-{fallback_hash}"

    def _pre_populate_categories(self):
        """
        Manually injects the user's official category tokens
        into the TokenTimestampManager to ensure consistency.
        """
        log.info("Pre-populating TokenManager with official category tokens...")
        
        official_categories = [
          {
            "token": "dc39d8b2858e4bc0b7ddf66ede8d734e",
            "name": "vehicle.motorcycle",
            "description": "vehicle.motorcycle category"
          },
          {
            "token": "d411b4e8157d445193034d6f408900d3",
            "name": "movable_object.bicyclerider",
            "description": "movable_object.bicyclerider category"
          },
          {
            "token": "e2325ce5697e45678ee0fe4017918290",
            "name": "movable_object.tourcar",
            "description": "movable_object.tourcar category"
          },
          {
            "token": "9a438c7df65d4ae0b5e87f603a3e91b7",
            "name": "movable_object.scooterrider",
            "description": "movable_object.scooterrider category"
          },
          {
            "token": "1046b59779f24cf7b55114161208b0f5",
            "name": "vehicle.bus",
            "description": "vehicle.bus category"
          },
          {
            "token": "57c2b779b57b496297048ea55aaed2c7",
            "name": "movable_object.bicyclegroup",
            "description": "movable_object.bicyclegroup category"
          },
          {
            "token": "869140488b264d7780ed9cc8233cb5ce",
            "name": "movable_object.van",
            "description": "movable_object.van category"
          },
          {
            "token": "69d88d0df8274f56995aacff1982ec65",
            "name": "vehicle.truck",
            "description": "vehicle.truck category"
          },
          {
            "token": "9a6c42f9792f40789bc0437eba0aef9b",
            "name": "movable_object.pedestrian",
            "description": "movable_object.pedestrian category"
          },
          {
            "token": "f15d03bf64834024a0601aae7a07c156",
            "name": "movable_object.scooter",
            "description": "movable_object.scooter category"
          },
          {
            "token": "366ad39f728a4ab5ae9a4146f528bd00",
            "name": "vehicle.bicycle",
            "description": "vehicle.bicycle category"
          },
          {
            "token": "f0add8f1828d4b7ca20d135edd7ecd4e",
            "name": "movable_object.unknown",
            "description": "movable_object.unknown category"
          },
          {
            "token": "6bc7bdefe76646e193288d5928a2d58a",
            "name": "movable_object.unknown1",
            "description": "movable_object.unknown1 category"
          },
          {
            "token": "3305eeb43e684538b00bcc41fc38d84e",
            "name": "vehicle.car",
            "description": "vehicle.car category"
          }
        ]
        
        count = 0
        for category in official_categories:
            cat_name = category['name']
            cat_token = category['token']
            if cat_name not in self.token_manager.category_tokens:
                self.token_manager.category_tokens[cat_name] = cat_token
                count += 1
        
        log.info(f"Injected {count} new official category tokens into TokenManager.")


    # --- JSON Writing Methods (called by write()) ---

    def _write_sensor_and_calib(self, calibrations):
        new_sensors = []
        new_calib_sensors = []
        
        for if_calib in calibrations:
            sensor_token = self.token_manager.get_sensor_token(if_calib.sensor_name)
            is_camera = len(if_calib.camera_intrinsic) > 0
            
            new_sensors.append({
                "token": sensor_token,
                "modality": "camera" if is_camera else "lidar",
                "channel": if_calib.sensor_name,
            })
            
            new_calib_sensors.append({
                "token": self.token_manager.get_calibration_token(if_calib.sensor_name),
                "sensor_token": sensor_token,
                "translation": if_calib.translation,
                "rotation": if_calib.rotation,
                "camera_intrinsic": if_calib.camera_intrinsic
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'sensor.json'), 
            new_sensors, 
            key_field='channel'
        )
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'calibrated_sensor.json'), 
            new_calib_sensors, 
            key_field='sensor_token'
        )

    def _write_sample_and_ego_pose(self, samples, ego_poses):
        sample_path = os.path.join(self.annot_out_dir, 'sample.json')
        ego_pose_path = os.path.join(self.annot_out_dir, 'ego_pose.json')
        
        all_samples = []
        all_ego_poses = []

        with json_file_lock:
            if os.path.exists(sample_path):
                try: all_samples = json.load(open(sample_path, 'r'))
                except: log.warning("sample.json corrupted. Overwriting.")
            if os.path.exists(ego_pose_path):
                try: all_ego_poses = json.load(open(ego_pose_path, 'r'))
                except: log.warning("ego_pose.json corrupted. Overwriting.")
        
        for if_sample in samples:
            all_samples.append({
                "token": self.token_manager.get_frame_token(if_sample.temp_frame_id),
                "timestamp": if_sample.timestamp_us,
                "scene_token": self.token_manager.get_scene_token() 
            })
        
        for if_pose in ego_poses:
            all_ego_poses.append({
                "token": self.token_manager.get_ego_pose_token(if_pose.temp_frame_id),
                "timestamp": if_pose.timestamp_us,
                "translation": if_pose.translation,
                "rotation": if_pose.rotation
            })

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
        
        with json_file_lock:
            json.dump(final_samples, open(sample_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample.json. Total items: {len(final_samples)}")
            json.dump(all_ego_poses, open(ego_pose_path, 'w'), indent=2)
            log.info(f"Merged and overwrote ego_pose.json. Total items: {len(all_ego_poses)}")
        
        if samples:
            # --- MODIFIED: Use formatted scene name ---
            raw_scene_name = samples[0].scene_name
            formatted_scene_name = self._format_scene_name(raw_scene_name)
            
            new_scene = {
                "token": self.token_manager.get_scene_token(),
                "log_token": self.generated_log_tokens[-1] if self.generated_log_tokens else "",
                "nbr_samples": len(samples),
                "first_sample_token": self.token_manager.get_frame_token(samples[0].temp_frame_id),
                "last_sample_token": self.token_manager.get_frame_token(samples[-1].temp_frame_id),
                "name": formatted_scene_name, # <-- FIXED
                "description": f"Scene {raw_scene_name}" # Keep original name in description
            }
            append_to_json_list(os.path.join(self.annot_out_dir, 'scene.json'), [new_scene])


    def _write_sample_data(self, sensor_data, sequence_name):
        sample_data_path = os.path.join(self.annot_out_dir, 'sample_data.json')
        
        all_sample_data = []
        with json_file_lock:
            if os.path.exists(sample_data_path):
                try: all_sample_data = json.load(open(sample_data_path, 'r'))
                except: log.warning("sample_data.json corrupted. Overwriting.")

        for if_data in sensor_data:
            sd_token = uuid.uuid4().hex
            is_camera = if_data.sensor_name.startswith("CAM_")
            timestamp = if_data.timestamp_us
            output_filename_base = f"{sequence_name}_frame_{timestamp}"
            
            if is_camera:
                output_filename = f"{output_filename_base}.jpg"
                fileformat = "jpg"
            else:
                if if_data.original_filename.endswith(".feather"):
                    output_filename = f"{output_filename_base}.pcd.bin"
                    fileformat = "pcd.bin"
                else: 
                    output_filename = f"{output_filename_base}.pcd.bin"
                    fileformat = "pcd.bin"

            all_sample_data.append({
                "token": sd_token,
                "sample_token": self.token_manager.get_frame_token(if_data.temp_frame_id),
                "ego_pose_token": self.token_manager.get_ego_pose_token(if_data.temp_frame_id),
                "calibrated_sensor_token": self.token_manager.get_calibration_token(if_data.sensor_name),
                "filename": f"samples/{if_data.sensor_name}/{output_filename}",
                "fileformat": fileformat,
                "width": 1440 if is_camera else 0,
                "height": 1080 if is_camera else 0,
                "timestamp": if_data.timestamp_us,
                "is_key_frame": if_data.is_keyframe,
            })
        
        sensor_groups = {}
        for sd in all_sample_data:
            token = sd['calibrated_sensor_token']
            if token not in sensor_groups: sensor_groups[token] = []
            sensor_groups[token].append(sd)

        final_sample_data = []
        for sensor_token, sd_list in sensor_groups.items():
            sorted_list = sorted(sd_list, key=lambda x: x['timestamp'])
            for i, sd in enumerate(sorted_list):
                sd['prev'] = sorted_list[i-1]['token'] if i > 0 else ""
                sd['next'] = sorted_list[i+1]['token'] if i < len(sorted_list) - 1 else ""
            final_sample_data.extend(sorted_list)
        
        with json_file_lock:
            json.dump(final_sample_data, open(sample_data_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample_data.json. Total items: {len(final_sample_data)}")

    def _write_category(self, instances):
        new_categories = []
        all_category_names_from_data = {inst.category_name for inst in instances}
        
        for name, token in self.token_manager.category_tokens.items():
            new_categories.append({
                "token": token,
                "name": name,
                "description": f"{name} category"
            })
        
        for name in all_category_names_from_data:
            if name not in self.token_manager.category_tokens:
                token = self.token_manager.get_category_token(name)
                new_categories.append({
                    "token": token,
                    "name": name,
                    "description": f"{name} category"
                })

        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'category.json'), 
            new_categories, 
            key_field='name' 
        )

    def _write_instance_and_annotation(self, instances, annotations):
        instance_path = os.path.join(self.annot_out_dir, 'instance.json')
        ann_path = os.path.join(self.annot_out_dir, 'sample_annotation.json')

        with json_file_lock:
            try:
                all_anns = json.load(open(ann_path, 'r')) if os.path.exists(ann_path) else []
            except: all_anns = []; log.warning("sample_annotation.json corrupted.")
            
            try:
                inst_list = json.load(open(instance_path, 'r')) if os.path.exists(instance_path) else []
                inst_db = {i['token']: i for i in inst_list}
            except: inst_db = {}; log.warning("instance.json corrupted.")

        new_anns_by_inst_id = {}
        for ann in annotations:
            if ann.temp_instance_id not in new_anns_by_inst_id:
                new_anns_by_inst_id[ann.temp_instance_id] = []
            new_anns_by_inst_id[ann.temp_instance_id].append(ann)

        inst_name_map = {inst.temp_instance_id: inst.category_name for inst in instances}
        
        used_category_tokens = set()
        for inst in inst_db.values():
            used_category_tokens.add(inst['category_token'])

        for temp_inst_id, new_anns_list in new_anns_by_inst_id.items():
            inst_token = self.token_manager.get_instance_token(temp_inst_id)
            new_anns_list.sort(key=lambda x: x.timestamp_us)
            
            last_ann_token_from_existing = ""
            if inst_token in inst_db:
                last_ann_token_from_existing = inst_db[inst_token]['last_annotation_token']

            generated_tokens = [self.token_manager.generate_annotation_token() for _ in new_anns_list]
            
            for i, if_ann in enumerate(new_anns_list):
                category_name = inst_name_map.get(temp_inst_id, "")
                attribute_tokens = []
                if category_name.startswith('vehicle.'):
                    attribute_tokens = [self.token_manager.get_attribute_token("vehicle.moving")]
                elif category_name.startswith('human.'):
                    attribute_tokens = [self.token_manager.get_attribute_token("pedestrian.moving")]

                ann_token = generated_tokens[i]
                prev_token = generated_tokens[i-1] if i > 0 else last_ann_token_from_existing
                next_token = generated_tokens[i+1] if i < len(generated_tokens) - 1 else ""
                
                all_anns.append({
                    "token": ann_token,
                    "sample_token": self.token_manager.get_frame_token(if_ann.temp_frame_id),
                    "instance_token": inst_token,
                    "attribute_tokens": attribute_tokens,
                    "visibility_token": self.token_manager.get_visibility_token("v4-0"),
                    "translation": if_ann.translation,
                    "size": if_ann.size,
                    "rotation": if_ann.rotation,
                    "prev": prev_token, "next": next_token,
                    "num_lidar_pts": 0, "num_radar_pts": 0
                })

            category_token = self.token_manager.get_category_token(inst_name_map.get(temp_inst_id, ""))
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
        
        log.info("Checking for unused categories to create dummy instances...")
        dummy_instance_count = 0
        
        for cat_name, cat_token in self.token_manager.category_tokens.items():
            if cat_token not in used_category_tokens:
                dummy_instance_token = self.token_manager.get_instance_token(f"dummy_instance_for_{cat_name}")
                
                if dummy_instance_token not in inst_db: 
                    inst_db[dummy_instance_token] = {
                        "token": dummy_instance_token,
                        "category_token": cat_token,
                        "nbr_annotations": 0,
                        "first_annotation_token": dummy_instance_token,
                        "last_annotation_token": dummy_instance_token
                    }
                    dummy_instance_count += 1
        
        if dummy_instance_count > 0:
            log.info(f"Created {dummy_instance_count} dummy instances to satisfy devkit (for unused categories).")

        with json_file_lock:
            json.dump(list(inst_db.values()), open(instance_path, 'w'), indent=2)
            log.info(f"Merged and overwrote instance.json. Total items: {len(inst_db)}")
            json.dump(all_anns, open(ann_path, 'w'), indent=2)
            log.info(f"Merged and overwrote sample_annotation.json. Total items: {len(all_anns)}")


    def _write_visibility(self):
        vis_levels = [
            {"level": "v1-0", "description": "visibility 0-40%"},
            {"level": "v2-0", "description": "visibility 40-60%"},
            {"level": "v3-0", "description": "visibility 60-80%"},
            {"level": "v4-0", "description": "visibility 80-100%"}
        ]
        new_entries = []
        for vis in vis_levels:
            new_entries.append({
                "token": self.token_manager.get_visibility_token(vis["level"]),
                "level": vis["level"],
                "description": vis["description"]
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'visibility.json'),
            new_entries,
            key_field='level'
        )

    def _write_attribute(self):
        attributes = [
            {"name": "vehicle.moving", "description": "Vehicle is moving (default stub)"},
            {"name": "pedestrian.moving", "description": "Pedestrian is moving (default stub)"},
        ]
        new_entries = []
        for attr in attributes:
            new_entries.append({
                "token": self.token_manager.get_attribute_token(attr["name"]),
                "name": attr["name"],
                "description": attr["description"]
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'attribute.json'),
            new_entries,
            key_field='name'
        )

    def _write_map(self):
        location = "Hyderabad"
        map_filename = f"maps/{location.lower()}.png"
        map_token = self.token_manager.get_map_token(f"map_{location}")
        
        new_map_entry = {
            "token": map_token,
            "log_tokens": self.generated_log_tokens, 
            "category": "semantic_prior",
            "filename": map_filename,
        }
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'map.json'),
            [new_map_entry],
            key_field='token'
        )
        
        # --- Create the main map image ---
        image_path = os.path.join(self.maps_out_dir, f"{location.lower()}.png")
        if not os.path.exists(image_path):
            try:
                img = Image.new('RGB', (10, 10), color='black')
                img.save(image_path, 'PNG')
                log.info(f"Created dummy map file: {image_path}")
            except Exception as e:
                log.error(f"Could not create dummy map image: {e}")

        # --- NEW: Copy this png to the map_expansion/basemap folder ---
        basemap_image_path = os.path.join(self.map_expansion_basemap_dir, f"{location.lower()}.png")
        if not os.path.exists(basemap_image_path):
            try:
                shutil.copyfile(image_path, basemap_image_path)
                log.info(f"Copied map basemap to: {basemap_image_path}")
            except Exception as e:
                log.error(f"Could not copy basemap image: {e}")


    def _write_log(self, scenes):
        new_entries = []
        for if_scene in scenes:
            logfile = f"{if_scene.name}-{datetime.now().strftime('%Y-%m-%d')}"
            log_token = self.token_manager.get_log_token(f"log_{logfile}") 
            
            self.generated_log_tokens.append(log_token)
            
            new_entries.append({
                "token": log_token,
                "logfile": logfile,
                "vehicle": "stub_vehicle",
                "date_captured": datetime.now().strftime('%Y-%m-%d'),
                "location": "Hyderabad"
            })
        
        merge_and_overwrite_json_list(
            os.path.join(self.annot_out_dir, 'log.json'),
            new_entries,
            key_field='token'
        )

    def _write_file_manifest(self, data: IntermediateData):
        new_entries = []
        frame_to_sensor_data = {}
        for sd in data.sensor_data:
            if sd.temp_frame_id not in frame_to_sensor_data:
                frame_to_sensor_data[sd.temp_frame_id] = []
            frame_to_sensor_data[sd.temp_frame_id].append(sd)

        for if_sample in data.samples:
            frame_id = if_sample.temp_frame_id
            sequence_name = if_sample.scene_name
            manifest_entry = {
                "frame_id": frame_id,
                "sequence": sequence_name,
                "sample_token": self.token_manager.get_frame_token(frame_id),
                "sensors": []
            }
            if frame_id not in frame_to_sensor_data: continue
            for sd in frame_to_sensor_data[frame_id]:
                timestamp = sd.timestamp_us
                output_filename_base = f"{sequence_name}_frame_{timestamp}"
                if sd.sensor_name.startswith("CAM_"):
                    output_filename = f"{output_filename_base}.jpg"
                    source_file = f"{sequence_name}/camera/{sd.original_filename}"
                else: # LIDAR
                    output_filename = f"{output_filename_base}.pcd.bin"
                    if sd.original_filename.endswith('.feather'):
                         source_file = f"{sequence_name}/lidar/{sd.original_filename}"
                    else: 
                         source_file = f"{sequence_name}/lidar/{sd.original_filename}"

                manifest_entry["sensors"].append({
                    "channel": sd.sensor_name,
                    "source_file": source_file,
                    "output_file": f"samples/{sd.sensor_name}/{output_filename}"
                })
            new_entries.append(manifest_entry)
            
        append_to_json_list(os.path.join(self.annot_out_dir, 'file_manifest.json'), new_entries)

    # --- REWRITTEN: Map Expansion Stub Method ---
    def _write_map_expansion(self):
        """
        Creates a stubbed singapore-queenstown.json map expansion file,
        matching the complex structure with polygon and node keys.
        """
        log.info("Creating stubbed map expansion file...")
        # --- FIXED: New filename ---
        expansion_path = os.path.join(self.map_expansion_expansion_dir, "singapore-queenstown.json")

        # Create a handful of nodes
        node_tokens = [uuid.uuid4().hex for _ in range(4)]
        nodes = [
            {"token": node_tokens[0], "x": 10.0, "y": 10.0},
            {"token": node_tokens[1], "x": 10.0, "y": -10.0},
            {"token": node_tokens[2], "x": -10.0, "y": -10.0},
            {"token": node_tokens[3], "x": -10.0, "y": 10.0}
        ]
        
        # Create one polygon
        poly_token = uuid.uuid4().hex
        polygons = [
            {
                "token": poly_token,
                "exterior_node_tokens": node_tokens,
                "holes": []
            }
        ]
        
        # Create lane dividers
        divider1_token = uuid.uuid4().hex
        divider2_token = uuid.uuid4().hex
        lane_dividers = [
            {
                "token": divider1_token,
                "line_token": uuid.uuid4().hex, # Stub
                "lane_divider_segments": [
                    {
                        "node_token": node_tokens[0],
                        "segment_type": "DOUBLE_DASHED_WHITE"
                    }
                ]
            },
            {
                "token": divider2_token,
                "line_token": uuid.uuid4().hex, # Stub
                "lane_divider_segments": [
                    {
                        "node_token": node_tokens[1],
                        "segment_type": "DOUBLE_DASHED_WHITE"
                    }
                ]
            }
        ]

        # Create one lane
        lanes = [
            {
                "token": uuid.uuid4().hex,
                "polygon_token": poly_token,
                "lane_type": "car",
                "from_edge_line_token": divider1_token,
                "to_edge_line_token": divider2_token,
                "left_lane_divider_segment": [], # Stub
                "right_lane_divider_segment": [] # Stub
            }
        ]
        
        # Create one road segment
        road_seg_token = uuid.uuid4().hex
        road_segments = [
            {
                "token": road_seg_token,
                "polygon_token": poly_token,
                "is_intersection": False
            }
        ]
        
        # Create one drivable area
        drivable_areas = [
            {
                "token": uuid.uuid4().hex,
                "road_segment_tokens": [road_seg_token]
            }
        ]

        # Assemble the final JSON structure
        stub_data = {
            "polygon": polygons,
            "node": nodes,
            "lane": lanes,
            "lane_divider_segment": lane_dividers,
            "road_segment": road_segments,
            "drivable_area": drivable_areas,
            "traffic_control": [] # Stub empty traffic control
        }

        try:
            # This file is a static stub, so we just overwrite it
            with open(expansion_path, 'w') as f:
                json.dump(stub_data, f, indent=2)
            log.info(f"Created stub map expansion file at: {expansion_path}")
        except Exception as e:
            log.error(f"FATAL: Could not write map expansion file: {e}")
            raise
            
    # --- NEW: Prediction Stub Method ---
    def _write_prediction(self, scenes, samples):
        """
        Creates a stubbed prediction.json file, merging scene entries.
        """
        if not scenes or not samples:
            log.warning("No scenes or samples found, skipping prediction.json.")
            return

        prediction_path = os.path.join(self.map_expansion_prediction_dir, "prediction.json")
        
        # Load existing prediction data
        prediction_data = {}
        with json_file_lock:
            if os.path.exists(prediction_path):
                try:
                    with open(prediction_path, 'r') as f:
                        prediction_data = json.load(f)
                    if not isinstance(prediction_data, dict):
                        log.warning("prediction.json is not a dictionary. Overwriting.")
                        prediction_data = {}
                except json.JSONDecodeError:
                    log.warning("prediction.json is corrupted. Overwriting.")
                    prediction_data = {}
            
            # --- Create new entry for this scene ---
            # --- MODIFIED: Use formatted scene name ---
            raw_scene_name = scenes[0].name
            formatted_scene_name = self._format_scene_name(raw_scene_name)
            
            # Find the first sample for this scene
            first_sample = min(samples, key=lambda x: x.timestamp_us)
            first_sample_token = self.token_manager.get_frame_token(first_sample.temp_frame_id)
            
            # --- FIXED: Create a LIST of stubbed predictions ---
            stubbed_predictions = []
            for _ in range(3): # Create 3 dummy predictions
                prediction_id = uuid.uuid4().hex
                prediction_string = f"{prediction_id}_{first_sample_token}"
                stubbed_predictions.append(prediction_string)
            
            # Add to the dictionary
            prediction_data[formatted_scene_name] = stubbed_predictions # <-- FIXED
            
            # Write the updated dictionary back to the file
            try:
                with open(prediction_path, 'w') as f:
                    json.dump(prediction_data, f, indent=2)
                log.info(f"Merged scene '{formatted_scene_name}' into prediction.json.")
            except Exception as e:
                log.error(f"FATAL: Could not write prediction.json: {e}")
                raise


    # --- File Processing Methods (called by write()) ---
    
    def _process_sensor_files(self, sensor_data, sequence_path, sequence_name):
        num_lidar = 0
        num_camera = 0
        
        for sd in sensor_data:
            timestamp = sd.timestamp_us
            output_filename_base = f"{sequence_name}_frame_{timestamp}"
            
            # --- MODIFIED: Handle .pcd, .feather, and images ---
            if sd.original_filename.endswith('.pcd'):
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                output_filename = f"{output_filename_base}.pcd.bin"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_lidar_file(src_file, dst_file)
                    num_lidar += 1
            
            elif sd.original_filename.endswith('.feather'):
                src_file = os.path.join(sequence_path, 'lidar', sd.original_filename)
                output_filename = f"{output_filename_base}.pcd.bin"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_feather_to_pcd_bin(src_file, dst_file) # <-- Use new helper
                    num_lidar += 1
            
            else: # It's a camera
                # Source filename can be "cam0/00000.png" (IDD3D) or "ring_front_center/1234.jpg" (AV2)
                src_file = os.path.join(sequence_path, 'camera', sd.original_filename)
                output_filename = f"{output_filename_base}.jpg"
                dst_folder = os.path.join(self.samples_out_dir, sd.sensor_name)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file = os.path.join(dst_folder, output_filename)
                
                if not os.path.exists(dst_file):
                    convert_camera_file(src_file, dst_file)
                    num_camera += 1
                
        log.info(f"Processed {num_lidar} new LiDAR files and {num_camera} new camera files.")

    def _duplicate_sweeps(self):
        if os.path.exists(self.sweeps_out_dir):
            try: shutil.rmtree(self.sweeps_out_dir)
            except Exception as e:
                log.error(f"Could not remove 'sweeps' directory: {e}")
                return
        try:
            shutil.copytree(self.samples_out_dir, self.sweeps_out_dir)
            log.info(f"Successfully duplicated 'samples' to 'sweeps'.")
        except Exception as e:
            log.error(f"FATAL: Could not copy 'samples' to 'sweeps': {e}")
            raise