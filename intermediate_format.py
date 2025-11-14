# intermediate_format.py
# ----------------------
# This file defines the "Intermediate Format" (IF) for data exchange.
#
# - READERS (e.g., Idd3dReader, ArgoverseReader) are responsible FOR CREATING
#   an instance of the main 'IntermediateData' class.
#
# - WRITERS (e.g., NuScenesWriter, KittiWriter) are responsible FOR CONSUMING
#   an instance of the 'IntermediateData' class.
#
# All 'temp_..._id' fields are strings used by the Reader to temporarily
# identify objects (e.g., "frame_0001", "obj_123"). The Writer is
# responsible for converting these into its own token/ID system.
# ----------------------
 
from dataclasses import dataclass, field
from typing import List, Dict, Any
 
@dataclass
class IFCalibration:
    """
    Defines the calibration (extrinsics and intrinsics) for a single sensor.
    """
    sensor_name: str                  # The channel/name of the sensor (e.D., "CAM_FRONT", "LIDAR_TOP")
    translation: List[float]          # Extrinsic translation [x, y, z] relative to the ego vehicle origin
    rotation: List[float]             # Extrinsic rotation (quaternion) [w, x, y, z]
    camera_intrinsic: List[List[float]] = field(default_factory=list)  # 3x3 intrinsic matrix, if a camera
 
@dataclass
class IFEgoPose:
    """
    Defines the ego vehicle's pose at a specific timestamp.
    """
    temp_frame_id: str                # The temporary ID for the frame this pose belongs to
    timestamp_us: int                 # Timestamp in microseconds
    translation: List[float]          # [x, y, z] in global coordinates
    rotation: List[float]             # Quaternion [w, x, y, z] in global coordinates
 
@dataclass
class IFInstance:
    """
    Defines a unique object instance (a track).
    """
    temp_instance_id: str             # The Reader's unique ID for this track (e.g., "obj_123")
    category_name: str                # The *standardized* category name (e.g., "vehicle.car")
 
@dataclass
class IFAnnotation:
    """
    Defines a single bounding box annotation for one instance in one frame.
    """
    temp_instance_id: str             # The ID of the instance being annotated
    temp_frame_id: str                # The ID of the frame this annotation is in
    translation: List[float]          # [x, y, z] in the global coordinate frame
    timestamp_us: int                 # Timestamp in microseconds
    rotation: List[float]             # Quaternion [w, x, y, z]
    size: List[float]                 # [width, length, height]
    attributes: List[str] = field(default_factory=list)  # Standardized attributes (e.g., "vehicle.moving")
 
@dataclass
class IFSensorData:
    """
    Represents a single piece of sensor data (e.g., one LiDAR scan, one camera image).
    """
    temp_frame_id: str                # The frame ID this sensor data belongs to
    sensor_name: str                  # The channel of the sensor (e.g., "LIDAR_TOP")
    
    # The ORIGINAL filename from the source dataset (e.g., "00000.pcd", "cam0/00000.png")
    # The Reader provides this, and the Writer uses it to find the source file.
    original_filename: str
    
    timestamp_us: int                 # Timestamp in microseconds
    is_keyframe: bool = True          # Whether this is a keyframe (for nuScenes)
 
@dataclass
class IFSample:
    """
    Represents a "keyframe" or "sample" in time, linking all sensor data
    and annotations for a single timestamp.
    """
    temp_frame_id: str                # The Reader's unique ID for this frame (e.g., "00000")
    timestamp_us: int                 # Timestamp in microseconds
    scene_name: str                   # The name of the scene this sample belongs to
 
@dataclass
class IFScene:
    """
    Defines a single "scene" or "log" (a continuous driving sequence).
    """
    name: str                         # The unique name for this scene (e.g., "idd3d_seq10")
    description: str                  # A human-readable description
 
@dataclass
class IntermediateData:
    """
    This is the main "package" of data passed from the Reader to the Writer.
    It contains all the processed information from the source dataset.
    """
    # --- Lists of objects ---
    scenes: List[IFScene] = field(default_factory=list)
    samples: List[IFSample] = field(default_factory=list)
    sensor_data: List[IFSensorData] = field(default_factory=list)
    annotations: List[IFAnnotation] = field(default_factory=list)
    instances: List[IFInstance] = field(default_factory=list)
    calibrations: List[IFCalibration] = field(default_factory=list)
    ego_poses: List[IFEgoPose] = field(default_factory=list)
    
    # --- File/Path Information ---
    # The Reader must populate these paths so the Writer knows
    # where to find the original physical files (e.g., .pcd, .png).
    
    # Absolute path to the root of the sequence being processed
    sequence_path: str = ""
 