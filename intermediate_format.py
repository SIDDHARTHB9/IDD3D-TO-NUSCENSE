from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class IFCalibration:
    sensor_name: str                      
    translation: List[float]              
    rotation: List[float]                 
    camera_intrinsic: List[List[float]] = field(default_factory=list)  
    
    def __post_init__(self):
        """Validate calibration data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        if self.camera_intrinsic:
            assert len(self.camera_intrinsic) == 3, "Camera intrinsic must be 3x3 matrix"
            assert all(len(row) == 3 for row in self.camera_intrinsic), "Each row must have 3 elements"


@dataclass
class IFEgoPose:
    temp_frame_id: str                    
    timestamp_us: int                     
    translation: List[float]              
    rotation: List[float]  

    def __post_init__(self):
        """Validate ego pose data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"


@dataclass
class IFInstance:
    temp_instance_id: str                 
    category_name: str                    
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert self.temp_instance_id, "Instance ID cannot be empty"
        assert self.category_name, "Category name cannot be empty"


@dataclass
class IFAnnotation:
    temp_instance_id: str                 
    temp_frame_id: str                    
    timestamp_us: int                     
    translation: List[float]              
    size: List[float]                     
    rotation: List[float]                 
    attributes: List[str] = field(default_factory=list)  
    num_lidar_pts: int = 0               
    num_radar_pts: int = 0                
    visibility: float = 1.0               
    
    def __post_init__(self):
        """Validate annotation data."""
        assert len(self.translation) == 3, "Translation must be [x, y, z]"
        assert len(self.size) == 3, "Size must be [width, length, height]"
        assert len(self.rotation) == 4, "Rotation must be quaternion [w, x, y, z]"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"
        assert 0.0 <= self.visibility <= 1.0, "Visibility must be in range [0.0, 1.0]"


@dataclass
class IFSensorData:
    temp_frame_id: str                    
    sensor_name: str                      
    original_filename: str                
                                          
    timestamp_us: int                     
    is_keyframe: bool = True            
    
    width: Optional[int] = None
    height: Optional[int] = None
    
    def __post_init__(self):
        assert self.temp_frame_id, "Frame ID cannot be empty"
        assert self.sensor_name, "Sensor name cannot be empty"
        assert self.original_filename, "Original filename cannot be empty"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"


@dataclass
class IFSample:

    temp_frame_id: str                    
    timestamp_us: int                     
    scene_name: str                       
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sample data."""
        assert self.temp_frame_id, "Frame ID cannot be empty"
        assert self.timestamp_us >= 0, "Timestamp must be non-negative"
        assert self.scene_name, "Scene name cannot be empty"


@dataclass
class IFScene:
    name: str                             
    description: str                     
    location: Optional[str] = None        
    date_captured: Optional[str] = None   
    weather: Optional[str] = None         
    time_of_day: Optional[str] = None     
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert self.name, "Scene name cannot be empty"


@dataclass
class IntermediateData:
    scenes: List[IFScene] = field(default_factory=list)
    samples: List[IFSample] = field(default_factory=list)
    sensor_data: List[IFSensorData] = field(default_factory=list)
    annotations: List[IFAnnotation] = field(default_factory=list)
    instances: List[IFInstance] = field(default_factory=list)
    calibrations: List[IFCalibration] = field(default_factory=list)
    ego_poses: List[IFEgoPose] = field(default_factory=list)
    
    sequence_path: str = ""               
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.samples and not self.scenes:
            raise ValueError("Cannot have samples without at least one scene")
    
    def validate(self) -> bool:
        if not self.scenes:
            raise ValueError("IntermediateData must contain at least one scene")
        
        if not self.samples:
            raise ValueError("IntermediateData must contain at least one sample")
        
        if not self.sequence_path:
            raise ValueError("sequence_path must be set")
        
        scene_names = {scene.name for scene in self.scenes}
        for sample in self.samples:
            if sample.scene_name not in scene_names:
                raise ValueError(f"Sample references unknown scene: {sample.scene_name}")
        
        sample_frame_ids = {sample.temp_frame_id for sample in self.samples}
        
        for sensor_data in self.sensor_data:
            if sensor_data.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"SensorData references unknown frame: {sensor_data.temp_frame_id}")
        
        for annotation in self.annotations:
            if annotation.temp_frame_id not in sample_frame_ids:
                raise ValueError(f"Annotation references unknown frame: {annotation.temp_frame_id}")
        
        instance_ids = {inst.temp_instance_id for inst in self.instances}
        
        for annotation in self.annotations:
            if annotation.temp_instance_id not in instance_ids:
                raise ValueError(f"Annotation references unknown instance: {annotation.temp_instance_id}")
        
        return True
    
    def summary(self) -> str:
        return f"""
IntermediateData Summary:
========================
Sequence Path:  {self.sequence_path}
Scenes:         {len(self.scenes)}
Samples:        {len(self.samples)}
Instances:      {len(self.instances)}
Annotations:    {len(self.annotations)}
Sensor Data:    {len(self.sensor_data)}
Calibrations:   {len(self.calibrations)}
Ego Poses:      {len(self.ego_poses)}

Scene Names:    {', '.join(scene.name for scene in self.scenes)}
Sensors:        {', '.join(sorted(set(cal.sensor_name for cal in self.calibrations)))}
Categories:     {', '.join(sorted(set(inst.category_name for inst in self.instances)))}
"""


def create_empty_intermediate_data(sequence_path: str) -> IntermediateData:
    return IntermediateData(sequence_path=sequence_path)


def validate_category_name(category_name: str) -> bool:
    if not category_name or '.' not in category_name:
        return False
    
    parts = category_name.split('.')
    if len(parts) != 2:
        return False
    
    group, specific = parts
    
    valid_groups = {
        'vehicle', 'human', 'movable_object', 
        'static_object', 'animal', 'flat'
    }
    return group in valid_groups and specific.strip() != ''