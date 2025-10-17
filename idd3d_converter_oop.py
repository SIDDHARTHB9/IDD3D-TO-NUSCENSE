"""
This script provides a small class-based framework so future conversions
and dataset/target adapters are easy to add.

It currently implements three converters based on the existing scripts:
 - LidarConverter: converts .pcd -> .pcd.bin (uses open3d if available)
 - CalibStubConverter: generates calibrated_sensor.json and sensors.json
 - AnnotationConverter: converts per-frame annotation JSONs to an
   intermediate `frames.json` file.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from typing import List, Optional
import numpy as np
import open3d as o3d

class BaseConverter:
    """Abstract converter base class."""

    def __init__(self, name: str):
        self.name = name
        self.dry_run = False

    def prepare(self):
        """Prepare output directories or validate inputs."""
        pass

    def run(self, data_loader: Optional[DataLoader] = None):
        """Execute conversion. Must be implemented by subclasses."""
        raise NotImplementedError()


class DataLoader:
    """Simple dataset loader that centralizes paths and basic IO for IDD3D.

    Purpose: provide a single place to discover files and read datasets so
    converters don't duplicate path-handling logic.
    """

    def __init__(self, root: str, sequence: str = '20220118103308_seq_10'):
        self.root = os.path.abspath(root)
        # default sequence-specific paths
        self.sequence = sequence
        self.seq_base = os.path.join(self.root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val', sequence)
        self.lidar_dir = os.path.join(self.seq_base, 'lidar')
        self.label_dir = os.path.join(self.seq_base, 'label')
        self.calib_dir = os.path.join(self.seq_base, 'calib')
        self.annot_json = os.path.join(self.seq_base, 'annot_data.json')

        # intermediate output locations under repository
        self.out_data = os.path.join(self.root, 'Intermediate_format/data')
        self.annot_out = os.path.join(self.root, 'Intermediate_format/anotations')
        # store converted lidar in the intermediate data folder under converted_lidar
        self.converted_lidar = os.path.join(self.out_data, 'converted_lidar')

    def list_lidar_files(self):
        if not os.path.exists(self.lidar_dir):
            return []
        return [os.path.join(self.lidar_dir, f) for f in sorted(os.listdir(self.lidar_dir)) if f.lower().endswith('.pcd')]

    def read_annotations(self):
        if not os.path.exists(self.annot_json):
            return {}
        with open(self.annot_json, 'r') as f:
            return json.load(f)

    def label_path(self, frame_id: str):
        return os.path.join(self.label_dir, f"{frame_id}.json")

    def ensure_output_dirs(self):
        os.makedirs(self.out_data, exist_ok=True)
        os.makedirs(self.annot_out, exist_ok=True)
        os.makedirs(self.converted_lidar, exist_ok=True)



class LidarConverter(BaseConverter):
    """Convert IDD3D PCD files to nuScenes .pcd.bin files.

    If open3d is not installed, the converter will still copy filenames and
    write empty bin files (smoke-mode) so the pipeline can be tested.
    """

    def __init__(self, src_dir: str, dst_dir: str):
        super().__init__('lidar')
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        os.makedirs(self.dst_dir, exist_ok=True)

    def _has_open3d(self) -> bool:
        return o3d is not None

    def run(self, data_loader: Optional[DataLoader] = None):
        # require both open3d and numpy for full conversion
        use_o3d = (o3d is not None and np is not None)
        # allow data_loader to override path discovery
        if data_loader is not None:
            files = [os.path.basename(p) for p in data_loader.list_lidar_files()]
            src_dir = data_loader.lidar_dir
            dst_dir = data_loader.converted_lidar
        else:
            if not os.path.exists(self.src_dir):
                print(f"Lidar source dir does not exist: {self.src_dir}")
                return
            files = [f for f in os.listdir(self.src_dir) if f.lower().endswith('.pcd')]
            print(f"Found {len(files)} pcd files in {self.src_dir}")
            src_dir = self.src_dir
            dst_dir = self.dst_dir

        total = len(files)
        if total == 0:
            print("No lidar files to convert.")
            return

        os.makedirs(dst_dir, exist_ok=True)
        converted = 0
        placeholders = 0
        errors = 0

        for i, fname in enumerate(files):
            src = os.path.join(src_dir, fname)
            base = os.path.splitext(fname)[0]
            dst = os.path.join(dst_dir, base + '.pcd.bin')
            try:
                if use_o3d:
                    pcd = o3d.io.read_point_cloud(src)
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    # no intensity - append zeros
                    if xyz.size == 0:
                        pts = xyz.astype(np.float32)
                    else:
                        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                        pts = np.hstack((xyz, intensity))
                    pts.astype(np.float32).tofile(dst)
                    converted += 1
                else:
                    # write an empty placeholder to allow downstream steps to run
                    open(dst, 'wb').close()
                    placeholders += 1
            except Exception:
                # write placeholder on failure to preserve pipeline
                try:
                    open(dst, 'wb').close()
                    placeholders += 1
                except Exception:
                    errors += 1

            # progress line (overwrites same terminal line)
            percent = (i + 1) / total * 100
            sys.stdout.write(f"\rConverting lidar: [{i+1}/{total}] {percent:5.1f}%  ({converted} converted, {placeholders} placeholders, {errors} errors)")
            sys.stdout.flush()

        # finish progress output with newline and short summary
        print()
        print(f"Finished lidar conversion: {converted} converted, {placeholders} placeholders, {errors} errors. Output directory: {dst_dir}")


class CalibStubConverter(BaseConverter):
    """Generate nuScenes-style calibration stubs and copy into intermediate folder."""

    def __init__(self, calib_dir: str, out_data_dir: str, sensors: Optional[List[str]] = None):
        super().__init__('calib')
        self.calib_dir = os.path.abspath(calib_dir)
        self.out_data_dir = os.path.abspath(out_data_dir)
        self.sensors = sensors or ['Lidar', 'cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5']
        os.makedirs(self.calib_dir, exist_ok=True)

    @staticmethod
    def _make_quat_identity():
        return [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def _make_translation_default(sensor_name: str):
        if sensor_name.upper().startswith('LIDAR'):
            return [0.0, 0.0, 1.8]
        return [0.0, 0.0, 1.6]

    def run(self, data_loader: Optional[DataLoader] = None):
        calibrated_list = []
        sensors_j = []
        for s in self.sensors:
            token = uuid.uuid4().hex
            sensor_token = uuid.uuid4().hex
            entry = {
                "token": token,
                "sensor_token": sensor_token,
                "translation": self._make_translation_default(s),
                "rotation": self._make_quat_identity(),
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

        def write_json(path, obj):
            if self.dry_run:
                print(f"[dry-run] would write {path} ({len(obj) if isinstance(obj, list) else 'obj'})")
                return
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
            print(f'Wrote {path}')

        # allow data loader to override output locations
        if data_loader is not None:
            cal_dir = data_loader.calib_dir
            out_calib_dir = os.path.join(data_loader.out_data, 'calibration')
        else:
            cal_dir = self.calib_dir
            out_calib_dir = os.path.join(self.out_data_dir, 'calibration')

        cal_path = os.path.join(cal_dir, 'calibrated_sensor.json')
        sensors_path = os.path.join(cal_dir, 'sensors.json')
        write_json(cal_path, calibrated_list)
        write_json(sensors_path, sensors_j)

        os.makedirs(out_calib_dir, exist_ok=True)
        # copy stubs into the standard calibration output folder
        with open(cal_path, 'r') as fsrc, open(os.path.join(out_calib_dir, 'calibrated_sensor.json'), 'w') as fdst:
            fdst.write(fsrc.read())
        with open(sensors_path, 'r') as fsrc, open(os.path.join(out_calib_dir, 'sensors.json'), 'w') as fdst:
            fdst.write(fsrc.read())
        print(f'Copied stubs into {out_calib_dir}')

        # also store a visible copy under Intermediate_format/data/calibstubs
        if data_loader is not None:
            calibstubs_dir = os.path.join(data_loader.out_data, 'calibstubs')
        else:
            calibstubs_dir = os.path.join(self.out_data_dir, 'calibstubs')
        os.makedirs(calibstubs_dir, exist_ok=True)
        with open(cal_path, 'r') as fsrc, open(os.path.join(calibstubs_dir, 'calibrated_sensor.json'), 'w') as fdst:
            fdst.write(fsrc.read())
        with open(sensors_path, 'r') as fsrc, open(os.path.join(calibstubs_dir, 'sensors.json'), 'w') as fdst:
            fdst.write(fsrc.read())
        print(f'Copied stubs into {calibstubs_dir}')


class AnnotationConverter(BaseConverter):
    """Convert per-frame annotation JSONs into an `frames.json` intermediate file."""

    def __init__(self, annot_json: str, label_folder: str, output_dir: str, sequence_name: str = 'seq'):
        super().__init__('annot')
        self.annot_json = os.path.abspath(annot_json)
        self.label_folder = os.path.abspath(label_folder)
        self.output_dir = os.path.abspath(output_dir)
        self.sequence_name = sequence_name
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_json(self, path: str):
        with open(path, 'r') as f:
            return json.load(f)

    def run(self, data_loader: Optional[DataLoader] = None):
        # prefer loader-provided annotations where available
        if data_loader is not None:
            annot_data = data_loader.read_annotations()
            label_folder = data_loader.label_dir
            out_dir = data_loader.annot_out
        else:
            if not os.path.exists(self.annot_json):
                print(f"Annotation file not found: {self.annot_json}")
                return
            annot_data = self._load_json(self.annot_json)
            label_folder = self.label_folder
            out_dir = self.output_dir
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
            label_path = os.path.join(label_folder, f"{frame_id}.json")
            if os.path.exists(label_path):
                try:
                    label_objects = self._load_json(label_path)
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
                    frame["objects"] = []
            frames.append(frame)

        out_path = os.path.join(out_dir, 'frames.json')
        if self.dry_run:
            print(f"[dry-run] would write frames.json to {out_path} ({len(frames)} frames)")
        else:
            with open(out_path, 'w') as f:
                json.dump(frames, f, indent=2)
            print(f'Wrote frames.json to {out_path} ({len(frames)} frames)')


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Run IDD3D -> nuScenes intermediate converters (OOP)')
    parser.add_argument('--lidar', action='store_true', help='Run lidar conversion')
    parser.add_argument('--calib', action='store_true', help='Generate calibration stubs')
    parser.add_argument('--annot', action='store_true', help='Run annotation conversion')
    parser.add_argument('--run-all', action='store_true', help='Run all converters')
    parser.add_argument('--verbose', action='store_true')
    

    # Common/default paths - adjust as needed or pass explicit env vars
    default_root = '/home/siddharthb9/Desktop/nuSceneses&IDD3D'
    parser.add_argument('--lidar-src', default=os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/lidar'))
    parser.add_argument('--lidar-dst', default=os.path.join(default_root, 'Intermediate_format/data/converted_lidar'))

    parser.add_argument('--calib-dir', default=os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/calib'))
    parser.add_argument('--out-data', default=os.path.join(default_root, 'Intermediate_format/data'))

    parser.add_argument('--annot-json', default=os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/annot_data.json'))
    parser.add_argument('--label-folder', default=os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/label'))
    parser.add_argument('--annot-out', default=os.path.join(default_root, 'Intermediate_format/anotations'))

    args = parser.parse_args(argv)

    # create a shared DataLoader and ensure output dirs exist
    data_loader = DataLoader(default_root)
    data_loader.ensure_output_dirs()

    to_run = []
    if args.run_all or args.lidar:
        to_run.append(('lidar', LidarConverter(args.lidar_src, args.lidar_dst)))
    if args.run_all or args.calib:
        to_run.append(('calib', CalibStubConverter(args.calib_dir, args.out_data)))
    if args.run_all or args.annot:
        to_run.append(('annot', AnnotationConverter(args.annot_json, args.label_folder, args.annot_out, sequence_name='seq_10')))

    if not to_run:
        # Default to running all converters when none were selected
        to_run = [
            ('lidar', LidarConverter(args.lidar_src, args.lidar_dst)),
            ('calib', CalibStubConverter(args.calib_dir, args.out_data)),
            ('annot', AnnotationConverter(args.annot_json, args.label_folder, args.annot_out, sequence_name='seq_10'))
        ]

    for key, conv in to_run:
        print(f'Running converter: {key} ({conv.__class__.__name__})')
        try:
            conv.run(data_loader)
        except Exception as e:
            print(f'Converter {key} failed: {e}')


if __name__ == '__main__':
    main()