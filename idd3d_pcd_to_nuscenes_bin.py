"""Thin wrapper to call LidarConverter from the OOP module.

This script keeps the original CLI entrypoint but delegates actual
conversion to `idd3d_converter_oop.LidarConverter` for maintainability.
"""

import os
from idd3d_converter_oop import LidarConverter


def main():
    default_root = '/home/siddharthb9/Desktop/nuSceneses&IDD3D'
    src = os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/lidar')
    dst = os.path.join(default_root, 'Intermediate_format/data/converted_lidar')
    converter = LidarConverter(src, dst)
    from idd3d_converter_oop import DataLoader
    dl = DataLoader(default_root)
    dl.ensure_output_dirs()
    converter.run(dl)


if __name__ == '__main__':
    main()
