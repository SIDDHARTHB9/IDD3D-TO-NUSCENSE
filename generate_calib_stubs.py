
"""Thin wrapper that delegates to the OOP CalibStubConverter."""

import os
from idd3d_converter_oop import CalibStubConverter


def main():
    default_root = '/home/siddharthb9/Desktop/nuSceneses&IDD3D'
    calib_dir = os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/calib')
    out_data = os.path.join(default_root, 'Intermediate_format/data')
    converter = CalibStubConverter(calib_dir, out_data)
    from idd3d_converter_oop import DataLoader
    dl = DataLoader(default_root)
    dl.ensure_output_dirs()
    converter.run(dl)


if __name__ == '__main__':
    main()