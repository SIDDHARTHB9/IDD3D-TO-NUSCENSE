"""Thin wrapper that delegates to the OOP converter's AnnotationConverter.

This script keeps the original entrypoint but centralizes conversion
logic in `idd3d_converter_oop.py`.
"""

import os
from idd3d_converter_oop import AnnotationConverter


def main():
    default_root = '/home/siddharthb9/Desktop/nuSceneses&IDD3D'
    annot_json = os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/annot_data.json')
    label_folder = os.path.join(default_root, 'idd3d_dataset_seq10 (c)/idd3d_seq10/train_val/20220118103308_seq_10/label')
    out_dir = os.path.join(default_root, 'Intermediate_format/anotations')
    converter = AnnotationConverter(annot_json, label_folder, out_dir, sequence_name='seq_10')
    from idd3d_converter_oop import DataLoader
    dl = DataLoader(default_root)
    dl.ensure_output_dirs()
    converter.run(dl)


if __name__ == '__main__':
    main()
