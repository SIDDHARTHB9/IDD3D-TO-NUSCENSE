'''
Create a folder as per your choice and put all the .pcd.bin files in it.
Change the folder_path variable below to that folder path.
Run the script. It will open a window and display the point clouds one by one.
Press 'q' to exit the window.
'''
import numpy as np
from glob import glob
import open3d as o3d
import os
import time

def load_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if points.shape[0] == 0:
        print(f"Skipping empty file: {bin_path}")
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

if __name__ == "__main__":
    folder_path = "/home/siddharthb9/Desktop/nuSceneses&IDD3D/Intermediate_format/data/converted_lidar"
    pcd_files = sorted(glob(os.path.join(folder_path, "*.pcd.bin")))

    if not pcd_files:
        print("No PCD files found in the specified folder.")
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Lidar .pcd.bin Visualization", width=1280, height=960)

        pcd = None  
        for idx, pcd_file_path in enumerate(pcd_files):
            new_pcd = load_bin(pcd_file_path)
            if new_pcd is None:
                continue

            print(f"Displaying frame {idx + 1}/{len(pcd_files)}: {pcd_file_path}")

            if pcd is None:
                pcd = new_pcd
                vis.add_geometry(pcd)
            else: 
                pcd.points = new_pcd.points
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)

        vis.destroy_window()


'''
For reading and checking the single file of .pcd.bin format

import numpy as np
import open3d as o3d

def load_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

pcd = load_bin("/home/siddharthb9/Desktop/nuSceneses&IDD3D/Intermediate_format/data/converted_lidar/01067.pcd.bin")
o3d.visualization.draw_geometries([pcd])
'''