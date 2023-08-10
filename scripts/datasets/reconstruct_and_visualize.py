import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage import filters
from visualization import VisOpen3D

out_dir = Path("/home/hli/Data/data/scannetpp_sdf2")
in_dir = Path("/mnt/c/Users/li/Documents")
original_in_dir = Path("/menegroth/scannetpp/data")
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2022-11-30_13-44"]
image_size = 384


def load_rgbd_image(rgb_file, depth_file, min_depth=0.015):
    # Load RGB and depth images
    rgb = o3d.io.read_image(str(rgb_file))
    depth = np.load(depth_file)
    mask = depth < min_depth
    # edges = filters.sobel(depth)
    # mask += edges > 0.05
    # for _ in range(5):
    #     mask = ndimage.binary_dilation(mask)
    depth[mask] = -1
    return rgb, o3d.geometry.Image(depth)


def create_point_cloud(rgb, depth, intrinsic_matrix, extrinsic_matrix, depth_scale=1):
    # Create a point cloud from the RGBD images
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(
        image_size,
        image_size,
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=depth_scale, depth_trunc=10, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic, extrinsic_matrix)

    return pcd


def create_combined_point_cloud(dataset, depth_scale=1, visualize_process=False):
    pcd_combined = o3d.geometry.PointCloud()

    if visualize_process:
        vis = VisOpen3D(width=image_size, height=image_size)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(axis)

    for key, value in dataset["frames"].items():
        # if key not in ["DSC00008", "DSC09834"]:
        #     continue
        print("processing: " + key)
        rgb, depth = load_rgbd_image(value["rgb_file"], value["depth_file"])
        pcd = create_point_cloud(rgb, depth, value["intrinsic"], value["extrinsic"], depth_scale)

        # # Transform the point cloud using the camera pose
        # pcd.transform(value["extrinsic"])

        # Combine the individual point clouds into a single point cloud
        pcd_combined += pcd

        if visualize_process:
            vis.add_geometry(pcd)
            vis.render()
            # input()

    return pcd_combined


def process_one_scene(scene):
    # Replace these with your actual file paths and camera parameters
    scene_dir = in_dir / scene
    scene_out_dir = out_dir / scene
    meta_file = scene_out_dir / "meta_data.json"
    transform_file = scene_dir / "dslr" / "nerfstudio" / "transforms.json"

    with open(meta_file, "r") as f:
        meta_dict = json.load(f)
    with open(transform_file, "r") as f:
        trafo_dict = json.load(f)

    scale = 1  # / meta_dict["worldtogt"][0][0]

    dataset = {}
    for split in ["frames", "test_frames"]:
        dataset[split] = {}
        for frame in meta_dict[split]:
            key = os.path.basename(frame["rgb_path"])[:-8]
            intrinsic = np.array(frame["intrinsics"])
            extrinsic = np.array(frame["camtoworld"])
            extrinsic = np.linalg.inv(extrinsic)
            rgb_file = scene_out_dir / frame["rgb_path"]
            depth_file = scene_out_dir / frame["sensor_depth_path"]
            dataset[split][key] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "rgb_file": rgb_file,
                "depth_file": depth_file,
            }
        # for frame in trafo_dict[split]:
        #     key = os.path.basename(frame["file_path"])[:-4]
        #     extrinsic = np.array(frame["transform_matrix"])

        #     pre_transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        #     pre_transform2 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #     pre_transform3 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        #     # extrinsic = pre_transform3 @ pre_transform2 @ extrinsic

        #     extrinsic = np.linalg.inv(extrinsic)
        #     extrinsic = pre_transform3 @ pre_transform2 @ extrinsic

        #     dataset[split][key].update({"extrinsic": extrinsic})

    pcd = create_combined_point_cloud(dataset, scale)

    vis = VisOpen3D(width=image_size, height=image_size)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(axis)
    vis.add_geometry(pcd)
    vis.run()

    os.makedirs(scene_out_dir / "reproject", exist_ok=True)
    o3d.io.write_point_cloud(str(scene_out_dir / "reproject" / "combined.ply"), pcd)


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene)
