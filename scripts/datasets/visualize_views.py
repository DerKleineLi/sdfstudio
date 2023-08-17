import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from visualization import VisOpen3D

dataset_list = [
    ["/home/hli/Data/data/sdfstudio-demo-data", "replica-room0"],
    ["/home/hli/Data/data/scannetpp_sdf2", "2022-11-19_11-22"],
    ["/home/hli/Data/data/scannet", "scan1"],
    ["/home/hli/Data/data/TNT", "Meetingroom_sdf_crop"],
]

# out_dir = Path("/home/hli/Data/data/sdfstudio-demo-data")
# scenes = ["replica-room0"]

selection = 2
out_dir = Path(dataset_list[selection][0])
scenes = [dataset_list[selection][1]]
image_size = 384


def process_one_scene(scene):
    # Replace these with your actual file paths and camera parameters
    scene_out_dir = out_dir / scene
    meta_file = scene_out_dir / "meta_data.json"

    with open(meta_file, "r") as f:
        meta_dict = json.load(f)

    dataset = {}
    splits = ["frames"] + (["test_frames"] if "test_frames" in meta_dict else [])
    poses = []
    for split in splits:
        dataset[split] = {}
        for frame in meta_dict[split]:
            key = os.path.basename(frame["rgb_path"])[:-8]
            intrinsic = np.array(frame["intrinsics"])
            extrinsic = np.array(frame["camtoworld"])
            poses.append(extrinsic[:3, 3])
            extrinsic = np.linalg.inv(extrinsic)
            dataset[split][key] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
            }
    poses = np.array(poses)
    print(f"bbox_size: {poses.max(0)-poses.min(0)}")

    vis = VisOpen3D(width=image_size, height=image_size)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # vis.add_geometry(axis)
    for frame in dataset["frames"].values():
        vis.draw_camera(frame["intrinsic"], frame["extrinsic"], scale=0.05, plane_scale=0.5)
    if "test_frames" in splits:
        for frame in dataset["test_frames"].values():
            vis.draw_camera(frame["intrinsic"], frame["extrinsic"], scale=0.05, plane_scale=0.5, color=[0.2, 0.8, 0.2])
    vis.reset_view_point()
    aabb = o3d.geometry.AxisAlignedBoundingBox(-np.ones((3, 1)), np.ones((3, 1)))
    aabb.color = np.array([1, 0, 0])
    vis.add_geometry(aabb)
    vis.reset_view_point()

    vis.run()


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene)
