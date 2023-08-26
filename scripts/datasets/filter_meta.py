import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh

out_dir = Path("/mnt/c/Users/li/Documents")
in_dir = Path("/mnt/c/Users/li/Documents")
original_in_dir = Path("/menegroth/scannetpp/data")
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2023-05-20_17-35"]
image_size = 384
boundry_extend = 5


def filter_frame(frames, file_list):
    frames_filtered = []
    for frame in frames:
        if frame["rgb_path"] in file_list:
            frames_filtered.append(frame)
    return frames_filtered


def process_one_scene(scene):
    # Replace these with your actual file paths and camera parameters
    scene_dir = in_dir / scene
    scene_out_dir = out_dir / scene
    transform_file = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    meta_file = scene_out_dir / "meta_data.json"

    with open(meta_file, "r") as f:
        meta_dict = json.load(f)

    file_list = os.listdir(scene_out_dir)
    file_list_dict = {"file_list": file_list}
    with open(scene_out_dir / "file_list.json", "w") as f:
        json.dump(file_list_dict, f, indent=4)

    meta_dict["frames"] = filter_frame(meta_dict["frames"], file_list)
    meta_dict["test_frames"] = filter_frame(meta_dict["test_frames"], file_list)

    world2gt = np.array(meta_dict["worldtogt"])

    poses = []
    for frame in meta_dict["frames"] + meta_dict["test_frames"]:
        c2w = np.array(frame["camtoworld"])
        pose = world2gt @ c2w
        poses.append(pose)

    cam2worlds_array = np.array(poses)

    mean_position = cam2worlds_array[:, :3, 3].mean(axis=0)
    dists = []
    for i, frame in enumerate(meta_dict["frames"] + meta_dict["test_frames"]):
        position = poses[i][:3, 3]
        dist = np.linalg.norm(position - mean_position)
        dists.append(dist)

    dists = np.array(dists)
    ids = np.argpartition(dists, -5)[-5:]

    for i, frame in enumerate(meta_dict["frames"] + meta_dict["test_frames"]):
        if i in ids:
            print(frame["rgb_path"])

    def get_to_unit_cube(cam2worlds_array):
        min_vertices = cam2worlds_array[:, :3, 3].min(axis=0)
        max_vertices = cam2worlds_array[:, :3, 3].max(axis=0)

        center = (min_vertices + max_vertices) / 2.0
        scene_length = np.max(max_vertices - min_vertices) * boundry_extend
        scale = 2.0 / (scene_length)

        # normalize to unit cube
        to_unit_cube = np.eye(4).astype(np.float32)
        to_unit_cube[:3, 3] = -center
        to_unit_cube[:3] *= scale
        return to_unit_cube, scene_length, center, scale

    to_unit_cube, scene_length, center, scale = get_to_unit_cube(cam2worlds_array)

    meta_dict["worldtogt"] = np.linalg.inv(to_unit_cube).tolist()
    for i, frame in enumerate(meta_dict["frames"] + meta_dict["test_frames"]):
        pose = to_unit_cube @ poses[i]
        frame["camtoworld"] = pose.tolist()

    with open(scene_out_dir / "meta_data_filtered.json", "w") as f:
        json.dump(meta_dict, f, indent=4)


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene)
