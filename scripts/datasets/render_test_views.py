import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh

out_dir = Path("/home/hli/Data/data/scannetpp_sdf2")
in_dir = Path("/mnt/c/Users/li/Documents")
original_in_dir = Path("/menegroth/scannetpp/data")
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2022-11-30_13-44"]
image_size = 384


class Renderer:
    def __init__(
        self,
        ply_file,
        intrinsic=None,
        image_width=None,
        image_height=None,
        post_transform=np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ):
        self.ply_file = ply_file
        self.image_width = image_width
        self.image_height = image_height
        self.post_transform = post_transform
        self.intrinsic = intrinsic

        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0))
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=image_width, viewport_height=image_height, point_size=1
        )

        # Load the ply file
        pcd = trimesh.load(ply_file)
        pts = pcd.vertices.copy()
        colors = pcd.colors.copy()[:, :3]
        m = pyrender.Mesh.from_points(pts, colors=colors)
        self.scene.add(m)

    def render(self, extrinsic, intrinsic=None, image_width=None, image_height=None):
        if image_width is None and image_height is None:
            renderer = self.renderer
        else:
            renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

        if intrinsic is None:
            intrinsic = self.intrinsic
        assert intrinsic is not None, "intrinsic must be specified"

        extrinsic = self.post_transform @ extrinsic

        # clear other cameras
        for node in list(self.scene.camera_nodes):
            self.scene.remove_node(node)

        camera = pyrender.IntrinsicsCamera(
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )

        self.scene.add(camera, pose=extrinsic)
        # world_ax = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
        # camera_ax = pyrender.Mesh.from_trimesh(trimesh.creation.axis(transform=extrinsic), smooth=False)
        # self.scene.add(world_ax)
        # self.scene.add(camera_ax)
        # pyrender.Viewer(self.scene, use_raymond_lighting=True, viewport_size=(self.image_width, self.image_height))
        # exit()
        color, depth = renderer.render(self.scene, flags=pyrender.RenderFlags.FLAT)
        return color


def process_one_scene(scene):
    # Replace these with your actual file paths and camera parameters
    scene_dir = in_dir / scene
    scene_out_dir = out_dir / scene
    ply_file = scene_out_dir / "reproject" / "combined.ply"
    transform_file = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    meta_file = scene_out_dir / "meta_data.json"

    with open(transform_file, "r") as f:
        trafo_dict = json.load(f)
    with open(meta_file, "r") as f:
        meta_dict = json.load(f)

    renderer = Renderer(ply_file, image_width=image_size, image_height=image_size, post_transform=np.eye(4))

    camera_dict = {}
    for frame in meta_dict["test_frames"]:
        key = os.path.basename(frame["rgb_path"])[:-8]
        intrinsic = np.array(frame["intrinsics"])
        extrinsic = np.array(frame["camtoworld"])
        pretransform = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]  # image mirror y  # camera to the back
        )
        extrinsic = extrinsic @ pretransform
        rgb_file = scene_out_dir / frame["rgb_path"]
        depth_file = scene_out_dir / frame["sensor_depth_path"]
        camera_dict[key] = {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "rgb_file": rgb_file,
            "depth_file": depth_file,
        }

    for key, value in camera_dict.items():
        print("processing: " + key)
        color = renderer.render(value["extrinsic"], value["intrinsic"])
        color = color.copy(order="C")
        plt.imsave(scene_out_dir / "reproject" / f"{key}.png", color)
        shutil.copyfile(value["rgb_file"], scene_out_dir / "reproject" / f"{key}_original.png")


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene)
