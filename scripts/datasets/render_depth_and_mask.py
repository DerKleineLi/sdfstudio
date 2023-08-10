import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import OpenGL.GL
import trimesh

# Disable antialiasing:
suppress_multisampling = True
old_gl_enable = OpenGL.GL.glEnable


def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)


OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample


def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)


OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample

import pyrender

out_dir = Path("/home/hli/Data/data/scannetpp_sdf2")  # the directory of your converted data in sdfstudio format
in_dir = Path("/mnt/c/Users/li/Documents")  # the directory of scannetpp
original_in_dir = Path("/menegroth/scannetpp/data")  # Don't change
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2022-11-30_13-44"]  # scenes you want to process
image_size = 384


class DepthRenderer:
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
        color, depth = renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

        # correct depth with far blending
        # alpha = color[:, :, 3]
        # mask = alpha > 0
        # zfar = camera.zfar

        # depth[mask] = (alpha[mask] * depth[mask] * zfar) / (255 * zfar - 255 * depth[mask] + alpha[mask] * depth[mask])

        return depth


def render_depth_and_mask(ply_file, camera_intrinsic, camera_extrinsic, image_width, image_height):
    # Load the ply file
    pcd = trimesh.load(ply_file)
    pts = pcd.vertices.copy()
    colors = pcd.colors.copy()[:, :3]

    # Set camera intrinsic parameters
    intrinsic_matrix = np.array(camera_intrinsic).reshape((3, 3))

    # Set camera extrinsic parameters
    extrinsic_matrix = np.array(camera_extrinsic).reshape((4, 4))
    post_transformation = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    extrinsic_matrix = post_transformation @ extrinsic_matrix

    # Create a perspective camera
    camera = pyrender.IntrinsicsCamera(
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )

    scene.add(camera, pose=extrinsic_matrix)

    # Create a scene
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0))

    # Add the point cloud to the scene
    m = pyrender.Mesh.from_points(pts, colors=colors)
    scene.add(m)
    # world_ax = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
    # camera_ax = pyrender.Mesh.from_trimesh(trimesh.creation.axis(transform=extrinsic_matrix), smooth=False)
    # scene.add(world_ax)
    # scene.add(camera_ax)

    # Add the camera to the scene
    scene.add(camera, pose=extrinsic_matrix)
    # pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(384, 384))

    # Create a renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

    # Render the color and depth images
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    # Generate the mask (alpha channel)
    mask = (color[:, :, 3] > 0).astype(np.uint8) * 255

    # Close the renderer
    renderer.delete()

    return depth, mask


def process_one_scene(scene):
    # Replace these with your actual file paths and camera parameters
    scene_dir = in_dir / scene
    scene_out_dir = out_dir / scene
    ply_file = scene_dir / "scans" / "1mm" / "aligned.ply"
    transform_file = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    meta_file = scene_out_dir / "meta_data.json"

    with open(transform_file, "r") as f:
        trafo_dict = json.load(f)
    with open(meta_file, "r") as f:
        meta_dict = json.load(f)

    scale = 1 / meta_dict["worldtogt"][0][0]
    renderer = DepthRenderer(ply_file, image_width=image_size, image_height=image_size)

    camera_dict = {}
    for frame in trafo_dict["frames"] + trafo_dict["test_frames"]:
        key = os.path.basename(frame["file_path"])[:-4]
        extrinsic = np.array(frame["transform_matrix"])
        camera_dict[key] = {"extrinsic": extrinsic}
    for frame in meta_dict["frames"] + meta_dict["test_frames"]:
        key = os.path.basename(frame["rgb_path"])[:-8]
        intrinsic = np.array(frame["intrinsics"])
        camera_dict[key].update({"intrinsic": intrinsic})
        frame["sensor_depth_path"] = f"{key}_gtdepth.npy"

    for key, value in camera_dict.items():
        print("processing: " + key)
        depth = renderer.render(value["extrinsic"], value["intrinsic"])
        depth *= scale
        np.save(scene_out_dir / f"{key}_gtdepth.npy", depth)
        # np.save(scene_out_dir / f"{key}_mask.npy", mask)
        plt.imsave(scene_out_dir / f"{key}_gtdepth.png", depth, cmap="viridis")
        # plt.imsave(scene_out_dir / f"{key}_mask.png", mask, cmap="gray")

    meta_dict.update({"has_sensor_depth": True})
    with open(meta_file, "w") as f:
        json.dump(meta_dict, f, indent=4)


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene)
