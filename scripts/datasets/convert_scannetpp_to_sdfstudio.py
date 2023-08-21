# adapted from
# ttps://github.com/EPFL-VILAB/omnidata
# https://github.com/autonomousvision/monosdf

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# from .convert_utils import NormalDepthPredictor

out_dir = "/cluster/angmar/hli/data/scannetpp_sdf4"
in_dir = "/menegroth/scannetpp/data"
original_in_dir = "/menegroth/scannetpp/data"
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2022-12-04_13-16", "2022-11-30_13-44", "2022-12-17_11-51", "2023-01-12_12-02","2023-05-17_14-17", "2023-05-20_17-35"]
image_size = 384
resize_world = True
target_focal_length = None #409.88
boundry_extend = 5

# normal_depth_predictor = NormalDepthPredictor()

# process scenes
for scene in scenes:
    trafo_json = os.path.join(in_dir, scene, "dslr", "nerfstudio", "transforms.json")
    out_path = os.path.join(out_dir, scene)
    meta_json_path = os.path.join(out_path, "meta_data.json")
    new_trafo_path = os.path.join(out_path, "transforms.json")
    if not os.path.exists(trafo_json):
        print("Error: missing transform.json for {}".format(scene))
        continue
    if os.path.exists(meta_json_path):
        print("Info: {} already processed".format(scene))
        continue
    print("Info: convert {}".format(scene))
    # read trafo file
    with open(trafo_json, "r") as tf:
        trafo_dict = json.load(tf)
    # read intrinsics
    fisheye = {
        "w": trafo_dict["w"],
        "h": trafo_dict["h"],
        "fx": trafo_dict["fl_x"],
        "fy": trafo_dict["fl_y"],
        "cx": trafo_dict["cx"],
        "cy": trafo_dict["cy"],
        "k1": trafo_dict["k1"],
        "k2": trafo_dict["k2"],
        "k3": trafo_dict["k3"],
        "k4": trafo_dict["k4"],
    }
    cam2worlds = dict()
    undistorted_rgb_paths = []
    test_undistorted_rgb_paths = []
    
    os.makedirs(out_path, exist_ok=True)
    
    for frame_list, file_list in zip(
        [trafo_dict["frames"], trafo_dict["test_frames"]], [undistorted_rgb_paths, test_undistorted_rgb_paths]
    ):
        for frame in frame_list:
            # read rgb
            in_rgb_path = Path(frame["file_path"])
            in_rgb_path = in_rgb_path.relative_to(original_in_dir)
            in_rgb_path = str(Path(in_dir) / in_rgb_path)
            rgb_filename = os.path.basename(in_rgb_path).replace(".JPG", "_rgb.png")
            out_rgb_path = os.path.join(out_path, rgb_filename)
            bgr = cv2.imread(in_rgb_path).astype(float)
            
            # change trafo file path
            frame["file_path"] = rgb_filename

            # get pose and convert from nerfstudio to colmap/opencv
            pose = frame["transform_matrix"]
            pretransform = np.array([
                [1,0,0,0],
                [0,-1,0,0], # image mirror y
                [0,0,-1,0], # camera to the back
                [0,0,0,1]
            ])
            posttransform = np.array([
                [1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]
            ]) # rotating the world along x for 180 degrees
            # pose = posttransform @ pose @ pretransform
            pose = pose @ pretransform
            cam2worlds[rgb_filename] = pose

            # undistort fisheye
            K = np.eye(3)
            K[0, 0] = fisheye["fx"]
            K[1, 1] = fisheye["fy"]
            K[0, 2] = fisheye["cx"]
            K[1, 2] = fisheye["cy"]
            D = np.array(
                [
                    fisheye["k1"],
                    fisheye["k2"],
                    fisheye["k3"],
                    fisheye["k4"],
                ]
            )
            new_K = K.copy()
            if target_focal_length is not None:
                new_K[0, 0] = target_focal_length
                new_K[1, 1] = target_focal_length
            bgr_undistorted = cv2.fisheye.undistortImage(bgr, K, D=D, Knew=new_K)
            # resize
            resize_size = (int(float(fisheye["w"]) * float(image_size) / float(fisheye["h"])), image_size) if target_focal_length is None else (fisheye["w"], fisheye["h"])
            bgr_resized = (
                transforms.functional.to_tensor(Image.fromarray(bgr_undistorted.astype(np.uint8)).resize(resize_size))
                .permute(1, 2, 0)
                .numpy()
            )
            # center crop
            offset_x = math.floor(float(resize_size[0] - image_size) / 2)
            offset_y = math.floor(float(resize_size[1] - image_size) / 2)
            bgr_cropped = bgr_resized[offset_y:offset_y + image_size, offset_x : offset_x + image_size]

            # write image
            cv2.imwrite(out_rgb_path, (bgr_cropped * 255.0).astype(np.uint8))
            file_list.append(out_rgb_path)

            # # predict normal and depth
            # normal_depth_predictor.predict_normal_depth(out_rgb_path, out_path)

    # resize and crop intrinsics
    fact_x = float(resize_size[0]) / float(fisheye["w"])
    fact_y = float(resize_size[1]) / float(fisheye["h"])
    K[0, 0] *= fact_x
    K[1, 1] *= fact_y
    K[0, 2] *= fact_x
    K[1, 2] *= fact_y
    K[0, 2] -= offset_x
    K[1, 2] -= offset_y

    undistorted_rgb_paths = sorted(undistorted_rgb_paths)
    test_undistorted_rgb_paths = sorted(test_undistorted_rgb_paths)

    cam2worlds_array = np.array(list(cam2worlds.values()))
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

    # write json file
    meta = dict()
    meta["camera_model"] = "OPENCV"
    meta["height"] = image_size
    meta["width"] = image_size
    meta["has_mono_prior"] = True
    meta["pairs"] = None
    if resize_world:
        meta["worldtogt"] = np.linalg.inv(to_unit_cube).tolist()
    else:
        meta["worldtogt"] = np.eye(4).tolist()
    meta["scene_box"] = dict()
    if resize_world:
        meta["scene_box"]["aabb"] = np.array(
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                ],
            ]
        ).tolist()
    else:
        meta["scene_box"]["aabb"] = (np.array(
            [
                [
                    -scene_length,
                    -scene_length,
                    -scene_length,
                ],
                [
                    scene_length,
                    scene_length,
                    scene_length,
                ],
            ]
        ) / 2).tolist()
    meta["scene_box"]["near"] = 0.05
    if resize_world:
        meta["scene_box"]["far"] = 2.5
        meta["scene_box"]["radius"] = 1.0
    else:
        meta["scene_box"]["far"] = scene_length
        meta["scene_box"]["radius"] = scene_length / 2
    meta["scene_box"]["collider_type"] = "box"

    frames = []
    test_frames = []
    for file_list, frame_list in zip([undistorted_rgb_paths, test_undistorted_rgb_paths], [frames, test_frames]):
        for rgb_path in file_list:
            frame = dict()
            rgb_filename = os.path.basename(rgb_path)
            stem = os.path.splitext(rgb_filename)[0]
            rgb_rel_path = rgb_filename
            frame["rgb_path"] = rgb_rel_path
            cam2world = cam2worlds[rgb_filename]
            cam2world[:3, 3] -= center
            if resize_world:
                cam2world[:3, 3] *= scale
            frame["camtoworld"] = cam2world.tolist()
            frame["intrinsics"] = K.tolist()
            frame["mono_depth_path"] = rgb_rel_path.replace("_rgb.png", "_depth.npy")
            frame["mono_normal_path"] = rgb_rel_path.replace("_rgb.png", "_normal.npy")
            frame_list.append(frame)
    meta["frames"] = frames
    meta["test_frames"] = test_frames

    with open(meta_json_path, "w") as mf:
        json.dump(meta, mf, indent=4)
        
    # write transformed transforms.json
    transform_matrix_array = np.array(list(frame["transform_matrix"] for frame in trafo_dict["frames"]+trafo_dict["test_frames"]))
    to_unit_cube, scene_length, center, scale = get_to_unit_cube(transform_matrix_array)
    trafo_dict["aabb_scale"] = 2
    trafo_dict.update({"fl_x": K[0, 0],
    "fl_y": K[1, 1],
    "cx": K[0, 2],
    "cy": K[1, 2],
    "w": image_size,
    "h": image_size,
    "sk_x": 0.0,
    "sk_y": 0.0,
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
    "k4": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "is_fisheye": False,
    "aabb_range": [[1,-1],[1,-1],[1,-1]],
    "sphere_center":[0,0,0],
    "sphere_radius": 1.732,
    "centered": True,
    "scaled": True,
    "camera_model": "OPENCV",
    })
    for frame in trafo_dict["frames"]+trafo_dict["test_frames"]:
        pose = np.array(frame["transform_matrix"])
        pose[:3, 3] -= center
        if resize_world:
            pose[:3, 3] *= scale
        frame["transform_matrix"] = pose.tolist()
    
    with open(new_trafo_path, "w") as tf:
        json.dump(trafo_dict, tf, indent=4)
