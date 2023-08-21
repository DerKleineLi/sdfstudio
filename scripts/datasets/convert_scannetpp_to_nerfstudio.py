# adapted from
# ttps://github.com/EPFL-VILAB/omnidata
# https://github.com/autonomousvision/monosdf

import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# from .convert_utils import NormalDepthPredictor

out_dir = "/cluster/angmar/hli/data/scannetpp_ns"
in_dir = "/menegroth/scannetpp/data"
original_in_dir = "/menegroth/scannetpp/data"
# scenes = sorted(os.listdir(in_dir))  # ["2022-12-17_11-51"]
scenes = ["2022-11-30_13-44"] #"2023-06-15_10-50",

for scene in scenes:
    trafo_json = os.path.join(in_dir, scene, "dslr", "nerfstudio", "transforms.json")
    out_path = os.path.join(out_dir, scene)
    new_trafo_path = os.path.join(out_path, "transforms.json")
    if not os.path.exists(trafo_json):
        print("Error: missing transform.json for {}".format(scene))
        continue
    # if os.path.exists(meta_json_path):
    #     print("Info: {} already processed".format(scene))
    #     continue
    print("Info: convert {}".format(scene))
    # read trafo file
    with open(trafo_json, "r") as tf:
        trafo_dict = json.load(tf)

    os.makedirs(out_path, exist_ok=True)
    
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
    
    poses = []
    
    print(len(trafo_dict["frames"]))
    
    for frame in trafo_dict["frames"] + trafo_dict["test_frames"]:
        # read rgb
        in_rgb_path = Path(frame["file_path"])
        in_rgb_path = in_rgb_path.relative_to(original_in_dir)
        in_rgb_path = str(Path(in_dir) / in_rgb_path)
        rgb_filename = os.path.basename(in_rgb_path).replace(".JPG", "_rgb.png")
        out_rgb_path = os.path.join(out_path, rgb_filename)
        
        bgr = cv2.imread(in_rgb_path).astype(float)
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
        bgr_undistorted = cv2.fisheye.undistortImage(bgr, K, D=D, Knew=new_K)
        bgr_resized = ( # no resize
                transforms.functional.to_tensor(Image.fromarray(bgr_undistorted.astype(np.uint8)))
                .permute(1, 2, 0)
                .numpy()
            )
        cv2.imwrite(out_rgb_path, (bgr_resized * 255.0).astype(np.uint8))
        # shutil.copy(in_rgb_path, out_rgb_path)
        frame["file_path"] = rgb_filename
        poses.append(np.array(frame["transform_matrix"])[:3,3])
    
    poses = np.array(poses)
    max_poses = np.max(poses, axis=0)
    min_poses = np.min(poses, axis=0)
    center = (max_poses + min_poses) / 2
    bbox_half_length = ((max_poses - min_poses) + 3)  
    
    for frame in trafo_dict["frames"] + trafo_dict["test_frames"]:
        pose = np.array(frame["transform_matrix"])
        pose[:3,3] -= center
        frame["transform_matrix"] = pose.tolist()
    aabb_range = np.array([bbox_half_length, -bbox_half_length]).T.tolist()
    trafo_dict.update({"fl_x": K[0, 0],
        "fl_y": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
        "w": 1752,
        "h": 1168,
        "sk_x": 0.0,
        "sk_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,
        "aabb_range": aabb_range,
        "aabb_scale": 8.0,
        "sphere_center":[0,0,0],
        "sphere_radius": np.linalg.norm(bbox_half_length),
        "centered": True,
        "scaled": False,
        "camera_model": "OPENCV",
        })

    with open(new_trafo_path, "w") as mf:
        json.dump(trafo_dict, mf, indent=4)