import json
import os
import shutil
from pathlib import Path

import numpy as np

in_dir = Path('/cluster/angmar/hli/data/dtu')
out_dir = Path('/cluster/angmar/hli/data/dtu_processed')
scenes = ["scan24"]  # scenes you want to process

def process_one_scene(scene, num_images = 20, random = False):
    scene_dir = in_dir / scene
    if not random:
        scene_out_dir = out_dir / scene / "no_random"
    else:
        scene_out_dir = out_dir / scene / "random"
        
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "meta_data.json", "r") as f:
        meta_dict = json.load(f)

    camera_dict = {}
    
    n = len(meta_dict["frames"])
    if random:
        ids = np.random.permutation(n)[:num_images]
    else:
        ids = np.arange(n)[-num_images:]
    meta_dict["frames"] = [meta_dict["frames"][i] for i in ids]

    for frame in meta_dict["frames"]:
        shutil.copy(scene_dir / frame["rgb_path"], scene_out_dir / frame["rgb_path"])

    with open(scene_out_dir / "meta_data.json", "w") as f:
        json.dump(meta_dict, f, indent=4)


if __name__ == "__main__":
    for scene in scenes:
        process_one_scene(scene, random=False)
        process_one_scene(scene, random=True)