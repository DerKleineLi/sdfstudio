import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

file_path = "/home/hli/Data/data/scannetpp_sdf2/2022-11-19_11-22/meta_data.json"
with open(file_path, "r") as f:
    meta = json.load(f)

for frame in meta["frames"]:
    pose = frame["camtoworld"]
    posttransform = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1]
    ]) # rotating the world along x for 180 degrees
    pose = posttransform @ pose
    frame["camtoworld"] = pose.tolist()

output_file_path = file_path[:-5] + "_flipped.json"
with open(output_file_path, "w") as f:
    json.dump(meta, f, indent=4)