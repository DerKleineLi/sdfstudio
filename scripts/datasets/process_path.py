import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

file_path = "/home/hli/Data/data/scannetpp/2022-11-30_13-44/dslr/nerfstudio/transforms.json"

target_image_dir = "/home/hli/Data/data/scannetpp_fisheye/2022-11-30_13-44"

with open(file_path, "r") as f:
    meta = json.load(f)

for frame in meta["frames"] + meta["test_frames"]:
    basename = os.path.basename(frame["file_path"])
    frame["file_path"] = basename

output_file_path = os.path.join(target_image_dir, "meta_data.json")
with open(output_file_path, "w") as f:
    json.dump(meta, f, indent=4)