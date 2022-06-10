from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.data.annotation import AnnotationWriter
from dh_segment_torch.training import Trainer
from dh_segment_torch.inference import PredictProcess
from dh_segment_torch.post_processing import PostProcessingPipeline

import os, glob, json, cv2

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageColor

data_path = "data/"


dataset_params = {
    "type": "csv",
    "csv_path": "data/test.csv",
    "base_dir": data_path,
    "pre_processing": {"transforms": []}#{"type": "fixed_size_resize", "output_size": 1e6}]}
}

color_label = {
    'path': os.path.join(data_path, "color_labels.json"),
    'colors': {
        'background': [0, 0, 0], # RGB
        'edge': [255, 255, 255] # RGB
    }
}

model_params = {
    "model": {
            "encoder": "resnet101",
            "decoder": {"decoder_channels": [512, 256, 128, 64, 32], "max_channels": 512}
        },
        "num_classes": 2,
        # "model_state_dict": sorted(glob.glob("../models/auto_vector/best_model_checkpoint_miou=*.pth"))[-1],
        "model_state_dict": sorted(glob.glob("../historical_model/best_model_checkpoint_miou=*.pth"))[-1],
        "device": "cuda:0"
}

process_params = Params({
    'data': dataset_params,
    'model': model_params,
    'batch_size': 1,
    'num_workers': 4,
    'add_path': True
})

result_dir = "results/"
output_dir = "predictions/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, output_dir), exist_ok=True)

predict_annots = PredictProcess.from_params(process_params)
predict_annots.process_to_probas_files(output_dir)


for path in sorted(glob.glob(os.path.join(output_dir, '*.npy'))):
    img_name = path[:-4]
    np_result = np.load(path)
    image = np.argmax(np_result, axis=0).astype('uint8')
    colors = list(color_label['colors'].values())
    canvas = np.zeros((image.shape[0], image.shape[1], 3)).astype('uint8')
    for i, color in enumerate(colors):
        canvas[image == i] = color

    print(os.path.join(result_dir, (img_name + ".png")))
    cv2.imwrite(os.path.join(result_dir, (img_name + ".png")), canvas)
  