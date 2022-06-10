## Imports

from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.data.annotation import AnnotationWriter
from dh_segment_torch.training import Trainer
from dh_segment_torch.inference import PredictProcess
from dh_segment_torch.post_processing import PostProcessingPipeline

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import os, glob, json, cv2, collections, torch 

## Loading data

folder_name = 'auto_vector' # Change this parameter to your project name
params = {
    'data_path' : 'data/'.format(folder_name), # Path to write the data
    'data_splitter': {'train_ratio': 0.7, 'val_ratio': 0.2, 'test_ratio': 0.1}, # splitting ratio of the data
    'copy_images': True, # Whether to copy the images
    'overwrite': True, # Whether to overwrite the images
    'progress': True # Whether to show progress
}

data_path = params.pop("data_path")

color_label = {
    'path': os.path.join(data_path, "color_labels.json"),
    'colors': {
        'background': [0, 0, 0], # RGB
        'edge': [255, 255, 255] # RGB
    }
}

## Process parameters

num_processes = params.pop("num_processes", 4)

relative_path = params.pop("relative_path", True)

params.setdefault("labels_dir", os.path.join(data_path, "labels"))
labels_dir = params.get("labels_dir")

params.setdefault("images_dir", os.path.join(data_path, "images"))
images_dir = params.get("images_dir")

params.setdefault("color_labels_file_path", os.path.join(data_path, "color_labels.json"))
params.setdefault("csv_path", os.path.join(data_path, "data.csv"))

data_splitter_params = params.pop("data_splitter", None)
train_csv_path = params.pop("train_csv", os.path.join(data_path, "train.csv"))
val_csv_path = params.pop("val_csv", os.path.join(data_path, "val.csv"))
test_csv_path = params.pop("test_csv", os.path.join(data_path, "test.csv"))

params.setdefault("type", "image")

labels_list = sorted(glob.glob(os.path.join(labels_dir, '*.*')))
images_list = sorted(glob.glob(os.path.join(images_dir, '*.*')))

data = pd.DataFrame({'image': images_list, 'label': labels_list})
data.to_csv(params['csv_path'], header=False, index=False)

if relative_path:
    data['image'] = data['image'].apply(lambda path: os.path.join("images", os.path.basename(path)))
    data['label'] = data['label'].apply(lambda path: os.path.join("labels", os.path.basename(path)))
    
if data_splitter_params:
    data_splitter = DataSplitter.from_params(data_splitter_params)
    data_splitter.split_data(data, train_csv_path, val_csv_path, test_csv_path)
    
for class_name in color_label['colors'].keys():
    if type(color_label['colors'][class_name]) == str:
        color_label['colors'][class_name] = list(ImageColor.getcolor(
            color_label['colors'][class_name], "RGB"))

with open(color_label['path'], 'w') as outfile:
    json.dump({'colors': list(color_label['colors'].values()),
              'one_hot_encoding': None,
              'labels': list(color_label['colors'].keys())}, outfile)

## Training params

params = {
    "color_labels": {"label_json_file": os.path.join(data_path, "color_labels.json")}, # Color labels produced before
    "train_dataset": {
        "type": "image_csv", # Image csv dataset
        "csv_filename":  os.path.join(data_path, "train.csv"),
        "base_dir": data_path,
        "repeat_dataset": 1,
        "compose": {"transforms": [{"type": "rotate", "p": 0.5},
                                   {"type": "vertical_flip", "p": 0.5},
                                   {"type": "horizontal_flip", "p": 0.5}]}
    },
    "val_dataset": {
        "type": "image_csv", # Validation dataset
        "csv_filename": os.path.join(data_path, "val.csv"),
        "base_dir": data_path,
        "compose": {"transforms": []}
    },
    "model": { # Model definition, original dhSegment
        "encoder": "resnet101", 
        "decoder": {
            "decoder_channels": [512, 256, 128, 64, 32],
            "max_channels": 512
        }
    },
    "initializer": {
        "initializers": [
            { "regexes": "decoder.*.conv2d.weight$", "type": "xavier_uniform" },
            { "regexes": "decoder.*.conv2d.bias$", "type": "zeros" }]
    },
    "metrics": [['miou', 'iou'], ['iou', {"type": 'iou', "average": None}], 
        ['Maccuracy', 'accuracy'], ['acc', {"type": 'accuracy', "average": None}], 
        ['Mprecision', 'precision'], ['pre', {"type": 'precision', "average": None}], 
        ['Mrecall', 'recall'], ['rec', {"type": 'recall', "average": None}]], # Metrics to compute
    "optimizer": {"lr": 5e-5}, # Learning rate
    "lr_scheduler": {"type": "exponential", "gamma": 0.9999},
    "val_metric": "+miou", # Metric to observe to consider a model better than another, the + indicates that we want to maximize
    "early_stopping": { "patience": 25}, # Number of validation steps without increase to tolerate, stops if reached
    "model_out_dir": f"../models/{folder_name}", # Path to model output
    "num_epochs": 60, # Number of epochs for training
    "device": "cuda:1",
    "num_accumulation_steps": 3,
    "evaluate_every_epoch": 1, # Number of epochs between each validation of the model
    "batch_size": 2, # Batch size (to be changed if the allocated GPU has little memory)
    "num_data_workers": 4,
    "track_train_metrics": False,
    "loggers": [
       {   # Tensorboard logging
           "type": 'tensorboard', 
           "log_dir": f"../tensorboard/{folder_name}/log",
           "log_every": 4, "log_images_every": 60
       }] 
}

## Train

trainer = Trainer.from_params(params)
trainer.train()