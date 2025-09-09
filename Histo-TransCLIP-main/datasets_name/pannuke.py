import os
from collections import defaultdict, OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden
import os
import glob
import random

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import h5py
import numpy
import json
import pandas as pd
from pathlib import Path
import io

templates = [
            "{}.",
            "a photomicrograph showing {}.",
            "a photomicrograph of {}.",
            "an image of {}.",
            "an image showing {}.",
            "an example of {}.",
            "{} is shown.",
            "this is {}.",
            "there is {}.",
            "a histopathological image showing {}.",
            "a histopathological image of {}.",
            "a histopathological photograph of {}.",
            "a histopathological photograph showing {}.",
            "shows {}.",
            "presence of {}.",
            "{} is present.",
            "an H&E stained image of {}.",
            "an H&E stained image showing {}.",
            "an H&E image showing {}.",
            "an H&E image of {}.",
            "{}, H&E stain.",
            "{}, H&E."
]
class Pannuke(DatasetBase):
    dataset_dir ="pannuke"

    def __init__(self, root, preprocess):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.dataset_test = os.path.join(self.dataset_dir, "evaluation_datasets/trainratio=0.70_size=224/PanNuke_test.csv")
        self.dataset_train = os.path.join(self.dataset_dir, "evaluation_datasets/trainratio=0.70_size=224/PanNuke_train.csv")

        self.template = templates

        test = self.read_data('test')
        val  = self.read_data('test')
        train = self.read_data('train')

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        if split == 'test':
            df = pd.read_csv(self.dataset_test)
        else:
            df = pd.read_csv(self.dataset_train)

        items = []
        for index, row in df.iterrows():
            impath = row['image']
            item = Datum(impath=impath, label=int(row['label']), classname=row['label_text'])
            items.append(item)
        return items
        