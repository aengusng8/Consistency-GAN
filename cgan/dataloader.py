import torch
import torch.nn as nn
import torchvision.transforms.functional as TF, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


from datasets import load_dataset
from functools import reduce
import math
import operator

import numpy as np
from skimage import transform


from cm.image_datasets import load_data


class TargetTransform:
    def __init__(self, class_cond=False):
        self.class_cond = class_cond

    def __call__(self, target):
        if not self.class_cond:
            return {}

def load_data_generator(config):
    if config["dataset"]["name"] == "fashion_mnist":

        tf = transforms.Compose(
            [
                transforms.Resize(
                    config["dataset"]["image_size"],
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop(config["dataset"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        dataset = datasets.FashionMNIST(
            root=config["dataset"]["location"],
            train=True,
            download=True,
            transform=tf,
            target_transform=TargetTransform(config["G"]["class_cond"]),
        )

        loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=1,
        )

        def _generator():
            while True:
                yield from loader

        generator = _generator()

    else:
        generator = load_data(
            data_dir=config["dataset"]["location"],
            batch_size=config["train"]["batch_size"],
            image_size=config["dataset"]["image_size"],
            class_cond=config["G"]["class_cond"],
        )

    return generator