from functools import partial
import json
import math
import warnings

from jsonmerge import merge

from ..G_nn import layers
from ..G_nn.tiny_unet import ImageDenoiserModelV1
from cm.script_util import create_model as create_large_G
from cgan.D_nn.discriminator import Discriminator_small, Discriminator_large

import torch.nn as nn


def load_config(file):
    defaults = {
        "model": {
            "sigma_data": 1.0,
            "patch_size": 1,
            "dropout_rate": 0.0,
            "augment_wrapper": True,
            "augment_prob": 0.0,
            "mapping_cond_dim": 0,
            "unet_cond_dim": 0,
            "cross_cond_dim": 0,
            "cross_attn_depths": None,
            "skip_stages": 0,
            "has_variance": False,
            "loss_config": "karras",
        },
        "dataset": {
            "type": "imagefolder",
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "betas": [0.95, 0.999],
            "eps": 1e-6,
            "weight_decay": 1e-3,
        },
        "lr_sched": {
            "type": "constant",
        },
        "ema_sched": {"type": "inverse", "power": 0.6667, "max_value": 0.9999},
    }
    config = json.load(file)
    return merge(defaults, config)


def create_small_G(config):
    G = ImageDenoiserModelV1(
        config["input_channels"],
        config["mapping_out"],
        config["depths"],
        config["channels"],
        config["self_attn_depths"],
        config["cross_attn_depths"],
        patch_size=config["patch_size"],
        dropout_rate=config["dropout_rate"],
        mapping_cond_dim=config["mapping_cond_dim"]
        + (9 if config["augment_wrapper"] else 0),
        unet_cond_dim=config["unet_cond_dim"],
        cross_cond_dim=config["cross_cond_dim"],
        skip_stages=config["skip_stages"],
        has_variance=config["has_variance"],
    )
    return G


def create_G(config):
    if config["type"] == "small":
        G = create_small_G(config)
    elif config["type"] == "large":
        G = create_large_G(
            image_size=config["image_size"],
            num_channels=config["num_channels"],
            num_res_blocks=config["num_res_blocks"],
            channel_mult=config["channel_mult"],
            learn_sigma=config["learn_sigma"],
            class_cond=config["class_cond"],
            use_checkpoint=config["use_checkpoint"],
            attention_resolutions=config["attention_resolutions"],
            num_heads=config["num_heads"],
            num_head_channels=config["num_head_channels"],
            num_heads_upsample=config["num_heads_upsample"],
            use_scale_shift_norm=config["use_scale_shift_norm"],
            dropout=config["dropout"],
            resblock_updown=config["resblock_updown"],
            use_fp16=config["use_fp16"],
        )
    else:
        raise ValueError(f"Unknown G type {config['type']}")
    return G


def create_D(config):
    Discriminator = (
        Discriminator_small if config["type"] == "small" else Discriminator_large
    )
    D = Discriminator(
        nc=2 * config["num_channels"],
        ngf=config["ngf"],
        t_emb_dim=config["t_emb_dim"],
        act=nn.LeakyReLU(0.2),
    )
    return D


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
