from .G_nn.small_unet import ImageDenoiserModelV1
from cm.script_util import create_model as create_large_G
from cgan.D_nn.score_sde.models.discriminator import Discriminator_small, Discriminator_large

import torch.nn as nn
from torch.nn import init


def create_small_G(config):
    if "kdiffusion" in config["type"]:
        config = config["kdiffusion"]
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
    elif "huggingface" in config["type"]:
        import fastcore.all as fc
        from diffusers import UNet2DModel

        config = config["huggingface"]

        def init_ddpm(G):
            for o in G.down_blocks:
                for p in o.resnets:
                    p.conv2.weight.data.zero_()
                    for p in fc.L(o.downsamplers):
                        init.orthogonal_(p.conv.weight)

            for o in G.up_blocks:
                for p in o.resnets:
                    p.conv2.weight.data.zero_()

            G.conv_out.weight.data.zero_()

        class UNet(UNet2DModel):
            def forward(self, x, timestep, **kwargs):
                return super().forward(x, timestep=timestep, **kwargs).sample

        G = UNet(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            block_out_channels=config["block_out_channels"],
            norm_num_groups=config["norm_num_groups"],
        )
        init_ddpm(G)
    else:
        raise ValueError(f"Unknown G type {config['type']}")

    return G


def create_G(config):
    if "small" in config["type"]:
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
