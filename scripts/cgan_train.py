"""
Train a diffusion netG on images.
"""

import argparse


from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults as netG_and_diffusion_defaults,
    create_model_and_diffusion as create_netG_and_diffusion,
    create_model as create_netG,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cgan.cgan_train_util import ConsistencyGANTrainLoop
from cgan.cgan_karras_diffusion import CGANKarrasDenoiser
from cgan.discriminator import Discriminator_small, Discriminator_large
import torch.distributed as dist
import torch.nn as nn

import copy


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating netG and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Discriminator
    logger.log("creating the netD...")
    Discriminator = (
        Discriminator_small
        if args.dataset
        in ["cifar10", "stackmnist", "tiny_imagenet_200", "stl10", "imagenet-64"]
        else Discriminator_large
    )
    netD = Discriminator(
        nc=2 * args.netD_num_channels,
        ngf=args.netD_ngf,
        t_emb_dim=args.netD_t_emb_dim,
        act=nn.LeakyReLU(0.2),
    ).to(dist_util.dev())

    # Generator
    logger.log("creating the netG...")
    netG_kwargs = args_to_dict(args, netG_defaults().keys())
    netG = create_netG(**netG_kwargs)
    netG.to(dist_util.dev())
    netG.train()
    if args.use_fp16:
        netG.convert_to_fp16()

    # Karras Diffusion
    cgan_diffusion_kwargs = args_to_dict(args, cgan_diffusion_defaults().keys())
    if "distillation" in args.training_mode:
        cgan_diffusion_kwargs.use_adjacent_points = True
        cgan_diffusion_kwargs.use_ode_solver = True
    diffusion = CGANKarrasDenoiser(
        **cgan_diffusion_kwargs,
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Data Loader
    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # Teacher model
    if len(args.teacher_model_path) > 0:  # path to the teacher score netG.
        logger.log(f"loading the teacher netG from {args.teacher_model_path}...")

        teacher_model_and_diffusion_kwargs = copy.deepcopy(netG_kwargs)
        teacher_model_and_diffusion_kwargs.update(diffusion_defaults())
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_netG, teacher_diffusion = create_netG_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_netG.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_netG.to(dist_util.dev())
        teacher_netG.eval()

        for dst, src in zip(netG.parameters(), teacher_netG.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_netG.convert_to_fp16()

    else:
        teacher_netG = None
        teacher_diffusion = None

    # load the target netG for distillation, if path specified.
    logger.log("creating the target netG")
    target_netG = create_netG(
        **netG_kwargs,
    )
    target_netG.to(dist_util.dev())
    target_netG.train()

    dist_util.sync_params(target_netG.parameters())
    dist_util.sync_params(target_netG.buffers())

    for dst, src in zip(target_netG.parameters(), netG.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_netG.convert_to_fp16()

    logger.log("training...")
    ConsistencyGANTrainLoop(
        netD=netD,
        model=netG,
        target_netG=target_netG,
        teacher_netG=teacher_netG,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lazy_reg=args.lazy_reg,
        r1_gamma=args.r1_gamma,
    ).run_loop()


def cgan_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.1,
        training_mode="consistency_gan_distillation",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=40,
        end_scales=40,
        distill_steps_per_iter=50000,
        loss_norm={
            "lpips": 1,
        },
        use_adjacent_points=True,
        use_ode_solver=True,
    )


def netD_defaults():
    return dict(
        dataset="imagenet-64",
        netD_num_channels=3,
        netD_ngf=64,
        netD_t_emb_dim=256,
        netD_patch_size=1,
        netD_use_local_loss=False,
        lazy_reg=15,
        r1_gamma=0.02,
    )


def netG_defaults():
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
    )


def diffusion_defaults():
    return dict(
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        distillation=True,
        weight_schedule="karras",
    )


def cgan_diffusion_defaults():
    diffusion_kwargs = diffusion_defaults()
    diffusion_kwargs.update(
        dict(
            use_adjacent_points=False,
            use_ode_solver=False,
        )
    )
    return diffusion_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(netD_defaults())
    defaults.update(netG_defaults())
    defaults.update(cgan_diffusion_defaults())
    defaults.update(cgan_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
