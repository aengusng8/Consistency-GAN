""" Train ConsistencyGAN with Karras Diffusion. """

import argparse
import warnings
import wandb

warnings.filterwarnings("ignore", category=UserWarning)


from cm import dist_util, logger
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cgan.cgan_train_util import ConsistencyGANTrainLoop
from cgan.cgan_karras_diffusion import CGANKarrasDenoiser
from cgan.configs.config import load_config, seed_everything
from cgan.models import create_G, create_D
from cgan.dataloader import load_data_generator

import torch.distributed as dist
import torch.nn as nn

import copy


def main(config):
    dist_util.setup_dist()
    logger.configure()

    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=config["EMA"]["target_ema_mode"],
        start_ema=config["EMA"]["start_ema"],
        scale_mode=config["loss"]["scale_mode"],
        start_scales=config["loss"]["start_scales"],
        end_scales=config["loss"]["end_scales"],
        total_steps=config["train"]["total_training_steps"],
        distill_steps_per_iter=config["EMA"]["distill_steps_per_iter"],
    )

    # Discriminator
    D = create_D(config["D"]).to(dist_util.dev())
    logger.log(f"Created {config['D']['type']} D for {config['dataset']['name']}")

    # Generator
    G = create_G(config["G"]).to(dist_util.dev())
    logger.log(f"Created {config['G']['type']} G for {config['dataset']['name']}")
    G.train()
    if config["G"]["use_fp16"]:
        G.convert_to_fp16()

    # Karras Diffusion
    cgan_diffusion_kwargs = config["diffusion"]
    cgan_diffusion_kwargs["loss_norm"] = config["loss"]["loss_norm"]
    if "distillation" in config["train"]["training_mode"]:
        cgan_diffusion_kwargs.use_adjacent_points = True
        cgan_diffusion_kwargs.use_ode_solver = True
    diffusion = CGANKarrasDenoiser(
        **cgan_diffusion_kwargs,
    )
    schedule_sampler = create_named_schedule_sampler(
        config["loss"]["schedule_sampler"], diffusion
    )
    logger.log(f"Created Karras diffusion")

    # Data Loader
    if config["train"]["batch_size"] == -1:
        batch_size = config["train"]["global_batch_size"] // dist.get_world_size()
        if config["train"]["global_batch_size"] % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {config['train']['global_batch_size']}"
            )
    else:
        batch_size = config["train"]["batch_size"]

    config["train"]["batch_size"] = batch_size
    data = load_data_generator(config)
    logger.log("Created data loader")

    # Teacher model
    # TODO: fix the code for creating teacher model
    if len(config["teacherG"]["teacher_model_path"]) > 0:
        logger.log(
            f"loading the teacher G from {config['teacherG']['teacher_model_path']}..."
        )

        teacher_model_and_diffusion_kwargs = copy.deepcopy(G_kwargs)
        teacher_model_and_diffusion_kwargs.update(diffusion_defaults())
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        # teacher_G, teacher_diffusion = create_G_and_diffusion(
        #     **teacher_model_and_diffusion_kwargs,
        # )
        teacher_G = create_G(config["G"])
        # teacher_diffusion = CGANKarrasDenoiser(

        teacher_G.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_G.to(dist_util.dev())
        teacher_G.eval()

        for dst, src in zip(G.parameters(), teacher_G.parameters()):
            dst.data.copy_(src.data)

        if config["G"]["use_fp16"]:
            teacher_G.convert_to_fp16()

    else:
        teacher_G = None
        teacher_diffusion = None

    # load the target G for distillation, if path specified.
    target_G = create_G(config["G"]).to(dist_util.dev())
    target_G.train()

    dist_util.sync_params(target_G.parameters())
    dist_util.sync_params(target_G.buffers())

    for dst, src in zip(target_G.parameters(), G.parameters()):
        dst.data.copy_(src.data)

    if config["G"]["use_fp16"]:
        target_G.convert_to_fp16()
    logger.log("Created target G")

    logger.log("training...")
    ConsistencyGANTrainLoop(
        D=D,
        model=G,
        target_G=target_G,
        teacher_G=teacher_G,
        diffusion=diffusion,
        teacher_diffusion=teacher_diffusion,
        training_mode=config["train"]["training_mode"],
        ema_scale_fn=ema_scale_fn,
        total_training_steps=config["train"]["total_training_steps"],
        data=data,
        batch_size=batch_size,
        microbatch=config["train"]["microbatch"],
        lr=config["optG"]["lr"],
        ema_rate=config["EMA"]["ema_rate"],
        log_interval=config["train"]["log_interval"],
        save_interval=config["train"]["save_interval"],
        resume_checkpoint=config["train"]["resume_checkpoint"],
        use_fp16=config["G"]["use_fp16"],
        fp16_scale_growth=config["G"]["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=config["optG"]["weight_decay"],
        lr_anneal_steps=config["optG"]["lr_anneal_steps"],
        lazy_reg=config["optD"]["lazy_reg"],
        r1_gamma=config["optD"]["r1_gamma"],
    ).run_loop()


def create_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--config", type=str, required=True, help="the configuration file")
    # p.add_argument("--wandb-entity", type=str, help="the wandb entity name")
    # p.add_argument("--wandb-group", type=str, help="the wandb group name")
    p.add_argument(
        "--wandb-project",
        type=str,
        help="the wandb project name (specify this to enable wandb)",
        default="Consistency-GAN",
    )
    # p.add_argument(
    #     "--wandb-save-model", action="store_true", help="save model to wandb"
    # )
    args = p.parse_args()
    return args


if __name__ == "__main__":
    seed_everything(42)

    args = create_argparser()
    config = load_config(open(args.config))

    wandb.init(
        project=args.wandb_project,
        # entity=args.wandb_entity,
        # group=args.wandb_group,
        config=config,
        save_code=True,
    )

    main(config)
