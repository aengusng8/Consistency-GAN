import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from ..cm import dist_util, logger
from ..cm.fp16_util import MixedPrecisionTrainer
from ..cm.nn import update_ema
from ..cm.resample import LossAwareSampler, UniformSampler
from ..cm.train_util import (
    TrainLoop,
    find_resume_checkpoint,
    log_loss_dict,
    get_blob_logdir,
)

from ..cm.fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np


class ConsistencyGANTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        netD,
        target_netG,
        teacher_netG,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.netD = netD
        self.target_netG = target_netG
        self.teacher_netG = teacher_netG
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps

        if target_netG:
            self._load_and_sync_target_parameters()
            self.target_netG.requires_grad_(False)
            self.target_netG.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_netG.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        if teacher_netG:
            self._load_and_sync_teacher_parameters()
            self.teacher_netG.requires_grad_(False)
            self.teacher_netG.eval()

        self.global_step = self.step

        # Generator
        self.netG = self.ddp_model
        self.mp_trainerG = self.mp_trainer
        self.optimizerG = self.opt

        # Discriminator
        # TODO: create MixedPrecisionTrainer (EMA and resume, if needed) for netD
        # self.netD_mp_trainer = MixedPrecisionTrainer(
        #     model=self.netD,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=self.fp16_scale_growth,
        # )
        self.optimizerD = RAdam(
            self.netD.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        num_epoch = self.total_training_steps // self.microbatch
        self.schedulerD = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizerD, num_epoch, eta_min=1e-5
        )

        self.setup_the_loss_functions()

    def setup_the_loss_functions(self):
        # 1. Consistency Generator Loss
        if self.training_mode == "consistency_gan_distillation":
            self.compute_consistency_generator_loss = functools.partial(
                self.diffusion.consistency_generator_loss,
                self.ddp_model,
                target_netG=self.target_netG,
                teacher_netG=self.teacher_netG,
                teacher_diffusion=self.teacher_diffusion,
            )
        elif self.training_mode == "consistency_gan_training":
            self.compute_consistency_generator_loss = functools.partial(
                self.diffusion.consistency_generator_loss,
                self.ddp_model,
                target_netG=self.target_netG,
            )
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        # 2. Adversarial Generator Loss
        self.compute_adversarial_generator_loss = (
            self.diffusion.adversarial_generator_loss
        )

        # 3. Adversarial Discriminator Loss
        self.compute_adversarial_discriminator_loss = (
            self.diffusion.adversarial_discriminator_loss
        )

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_netG")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_target_checkpoint}..."
                )
                self.target_netG.load_state_dict(
                    dist_util.load_state_dict(
                        resume_target_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.target_netG.parameters())
        dist_util.sync_params(self.target_netG.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_netG")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_teacher_checkpoint}..."
                )
                self.teacher_netG.load_state_dict(
                    dist_util.load_state_dict(
                        resume_teacher_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.teacher_netG.parameters())
        dist_util.sync_params(self.teacher_netG.buffers())

    def run_loop(self):
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainerG.master_params,
                rate=target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def run_step(self, batch, cond):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            ema, num_scales = self.ema_scale_fn(self.global_step)

            # Start updating discriminator and generator
            losses = {}

            # 1. Consistency Generator Loss
            # 1.1 Compute the G's loss
            if last_batch or not self.use_ddp:
                consis_netG_loss = self.compute_consistency_generator_loss(
                    x_start=micro,
                    num_scales=num_scales,
                    netG_kwargs=micro_cond,
                )
            else:
                with self.ddp_model.no_sync():
                    consis_netG_loss = self.compute_consistency_generator_loss(
                        x_start=micro,
                        num_scales=num_scales,
                        netG_kwargs=micro_cond,
                    )

            consis_netG_loss = (consis_netG_loss * weights).mean()
            losses["Consistency Generator Loss"] = consis_netG_loss.item()

            # 1.2 Update G's parameters
            self.mp_trainerG.zero_grad()
            self.mp_trainerG.backward(consis_netG_loss)
            took_step = self.mp_trainerG.optimize(self.optimizerG)

            # 2. Adversarial Discriminator Loss
            # 2.1 Compute the D's loss
            for p in self.netD.parameters():
                p.requires_grad = True
            self.netD.zero_grad()

            (
                errD_real,
                grad_penalty,
                errD_fake,
            ) = self.compute_adversarial_discriminator_loss(
                netD=self.netD,
                netG=self.netG,
                netG_kwargs=micro_cond,
                x_start=micro,
                num_scales=num_scales,
                dims=micro.ndim,
                lazy_reg=self.lazy_reg,
                r1_gamma=self.r1_gamma,
                global_step=self.global_step,
                distiller=None,
            )

            errD_real *= weights
            errD_fake *= weights
            grad_penalty *= weights
            losses["Adversarial Discriminator Loss"] = (errD_real + errD_fake).item()

            # 2.2 Update D's parameters
            self.optimizerD.zero_grad()
            errD_real.backward(retain_graph=True)
            grad_penalty.backward()
            errD_fake.backward()
            self.optimizerD.step()

            for p in self.netD.parameters():
                p.requires_grad = False

            # 3. Adversarial Generator Loss
            # 3.1 Compute the G's loss
            if last_batch or not self.use_ddp:
                adver_netG_loss = self.compute_adversarial_generator_loss(
                    netD=self.netD,
                    netG=self.netG,
                    netG_kwargs=micro_cond,
                    x_start=micro,
                    num_scales=num_scales,
                    dims=micro.ndim,
                )
            else:
                with self.ddp_model.no_sync():
                    adver_netG_loss = self.compute_adversarial_generator_loss(
                        netD=self.netD,
                        netG=self.netG,
                        netG_kwargs=micro_cond,
                        x_start=micro,
                        num_scales=num_scales,
                        dims=micro.ndim,
                    )

            adver_netG_loss *= weights
            losses["Adversarial Generator Loss"] = adver_netG_loss.item()
            # 3.2 Update G's parameters
            self.mp_trainerG.zero_grad()
            self.mp_trainerG.backward(adver_netG_loss)
            _ = self.mp_trainerG.optimize(self.optimizerG)

            # NOTE, try: netG_loss = consis_netG_loss
            netG_loss = consis_netG_loss + adver_netG_loss

            # Weighting the losses
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, netG_loss.detach())

            if took_step:
                self._update_ema()
                if self.target_model:
                    self._update_target_ema()
                self.step += 1
                self.global_step += 1

            # logging
            log_loss_dict(self.diffusion, t, {k: v for k, v in losses.items()})
            self.log_step()

    def save(self):
        import blobfile as bf

        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainerG.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving netG {rate}...")
                if not rate:
                    filename = f"netG__{step:06d}.pt"
                else:
                    filename = f"ema-netG__{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        logger.log("saving optimizer state...")
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optimizerG{step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.optimizerG.state_dict(), f)

        if dist.get_rank() == 0:
            if self.target_netG:
                logger.log("saving target model state")
                filename = f"target_netG{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_netG.state_dict(), f)
            if self.teacher_netG and self.training_mode == "progdist":
                logger.log("saving teacher model state")
                filename = f"teacher_netG{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.teacher_netG.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but optimizerG/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainerG.master_params)
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)